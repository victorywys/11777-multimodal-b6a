from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

import time
import os
from six.moves import cPickle

import opts
import models
from dataloader import *
import eval_utils
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward

try:
    import tensorboardX as tb
except ImportError:
    print("tensorboardX is not installed")
    tb = None

def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)

def one_hot(inp, num, dtype=torch.float):
    a = torch.zeros_like(inp, dtype=dtype)
    a = a.unsqueeze(-1) * torch.zeros(num, dtype=dtype).to(a.device)
    out = a.scatter_(dim=-1,index=inp.unsqueeze(-1),value=1)
    return out

def pretrain(opt):
    # Deal with feature things before anything
    opt.use_att = utils.if_use_att(opt.caption_model)
    if opt.use_box: opt.att_feat_size = opt.att_feat_size + 5

    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    tb_summary_writer = tb and tb.SummaryWriter(opt.checkpoint_path)

    infos = {}
    histories = {}
    if opt.start_from is not None:
        # open old infos and check if models are compatible
        with open(os.path.join(opt.start_from, 'infos_'+opt.id+'.pkl')) as f:
            infos = cPickle.load(f)
            saved_model_opt = infos['opt']
            need_be_same=["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme

        if os.path.isfile(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')):
            with open(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')) as f:
                histories = cPickle.load(f)

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)

    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    use_device = [0, 1, 2, 3]

    if opt.caption_model == "SLD":
        tmp_opt = opt
        tmp_opt.caption_model = "speakerlistener"
        model = models.setup(tmp_opt).cuda()
        print(model)
        # tmp_opt.caption_model = "gdiscriminator"
        # GDismodel = models.setup(tmp_opt).cuda()
        # dp_GDismodel = torch.nn.DataParallel(GDismodel, device_ids=use_device)
        # tmp_opt.caption_model = "ediscriminator"
        # EDismodel = models.setup(tmp_opt).cuda()
        # dp_EDismodel = torch.nn.DataParallel(EDismodel, device_ids=use_device)
    else:
        model = models.setup(opt).cuda()
    #dp_model = model
    dp_model = torch.nn.DataParallel(model, device_ids=use_device)

    epoch_done = True
    # Assure in training mode
    dp_model.train()

    if opt.label_smoothing > 0:
        crit = utils.LabelSmoothing(smoothing=opt.label_smoothing)
    else:
        crit = utils.LanguageModelCriterion()
    speaker_hinge_crit = utils.SpeakerHingeCriterion()
    listener_hinge_crit = utils.ListenerHingeCriterion()
    rl_crit = utils.RewardCriterion()

    if opt.noamopt:
        assert opt.caption_model == 'transformer', 'noamopt can only work with transformer'
        optimizer = utils.get_std_opt(model, factor=opt.noamopt_factor, warmup=opt.noamopt_warmup)
        optimizer._step = iteration
    elif opt.reduce_on_plateau:
        optimizer = utils.build_optimizer(model.parameters(), opt)
        optimizer = utils.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    else:
        optimizer = utils.build_optimizer(model.parameters(), opt)
    # Load the optimizer
    if vars(opt).get('start_from', None) is not None and os.path.isfile(os.path.join(opt.start_from,"optimizer.pth")):
        optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))

    while True:
        if epoch_done:
            if not opt.noamopt and not opt.reduce_on_plateau:
                # Assign the learning rate
                if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                    frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                    decay_factor = opt.learning_rate_decay_rate  ** frac
                    opt.current_lr = opt.learning_rate * decay_factor
                else:
                    opt.current_lr = opt.learning_rate
                utils.set_lr(optimizer, opt.current_lr) # set the decayed rate
            # Assign the scheduled sampling prob
            if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                model.ss_prob = opt.ss_prob

            # If start self critical training
            if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                sc_flag = True
                init_scorer(opt.cached_tokens)
            else:
                sc_flag = False

            epoch_done = False

        start = time.time()
        # Load data from train split (0)
        data = loader.get_batch('train')
        print('Read data:', time.time() - start)

        torch.cuda.synchronize()
        start = time.time()

        tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks'], data['negative_att_feats'], data['negative_labels'], data['negative_masks']]
        tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
        fc_feats, att_feats, labels, masks, att_masks, negative_att_feats, negative_labels, negative_masks = tmp

        optimizer.zero_grad()
        if not sc_flag:
            speaker_gen, image_emb, seq_emb = dp_model(fc_feats, att_feats, labels, att_masks)
            negative_speaker_gen, negative_image_emb, _  = dp_model(fc_feats, negative_att_feats, labels, att_masks)
            negative_seq_emb, _ = dp_model(fc_feats, att_feats, negative_labels, att_masks, get_emb_only=True)

            lossS1 = crit(speaker_gen, labels[:,1:], masks[:,1:])
            lossS2 = speaker_hinge_crit(speaker_gen, negative_speaker_gen, labels[:,1:], masks[:,1:], negative_labels[:, 1:], negative_masks[:, 1:])
            lossL = listener_hinge_crit(image_emb, negative_image_emb, seq_emb, negative_seq_emb)

            loss = lossS1 + lossS2 + lossL
        else:
            gen_result, sample_logprobs = dp_model(fc_feats, att_feats, att_masks, opt={'sample_max':0}, mode='sample')
            reward = get_self_critical_reward(dp_model, fc_feats, att_feats, att_masks, data, gen_result, opt)
            loss = rl_crit(sample_logprobs, gen_result.data, torch.from_numpy(reward).float().cuda())

        loss.backward()
        utils.clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()
        train_loss = loss.item()
        speaker_loss1 = lossS1.item()
        speaker_loss2 = lossS2.item()
        listener_loss = lossL.item()
        torch.cuda.synchronize()
        end = time.time()
        if not sc_flag:
            print("iter {} (epoch {}), train_loss = {:.3f}, lossS1 = {:.3f}, lossS2 = {:.3f}, lossL = {:.3f}, time/batch = {:.3f}" \
                .format(iteration, epoch, train_loss, speaker_loss1, speaker_loss2, listener_loss, end - start))
        else:
            print("iter {} (epoch {}), avg_reward = {:.3f}, time/batch = {:.3f}" \
                .format(iteration, epoch, np.mean(reward[:,0]), end - start))

        # Update the iteration and epoch
        iteration += 1
        if data['bounds']['wrapped']:
            epoch += 1
            epoch_done = True

        # Write the training loss summary
        if (iteration % opt.losses_log_every == 0):
            add_summary_value(tb_summary_writer, 'train_loss', train_loss, iteration)
            if opt.noamopt:
                opt.current_lr = optimizer.rate()
            elif opt.reduce_on_plateau:
                opt.current_lr = optimizer.current_lr
            add_summary_value(tb_summary_writer, 'learning_rate', opt.current_lr, iteration)
            add_summary_value(tb_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
            if sc_flag:
                add_summary_value(tb_summary_writer, 'avg_reward', np.mean(reward[:,0]), iteration)

            loss_history[iteration] = train_loss if not sc_flag else np.mean(reward[:,0])
            lr_history[iteration] = opt.current_lr
            ss_prob_history[iteration] = model.ss_prob

        # make evaluation on validation set, and save model
        if (iteration % opt.save_checkpoint_every == 0):
            # eval model
            eval_kwargs = {'split': 'val',
                            'dataset': opt.input_json}
            eval_kwargs.update(vars(opt))
            val_loss, predictions, lang_stats = eval_utils.eval_split(dp_model, crit, loader, eval_kwargs)

            if opt.reduce_on_plateau:
                if opt.language_eval == 1:
                    if 'CIDEr' in lang_stats:
                        optimizer.scheduler_step(-lang_stats['CIDEr'])
                    else:
                        optimizer.scheduler_step(val_loss)
                else:
                    optimizer.scheduler_step(val_loss)

            # Write validation result into summary
            add_summary_value(tb_summary_writer, 'validation loss', val_loss, iteration)
            if opt.language_eval == 1:
                for k,v in lang_stats.items():
                    add_summary_value(tb_summary_writer, k, v, iteration)
            val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

            # Save model if is improving on validation result
            if opt.language_eval == 1:
                current_score = lang_stats['CIDEr']
            else:
                current_score = - val_loss

            best_flag = False
            if True: # if true
                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True
                checkpoint_path = os.path.join(opt.checkpoint_path, 'model.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
                optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer.pth')
                torch.save(optimizer.state_dict(), optimizer_path)

                # Dump miscalleous informations
                infos['iter'] = iteration
                infos['epoch'] = epoch
                infos['iterators'] = loader.iterators
                infos['split_ix'] = loader.split_ix
                infos['best_val_score'] = best_val_score
                infos['opt'] = opt
                infos['vocab'] = loader.get_vocab()

                histories['val_result_history'] = val_result_history
                histories['loss_history'] = loss_history
                histories['lr_history'] = lr_history
                histories['ss_prob_history'] = ss_prob_history
                with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'.pkl'), 'wb') as f:
                    cPickle.dump(infos, f)
                with open(os.path.join(opt.checkpoint_path, 'histories_'+opt.id+'.pkl'), 'wb') as f:
                    cPickle.dump(histories, f)

                if best_flag:
                    checkpoint_path = os.path.join(opt.checkpoint_path, 'model-best.pth')
                    torch.save(model.state_dict(), checkpoint_path)
                    print("model saved to {}".format(checkpoint_path))
                    with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'-best.pkl'), 'wb') as f:
                        cPickle.dump(infos, f)

        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break

def train_discriminator(opt):
    # Deal with feature things before anything
    opt.use_att = utils.if_use_att(opt.caption_model)
    if opt.use_box: opt.att_feat_size = opt.att_feat_size + 5

    tmp_opt_loader = opt
    tmp_opt_loader.batch_size = opt.batch_size // 3
    loader = DataLoader(tmp_opt_loader)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    tb_summary_writer = tb and tb.SummaryWriter(opt.checkpoint_path)

    infos = {}
    histories = {}
    if opt.start_from is not None:
        # open old infos and check if models are compatible
        with open(os.path.join(opt.start_from, 'infos_'+opt.id+'.pkl')) as f:
            infos = cPickle.load(f)
            saved_model_opt = infos['opt']
            need_be_same=["rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme

        if os.path.isfile(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')):
            with open(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')) as f:
                histories = cPickle.load(f)

    #iteration = infos.get('iter', 0)
#    epoch = infos.get('epoch', 0)

    iteration = 0
    epoch = 0

    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    use_device = [0, 1, 2]

    if opt.caption_model == "SLD":
        tmp_opt = opt
        tmp_opt.caption_model = "speakerlistener"
        model = models.setup(tmp_opt).cuda()
        print(model)
        # tmp_opt.caption_model = "gdiscriminator"
        # GDismodel = models.setup(tmp_opt).cuda()
        # dp_GDismodel = torch.nn.DataParallel(GDismodel, device_ids=use_device)
        # tmp_opt.caption_model = "ediscriminator"
        # EDismodel = models.setup(tmp_opt).cuda()
        # dp_EDismodel = torch.nn.DataParallel(EDismodel, device_ids=use_device)
    else:
        model = models.setup(opt).cuda()
    gen_model = model

    tmp_opt.caption_model = "gdiscriminator"
    tmp_opt.start_from = tmp_opt.discriminator_start_from
    print(tmp_opt)
    gd_model = models.setup(tmp_opt).cuda()

    tmp_opt = opt
    tmp_opt.caption_model = "ediscriminator"
    tmp_opt.start_from = tmp_opt.discriminator_start_from
    ed_model = models.setup(tmp_opt).cuda()

    dp_gd_model = torch.nn.DataParallel(gd_model, device_ids=use_device)
    dp_ed_model = torch.nn.DataParallel(ed_model, device_ids=use_device)
    #dp_gd_model = gd_model
    #dp_ed_model = ed_model

    epoch_done = True
    # Assure in training mode
    gen_model.eval()
    gd_model.train()
    ed_model.train()

    gd_crit = utils.GDiscriminatorCriterion()
    ed_crit = utils.EDiscriminatorCriterion()

    if opt.reduce_on_plateau:
        print(gd_model.parameters())
        print(ed_model.parameters())
        gd_optimizer = utils.build_optimizer(gd_model.parameters(), opt)
        ed_optimizer = utils.build_optimizer(ed_model.parameters(), opt)
        print(gd_optimizer)
        print(dir(gd_optimizer))
        gd_optimizer = utils.ReduceLROnPlateau(gd_optimizer, factor=0.5, patience=3)
        ed_optimizer = utils.ReduceLROnPlateau(ed_optimizer, factor=0.5, patience=3)
        print(gd_optimizer)
        print(dir(gd_optimizer))
    else:
        gd_optimizer = utils.build_optimizer(gd_model.parameters(), opt)
        ed_optimizer = utils.build_optimizer(ed_model.parameters(), opt)
    while True:
        if epoch_done:
            if not opt.noamopt and not opt.reduce_on_plateau:
                # Assign the learning rate
                if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                    frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                    decay_factor = opt.learning_rate_decay_rate  ** frac
                    opt.current_lr = opt.learning_rate * decay_factor
                else:
                    opt.current_lr = opt.learning_rate
                utils.set_lr(gd_optimizer, opt.current_lr) # set the decayed rate
                utils.set_lr(ed_optimizer, opt.current_lr)
            # Assign the scheduled sampling prob
            if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                model.ss_prob = opt.ss_prob

            # If start self critical training
            if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                sc_flag = True
                init_scorer(opt.cached_tokens)
            else:
                sc_flag = False

            epoch_done = False

        start = time.time()
        # Load data from train split (0)
        data = loader.get_batch('train')
        print('Read data:', time.time() - start)

        torch.cuda.synchronize()
        start = time.time()

        tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks'], data['negative_att_feats'], data['negative_labels'], data['negative_masks']]
        tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
        fc_feats, att_feats, labels, masks, att_masks, negative_att_feats, negative_labels, negative_masks = tmp

        gd_optimizer.zero_grad()
        ed_optimizer.zero_grad()

        speaker_gen, image_emb, seq_emb = gen_model(fc_feats, att_feats, labels, att_masks)
        _, negative_image_emb = gen_model(fc_feats, negative_att_feats, labels, att_masks, get_emb_only=True)

        speaker_gen = speaker_gen.detach()
        image_emb = image_emb.detach()
        seq_emb = seq_emb.detach()
        negative_image_emb = negative_image_emb.detach()

        speaker_label = torch.argmax(speaker_gen, -1)
        gd_data = []
        ed_data = []
        batch_size = labels.size()[0]
        for i in range(batch_size):
            gd_data.append((fc_feats[i, :], att_feats[i, :], labels[i, 1:], 0))
            gd_data.append((fc_feats[i, :], att_feats[i, :], negative_labels[i, 1:], 1))
            gd_data.append((fc_feats[i, :], att_feats[i, :], speaker_label[i, :], 2))
            ed_data.append((fc_feats[i, :], att_feats[i, :], seq_emb[i, :], 0))
            ed_data.append((fc_feats[i, :], att_feats[i, :], image_emb[i, :], 1))
            ed_data.append((fc_feats[i, :], att_feats[i, :], negative_image_emb[i, :], 2))
        random.shuffle(gd_data)
        random.shuffle(ed_data)
        gd_fc_feat = torch.stack([d[0] for d in gd_data])
        gd_att_feat = torch.stack([d[1] for d in gd_data])
        gd_seq = torch.stack([d[2] for d in gd_data])
        gd_label = torch.tensor([d[3] for d in gd_data]).cuda()

        ed_fc_feat = torch.stack([d[0] for d in ed_data])
        ed_att_feat = torch.stack([d[1] for d in ed_data])
        ed_emb = torch.stack([d[2] for d in ed_data])
        ed_label = torch.tensor([d[3] for d in ed_data]).cuda()

        gd_pred = dp_gd_model(gd_fc_feat, gd_att_feat, gd_seq)
        ed_pred = dp_ed_model(ed_fc_feat, ed_att_feat, ed_emb)
        lossG = gd_crit(gd_pred, gd_label)
        lossE = ed_crit(ed_pred, ed_label)

        lossG.backward()
        lossE.backward()

        utils.clip_gradient(gd_optimizer, opt.grad_clip)
        utils.clip_gradient(ed_optimizer, opt.grad_clip)
        gd_optimizer.step()
        ed_optimizer.step()
        gd_loss = lossG.item()
        ed_loss = lossE.item()
        train_loss = gd_loss + ed_loss
        torch.cuda.synchronize()
        end = time.time()
        if not sc_flag:
            print("iter {} (epoch {}), train_loss = {:.3f}, gd_loss = {:.3f}, ed_loss = {:.3f}, time/batch = {:.3f}" \
                .format(iteration, epoch, train_loss, gd_loss, ed_loss, end - start))
        else:
            print("iter {} (epoch {}), avg_reward = {:.3f}, time/batch = {:.3f}" \
                .format(iteration, epoch, np.mean(reward[:,0]), end - start))

        # Update the iteration and epoch
        iteration += 1
        if data['bounds']['wrapped']:
            epoch += 1
            epoch_done = True

        # Write the training loss summary
        if (iteration % opt.losses_log_every == 0):
            add_summary_value(tb_summary_writer, 'train_loss', train_loss, iteration)
            if opt.noamopt:
                opt.current_lr = gd_optimizer.rate()
            elif opt.reduce_on_plateau:
                opt.current_lr = gd_optimizer.current_lr
            add_summary_value(tb_summary_writer, 'learning_rate', opt.current_lr, iteration)
            add_summary_value(tb_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
            if sc_flag:
                add_summary_value(tb_summary_writer, 'avg_reward', np.mean(reward[:,0]), iteration)

            loss_history[iteration] = train_loss if not sc_flag else np.mean(reward[:,0])
            lr_history[iteration] = opt.current_lr
            ss_prob_history[iteration] = model.ss_prob

        # make evaluation on validation set, and save model
        if (iteration % opt.save_checkpoint_every == 0):
            tot_num = 0
            gd_true = 0
            ed_true = 0
            gd_model.eval()
            ed_model.eval()
            for i in range(40):
                # eval model
                data = loader.get_batch('val')
                print('Read data:', time.time() - start)

                torch.cuda.synchronize()
                start = time.time()

                tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks'], data['negative_att_feats'], data['negative_labels'], data['negative_masks']]
                tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
                fc_feats, att_feats, labels, masks, att_masks, negative_att_feats, negative_labels, negative_masks = tmp

                gd_optimizer.zero_grad()
                ed_optimizer.zero_grad()

                speaker_gen, image_emb, seq_emb = gen_model(fc_feats, att_feats, labels, att_masks)
                _, negative_image_emb = gen_model(fc_feats, negative_att_feats, labels, att_masks, get_emb_only=True)

                speaker_gen = speaker_gen.detach()
                image_emb = image_emb.detach()
                seq_emb = seq_emb.detach()
                negative_image_emb = negative_image_emb.detach()

                speaker_label = torch.argmax(speaker_gen, -1)
                gd_data = []
                ed_data = []
                batch_size = labels.size()[0]
                for i in range(batch_size):
                    gd_data.append((fc_feats[i, :], att_feats[i, :], labels[i, 1:], 0))
                    gd_data.append((fc_feats[i, :], att_feats[i, :], negative_labels[i, 1:], 1))
                    gd_data.append((fc_feats[i, :], att_feats[i, :], speaker_label[i, :], 2))
                    ed_data.append((fc_feats[i, :], att_feats[i, :], seq_emb[i, :], 0))
                    ed_data.append((fc_feats[i, :], att_feats[i, :], image_emb[i, :], 1))
                    ed_data.append((fc_feats[i, :], att_feats[i, :], negative_image_emb[i, :], 2))
                gd_fc_feat = torch.stack([d[0] for d in gd_data])
                gd_att_feat = torch.stack([d[1] for d in gd_data])
                gd_seq = torch.stack([d[2] for d in gd_data])
                gd_label = torch.tensor([d[3] for d in gd_data]).cuda()

                ed_fc_feat = torch.stack([d[0] for d in ed_data])
                ed_att_feat = torch.stack([d[1] for d in ed_data])
                ed_emb = torch.stack([d[2] for d in ed_data])
                ed_label = torch.tensor([d[3] for d in ed_data]).cuda()

                gd_pred = dp_gd_model(gd_fc_feat, gd_att_feat, gd_seq)
                ed_pred = dp_ed_model(ed_fc_feat, ed_att_feat, ed_emb)
                tot_num += batch_size * 3
                gd_true += torch.sum(torch.eq(torch.argmax(gd_pred, -1), gd_label)).item()
                ed_true += torch.sum(torch.eq(torch.argmax(ed_pred, -1), ed_label)).item()

            gd_acc = float(gd_true) / tot_num
            ed_acc = float(ed_true) / tot_num
            print("test %d cases on validation set, gd_acc: %.3f, ed_acc: %.3f" % (tot_num, gd_acc * 100, ed_acc * 100))
            gd_model.train()
            ed_model.train()

            if opt.reduce_on_plateau:
                if opt.language_eval == 1:
                    if 'CIDEr' in lang_stats:
                        gd_optimizer.scheduler_step(-lang_stats['CIDEr'])
                    else:
                        gd_optimizer.scheduler_step(-gd_acc)
                        ed_optimizer.scheduler_step(-ed_acc)
                else:
                   gd_optimizer.scheduler_step(-gd_acc)
                   ed_optimizer.scheduler_step(-ed_acc)

            # Save model if is improving on validation result
            current_score = gd_acc + ed_acc

            best_flag = False
            if True: # if true
                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True
                checkpoint_path = os.path.join(opt.checkpoint_path, 'gd_model.pth')
                torch.save(gd_model.state_dict(), checkpoint_path)
                checkpoint_path = os.path.join(opt.checkpoint_path, 'ed_model.pth')
                torch.save(ed_model.state_dict(), checkpoint_path)

                print("model saved to {}".format(checkpoint_path))
                optimizer_path = os.path.join(opt.checkpoint_path, 'gd_optimizer.pth')
                torch.save(gd_optimizer.state_dict(), optimizer_path)
                optimizer_path = os.path.join(opt.checkpoint_path, 'ed_optimizer.pth')
                torch.save(ed_optimizer.state_dict(), optimizer_path)



                # Dump miscalleous informations
                infos['iter'] = iteration
                infos['epoch'] = epoch
                infos['iterators'] = loader.iterators
                infos['split_ix'] = loader.split_ix
                infos['best_val_score'] = best_val_score
                infos['opt'] = opt
                infos['vocab'] = loader.get_vocab()

                histories['val_result_history'] = val_result_history
                histories['loss_history'] = loss_history
                histories['lr_history'] = lr_history
                histories['ss_prob_history'] = ss_prob_history
                with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'.pkl'), 'wb') as f:
                    cPickle.dump(infos, f)
                with open(os.path.join(opt.checkpoint_path, 'histories_'+opt.id+'.pkl'), 'wb') as f:
                    cPickle.dump(histories, f)

                if best_flag:
                    checkpoint_path = os.path.join(opt.checkpoint_path, 'model-best.pth')
                    torch.save(model.state_dict(), checkpoint_path)
                    print("model saved to {}".format(checkpoint_path))
                    with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'-best.pkl'), 'wb') as f:
                        cPickle.dump(infos, f)

        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break

def joint_training(opt):
    # Deal with feature things before anything
    opt.use_att = utils.if_use_att(opt.caption_model)
    if opt.use_box: opt.att_feat_size = opt.att_feat_size + 5

    tmp_opt_loader = opt
    tmp_opt_loader.batch_size = opt.batch_size // 3
    loader = DataLoader(tmp_opt_loader)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    tb_summary_writer = tb and tb.SummaryWriter(opt.checkpoint_path)

    infos = {}
    histories = {}
    if opt.start_from is not None:
        # open old infos and check if models are compatible
        with open(os.path.join(opt.start_from, 'infos_'+opt.id+'.pkl')) as f:
            infos = cPickle.load(f)
            saved_model_opt = infos['opt']
            need_be_same=["rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme

        if os.path.isfile(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')):
            with open(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')) as f:
                histories = cPickle.load(f)

    #iteration = infos.get('iter', 0)
#    epoch = infos.get('epoch', 0)

    iteration = 0
    epoch = 0

    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    use_device = [0, 1, 2, 3]

    if opt.caption_model == "SLD":
        tmp_opt = opt
        tmp_opt.caption_model = "speakerlistener"
        model = models.setup(tmp_opt).cuda()
        # tmp_opt.caption_model = "gdiscriminator"
        # GDismodel = models.setup(tmp_opt).cuda()
        # dp_GDismodel = torch.nn.DataParallel(GDismodel, device_ids=use_device)
        # tmp_opt.caption_model = "ediscriminator"
        # EDismodel = models.setup(tmp_opt).cuda()
        # dp_EDismodel = torch.nn.DataParallel(EDismodel, device_ids=use_device)
    else:
        model = models.setup(opt).cuda()
    gen_model = model
    print(gen_model)
    dp_gen_model = torch.nn.DataParallel(gen_model, device_ids=use_device)
#    dp_gen_model = gen_model

    tmp_opt = opt
    tmp_opt.caption_model = "discriminator"
    dis_model = models.setup(tmp_opt).cuda()
    print(dis_model)
    dp_dis_model = torch.nn.DataParallel(dis_model, device_ids=use_device)
#    dp_dis_model = dis_model

    epoch_done = True
    # Assure in training mode
    gen_model.train()
    dis_model.train()

    if opt.label_smoothing > 0:
        crit1 = utils.LabelSmoothing(smoothing=opt.label_smoothing)
    else:
        crit1 = utils.LanguageModelCriterion()
    crit2 = utils.SpeakerHingeCriterion()
    crit3 = utils.ListenerHingeCriterion()
    crit4 = utils.JointCriterion() # for Generation Discriminator
    crit5 = utils.JointCriterion() # for Embedding Discriminator
    gen_crit = (crit1, crit2, crit3, crit4, crit5)

    crit6 = utils.GDiscriminatorCriterion()
    crit7 = utils.EDiscriminatorCriterion()
    dis_crit = (crit6, crit7)

    if opt.reduce_on_plateau:
        print(gen_model.parameters())
        print(dis_model.parameters())
        gen_optimizer = utils.build_optimizer(gen_model.parameters(), opt)
        dis_optimizer = utils.build_optimizer(dis_model.parameters(), opt)
        gen_optimizer = utils.ReduceLROnPlateau(gen_optimizer, factor=0.5, patience=3)
        dis_optimizer = utils.ReduceLROnPlateau(dis_optimizer, factor=0.5, patience=3)
    else:
        gen_optimizer = utils.build_optimizer(gen_model.parameters(), opt)
        dis_optimizer = utils.build_optimizer(dis_model.parameters(), opt)

    while True:
        if epoch_done:
            if not opt.noamopt and not opt.reduce_on_plateau:
                # Assign the learning rate
                if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                    frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                    decay_factor = opt.learning_rate_decay_rate  ** frac
                    opt.current_lr = opt.learning_rate * decay_factor
                else:
                    opt.current_lr = opt.learning_rate
                utils.set_lr(gen_optimizer, opt.current_lr) # set the decayed rate
                utils.set_lr(dis_optimizer, opt.current_lr)
            # Assign the scheduled sampling prob
            if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                gen_model.ss_prob = opt.ss_prob

            # If start self critical training
            if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                sc_flag = True
                init_scorer(opt.cached_tokens)
            else:
                sc_flag = False

            epoch_done = False

        start = time.time()
        # Load data from train split (0)
        data = loader.get_batch('train')
        print('Read data:', time.time() - start)

        torch.cuda.synchronize()
        start = time.time()

        tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks'], data['negative_att_feats'], data['negative_labels'], data['negative_masks']]
        tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
        fc_feats, att_feats, labels, masks, att_masks, negative_att_feats, negative_labels, negative_masks = tmp

        gen_optimizer.zero_grad()
        dis_optimizer.zero_grad()

        speaker_gen, image_emb, seq_emb = dp_gen_model(fc_feats, att_feats, labels, att_masks, use_gumbel=True, hard_gumbel=False)
        speaker_gen, gumbel_output = speaker_gen
        negative_speaker_gen, negative_image_emb, _  = dp_gen_model(fc_feats, negative_att_feats, labels, att_masks)
        negative_seq_emb, negative_image_emb = dp_gen_model(fc_feats, negative_att_feats, labels, att_masks, get_emb_only=True)

        gd_data = []
        ed_data = []
        batch_size = labels.size()[0]
        for i in range(batch_size):
            gd_data.append((fc_feats[i, :], att_feats[i, :], one_hot(labels[i, 1:], opt.vocab_size + 1).to(torch.float), torch.sum(1 - torch.eq(labels[i, 1:], 0)), 0))
            gd_data.append((fc_feats[i, :], att_feats[i, :], one_hot(negative_labels[i, 1:], opt.vocab_size + 1).to(torch.float), torch.sum(1 - torch.eq(negative_labels[i, 1:], 0)), 1))
            gd_data.append((fc_feats[i, :], att_feats[i, :], gumbel_output[i, :], torch.sum(1 - torch.eq(torch.argmax(gumbel_output[i, :], -1), 0)), 2))
            ed_data.append((fc_feats[i, :], att_feats[i, :], seq_emb[i, :], 0))
            ed_data.append((fc_feats[i, :], att_feats[i, :], image_emb[i, :], 1))
            ed_data.append((fc_feats[i, :], att_feats[i, :], negative_image_emb[i, :], 2))
        random.shuffle(gd_data)
        random.shuffle(ed_data)
        gd_fc_feat = torch.stack([d[0] for d in gd_data])
        gd_att_feat = torch.stack([d[1] for d in gd_data])
        gd_seq = torch.stack([d[2] for d in gd_data])
        gd_seq_length = torch.tensor([d[3] for d in gd_data]).cuda()
        gd_label = torch.tensor([d[4] for d in gd_data]).cuda()

        ed_fc_feat = torch.stack([d[0] for d in ed_data])
        ed_att_feat = torch.stack([d[1] for d in ed_data])
        ed_emb = torch.stack([d[2] for d in ed_data])
        ed_label = torch.tensor([d[3] for d in ed_data]).cuda()

        gd_pred = dp_dis_model(gd_fc_feat, gd_att_feat, gd_seq, dis_mode="gd", use_prob=True, label_len=gd_seq_length)
        ed_pred = dp_dis_model(ed_fc_feat, ed_att_feat, ed_emb, dis_mode="ed")

        gen_loss = gen_crit[0](speaker_gen, labels[:, 1:], masks[:, 1:]) +\
                gen_crit[1](speaker_gen, negative_speaker_gen, labels[:, 1:], masks[:, 1:], negative_labels[:, 1:], negative_masks[:, 1:]) +\
                gen_crit[2](image_emb, negative_image_emb, seq_emb, negative_seq_emb) +\
                gen_crit[3](gd_pred) +\
                gen_crit[4](ed_pred)
        gen_loss.backward(retain_graph=True)
        utils.clip_gradient(gen_optimizer, opt.grad_clip)
        gen_optimizer.step()

        dis_loss = dis_crit[0](gd_pred, gd_label) + dis_crit[1](ed_pred, ed_label)
        dis_loss.backward()
        utils.clip_gradient(dis_optimizer, opt.grad_clip)
        dis_optimizer.step()

        G_loss = gen_loss.item()
        D_loss = dis_loss.item()
        train_loss = gen_loss + dis_loss
        torch.cuda.synchronize()
        end = time.time()
        if not sc_flag:
            print("iter {} (epoch {}), train_loss = {:.3f}, gen_loss = {:.3f}, dis_loss = {:.3f}, time/batch = {:.3f}" \
                .format(iteration, epoch, train_loss, gen_loss, dis_loss, end - start))
        else:
            print("iter {} (epoch {}), avg_reward = {:.3f}, time/batch = {:.3f}" \
                .format(iteration, epoch, np.mean(reward[:,0]), end - start))

        # Update the iteration and epoch
        iteration += 1
        if data['bounds']['wrapped']:
            epoch += 1
            epoch_done = True

        # Write the training loss summary
        if (iteration % opt.losses_log_every == 0):
            add_summary_value(tb_summary_writer, 'train_loss', train_loss, iteration)
            if opt.noamopt:
                opt.current_lr = gen_optimizer.rate()
            elif opt.reduce_on_plateau:
                opt.current_lr = gen_optimizer.current_lr
            add_summary_value(tb_summary_writer, 'learning_rate', opt.current_lr, iteration)
            add_summary_value(tb_summary_writer, 'scheduled_sampling_prob', gen_model.ss_prob, iteration)
            if sc_flag:
                add_summary_value(tb_summary_writer, 'avg_reward', np.mean(reward[:,0]), iteration)

            loss_history[iteration] = train_loss if not sc_flag else np.mean(reward[:,0])
            lr_history[iteration] = opt.current_lr
            ss_prob_history[iteration] = gen_model.ss_prob


        # TODO: evaluation
        # make evaluation on validation set, and save model
        if (iteration % opt.save_checkpoint_every == 0):
            # eval model
            eval_kwargs = {'split': 'val',
                            'dataset': opt.input_json}
            eval_kwargs.update(vars(opt))
            val_loss, predictions, lang_stats = eval_utils.eval_split(dp_gen_model, gen_crit[0], loader, eval_kwargs)

            if opt.reduce_on_plateau:
                if opt.language_eval == 1:
                    if 'CIDEr' in lang_stats:
                        gen_optimizer.scheduler_step(-lang_stats['CIDEr'])
                        dis_optimizer.scheduler_step(-lang_stats['CIDEr'])
                    else:
                        gen_optimizer.scheduler_step(val_loss)
                        dis_optimizer.scheduler_step(val_loss)
                else:
                    gen_optimizer.scheduler_step(val_loss)
                    dis_optimizer.scheduler_step(val_loss)

            # Write validation result into summary
            add_summary_value(tb_summary_writer, 'validation loss', val_loss, iteration)
            if opt.language_eval == 1:
                for k,v in lang_stats.items():
                    add_summary_value(tb_summary_writer, k, v, iteration)
            val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

            # Save model if is improving on validation result
            if opt.language_eval == 1:
                current_score = lang_stats['CIDEr']
            else:
                current_score = - val_loss

            best_flag = False
            if True: # if true
                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True
                checkpoint_path = os.path.join(opt.checkpoint_path, 'gen_model.pth')
                torch.save(gen_model.state_dict(), checkpoint_path)
                print("gen_model saved to {}".format(checkpoint_path))
                optimizer_path = os.path.join(opt.checkpoint_path, 'gen_optimizer.pth')
                torch.save(gen_optimizer.state_dict(), optimizer_path)

                checkpoint_path = os.path.join(opt.checkpoint_path, 'dis_model.pth')
                torch.save(dis_model.state_dict(), checkpoint_path)
                print("dis_model saved to {}".format(checkpoint_path))
                optimizer_path = os.path.join(opt.checkpoint_path, 'dis_optimizer.pth')
                torch.save(dis_optimizer.state_dict(), optimizer_path)


                # Dump miscalleous informations
                infos['iter'] = iteration
                infos['epoch'] = epoch
                infos['iterators'] = loader.iterators
                infos['split_ix'] = loader.split_ix
                infos['best_val_score'] = best_val_score
                infos['opt'] = opt
                infos['vocab'] = loader.get_vocab()

                histories['val_result_history'] = val_result_history
                histories['loss_history'] = loss_history
                histories['lr_history'] = lr_history
                histories['ss_prob_history'] = ss_prob_history
                with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'.pkl'), 'wb') as f:
                    cPickle.dump(infos, f)
                with open(os.path.join(opt.checkpoint_path, 'histories_'+opt.id+'.pkl'), 'wb') as f:
                    cPickle.dump(histories, f)

                if best_flag:
                    checkpoint_path = os.path.join(opt.checkpoint_path, 'model-best.pth')
                    torch.save(gen_model.state_dict(), checkpoint_path)
                    print("model saved to {}".format(checkpoint_path))
                    with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'-best.pkl'), 'wb') as f:
                        cPickle.dump(infos, f)

        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break

if __name__ == "__main__":
    opt = opts.parse_opt()
    print(opt.checkpoint_path)
    if opt.train_mode == "pretrain":
        pretrain(opt)
    elif opt.train_mode == "discriminator":
        train_discriminator(opt)
    elif opt.train_mode == "joint":
        joint_training(opt)
