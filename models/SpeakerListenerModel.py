from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils
from torch.nn.utils.rnn import pad_packed_sequence

from .CaptionModel import CaptionModel

class LSTMCore(nn.Module):
    def __init__(self, opt):
        super(LSTMCore, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.drop_prob_lm = opt.drop_prob_lm

        # Build a LSTM
        self.i2h = nn.Linear(self.input_encoding_size, 5 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.dropout = nn.Dropout(self.drop_prob_lm)

    def forward(self, xt, state):

        all_input_sums = self.i2h(xt) + self.h2h(state[0][-1])
        sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
        sigmoid_chunk = F.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)

        in_transform = torch.max(\
            all_input_sums.narrow(1, 3 * self.rnn_size, self.rnn_size),
            all_input_sums.narrow(1, 4 * self.rnn_size, self.rnn_size))
        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        next_h = out_gate * F.tanh(next_c)

        output = self.dropout(next_h)
        state = (next_h.unsqueeze(0), next_c.unsqueeze(0))
        return output, state

class SpeakerListenerModel(CaptionModel):
    def __init__(self, opt):
        super(SpeakerListenerModel, self).__init__()
        self.speaker = Speaker(opt)
        self.listener = Listener(opt)

        self.ss_prob = 0.0 # Schedule sampling probability

    def _forward(self, fc_feats, att_feats, seq, att_masks=None, get_emb_only=False, use_gumbel=False, hard_gumbel=False):
        if get_emb_only:
            label_emb = self.get_label_emb(seq)
            img_emb = self.speaker.get_img_embedding(fc_feats, att_feats, seq, att_masks)
            return label_emb, img_emb
        image_emb = self.speaker.get_img_embedding(fc_feats, att_feats, seq, att_masks)
        speaker_gen = self.speaker.forward_with_embedding(image_emb, fc_feats, att_feats, seq, att_masks, use_gumbel, hard_gumbel)
        seq_len = torch.sum(1 - torch.eq(seq, 0), 1) + 1
        label_emb = self.listener(seq, seq_len)
        return speaker_gen, image_emb, label_emb

    def _sample_beam(self, *args, **kwargs):
        return self.speaker._sample_beam(*args, **kwargs)

    def _sample(self, *args, **kwargs):
        return self.speaker._sample(*args, **kwargs)

    def get_label_emb(self, seq):
        seq_len = torch.sum(1 - torch.eq(seq, 0), 1) + 1
        label_emb = self.listener(seq, seq_len)
        return label_emb

class Listener(nn.Module):
    def __init__(self, opt):
        super(Listener, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self.core = LSTMCore(opt)
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length

        self.listener_num_layers = opt.listener_num_layers
        self.listener_embedding_size = opt.listener_embedding_size
        assert self.listener_num_layers > 0
        self.add_module("label_linear_0", nn.Linear(self.rnn_size, self.listener_embedding_size))
        self.add_module("label_bn_0", nn.BatchNorm1d(self.listener_embedding_size))
        for i in range(1, self.listener_num_layers):
            self.add_module("label_linear_%d" % i, nn.Linear(self.listener_embedding_size, self.listener_embedding_size))
            self.add_module("label_bn_%d" % i, nn.BatchNorm1d(self.listener_embedding_size))


    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'lstm':
            return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                    weight.new_zeros(self.num_layers, bsz, self.rnn_size))
        else:
            return weight.new_zeros(self.num_layers, bsz, self.rnn_size)

    def forward(self, label, label_len):
        batch_size, max_label_len = label.size()
        state = self.init_hidden(batch_size)
        outputs = []
        for i in range(max_label_len):
            xt = self.embed(label[:, i])
            output, state = self.core(xt, state)
            outputs.append(output)
        sent_embed = []
        for i in range(batch_size):
            sent_embed.append(outputs[label_len[i]][i])
        x = torch.stack(sent_embed)
        for i in range(self.listener_num_layers - 1):
            x = self.__getattr__("label_linear_%d" % i)(x)
            x = self.__getattr__("label_bn_%d" % i)(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.1)
        x = self.__getattr__("label_linear_%d" % (self.listener_num_layers - 1))(x)
        x = self.__getattr__("label_bn_%d" % (self.listener_num_layers - 1))(x)
        return x

class Speaker(CaptionModel):
    def __init__(self, opt):
        super(Speaker, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size

        self.ss_prob = 0.0 # Schedule sampling probability

        self.listener_num_layers = opt.listener_num_layers
        self.listener_embedding_size = opt.listener_embedding_size
        if self.listener_num_layers <= 0:
            raise Exception("The layers used to get the embedding must be a positive interger, but got %d instead." % self.listener_num_layers)

        self.add_module("img_bn_0", nn.BatchNorm1d(self.fc_feat_size * 2))
        self.add_module("img_linear_0", nn.Linear(self.fc_feat_size * 2, self.listener_embedding_size))
        for i in range(1, self.listener_num_layers):
            self.add_module("img_bn_%d" % i, nn.BatchNorm1d(self.listener_embedding_size))
            self.add_module("img_linear_%d" % i, nn.Linear(self.listener_embedding_size, self.listener_embedding_size))
        self.add_module("img_bn_2", nn.BatchNorm1d(self.listener_embedding_size))

        self.img_embed = nn.Linear(self.fc_feat_size + self.listener_embedding_size, self.input_encoding_size)
        self.core = LSTMCore(opt)
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)
        for i in range(self.listener_num_layers):
            self.__getattr__("img_linear_%d" % i).bias.data.fill_(0)
            self.__getattr__("img_linear_%d" % i).weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'lstm':
            return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                    weight.new_zeros(self.num_layers, bsz, self.rnn_size))
        else:
            return weight.new_zeros(self.num_layers, bsz, self.rnn_size)

    def get_img_embedding(self, fc_feats, att_feats, seq, att_masks=None):
        att_feats = att_feats.to(torch.float32)
        x = self.__getattr__("img_bn_0")(torch.cat((fc_feats, att_feats), -1))
        for i in range(self.listener_num_layers - 1):
            x = self.__getattr__("img_linear_%d" % i)(x)
            x = self.__getattr__("img_bn_%d" % (i + 1))(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.1)
        x = self.__getattr__("img_linear_%d" % (self.listener_num_layers - 1))(x)
        x = self.__getattr__("img_bn_%d" % (self.listener_num_layers))(x)
        return x

    def forward_with_embedding(self, xe, fc_feats, att_feats, seq, att_masks=None, use_gumbel=False, hard_gumbel=False):
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        outputs = []
        gumbel_outputs = []
        for i in range(seq.size(1)):
            if i == 0:
                xt = self.img_embed(torch.cat((fc_feats, xe), -1))
            else:
                if self.training and i >= 2 and self.ss_prob > 0.0: # otherwiste no need to sample
                    sample_prob = fc_feats.data.new(batch_size).uniform_(0, 1)
                    sample_mask = sample_prob < self.ss_prob
                    if sample_mask.sum() == 0:
                        it = seq[:, i-1].clone()
                    else:
                        sample_ind = sample_mask.nonzero().view(-1)
                        it = seq[:, i-1].data.clone()
                        #prob_prev = torch.exp(outputs[-1].data.index_select(0, sample_ind)) # fetch prev distribution: shape Nx(M+1)
                        #it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1))
                        prob_prev = torch.exp(outputs[-1].data) # fetch prev distribution: shape Nx(M+1)
                        it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                else:
                    it = seq[:, i-1].clone()
                # break if all the sequences end
                if i >= 2 and seq[:, i-1].sum() == 0:
                    break
                xt = self.embed(it)

            output, state = self.core(xt, state)
            logits = self.logit(output)
            output = F.log_softmax(logits, dim=1)
            outputs.append(output)
            if use_gumbel:
                gumbel_output = F.gumbel_softmax(logits, 0.25, hard=hard_gumbel)
                gumbel_outputs.append(gumbel_output)

        before_padding = torch.cat([_.unsqueeze(1) for _ in outputs[1:]], 1).contiguous()
        batch_len = before_padding.size(1)
        if batch_len != self.seq_length + 1:
            ret = torch.cat((before_padding, torch.zeros(batch_size, self.seq_length - batch_len + 1, self.vocab_size + 1).cuda()), 1)
#            print(ret)
        else:
            ret = before_padding
        if use_gumbel:
            gumbel_before_padding = torch.cat([_.unsqueeze(1) for _ in gumbel_outputs[1:]], 1).contiguous()
            batch_len = before_padding.size(1)
            if batch_len != self.seq_length + 1:
                gumbel_ret = torch.cat((gumbel_before_padding, torch.zeros(batch_size, self.seq_length - batch_len + 1, self.vocab_size + 1).cuda()), 1)
            else:
                gumbel_ret = gumbel_before_padding
#            print("ret:")
#            for j in range(ret.size(1)):
#                print(torch.argmax(ret, -1)[0,j], torch.exp(ret[0,j,torch.argmax(ret, -1)[0,j]]))
#            print("gumbel_ret:")
#            for j in range(ret.size(1)):
#                print(torch.argmax(gumbel_ret, -1)[0, j], gumbel_ret[0,j,torch.argmax(gumbel_ret, -1)[0,j]])
            return ret, gumbel_ret
        return ret

    def _forward(self, fc_feats, att_feats, seq, att_masks=None, use_gumbel=False, hard_gumbel=False):
        xe = self.get_img_embedding(fc_feats, att_feats, seq, att_masks)
        ret = self.foward_with_embedding(xe, fc_feats, att_feats, seq, att_masks, use_gumbel, hard_gumbel)
        return ret

    def get_logprobs_state(self, it, state):
        # 'it' is contains a word index
        xt = self.embed(it)

        output, state = self.core(xt, state)
        logprobs = F.log_softmax(self.logit(output), dim=1)

        return logprobs, state

    def _sample_beam(self, fc_feats, att_feats, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        xe = self.get_img_embedding(fc_feats, att_feats, None)
        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            for t in range(2):
                if t == 0:
                    xt = self.img_embed(torch.cat((fc_feats[k:k+1], xe[k:k+1]), -1)).expand(beam_size, self.input_encoding_size)
                elif t == 1: # input <bos>
                    it = fc_feats.data.new(beam_size).long().zero_()
                    xt = self.embed(it)

                output, state = self.core(xt, state)
                logprobs = F.log_softmax(self.logit(output), dim=1)
            self.done_beams[k] = self.beam_search(state, logprobs, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def _sample(self, fc_feats, att_feats, att_masks=None, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        if beam_size > 1:
            return self.sample_beam(fc_feats, att_feats, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        seq = fc_feats.new_zeros(batch_size, self.seq_length, dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size, self.seq_length)
        xe = self.get_img_embedding(fc_feats, att_feats, None)
        for t in range(self.seq_length + 2):
            if t == 0:
                xt = self.img_embed(torch.cat((fc_feats, xe), -1))
            else:
                if t == 1: # input <bos>
                    it = fc_feats.data.new(batch_size).long().zero_()
                xt = self.embed(it)

            output, state = self.core(xt, state)
            logprobs = F.log_softmax(self.logit(output), dim=1)

            # sample the next_word
            if t == self.seq_length + 1: # skip if we achieve maximum length
                break
            if sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data).cpu() # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()
                it = torch.multinomial(prob_prev, 1).cuda()
                sampleLogprobs = logprobs.gather(1, it) # gather the logprobs at sampled positions
                it = it.view(-1).long() # and flatten indices for downstream processing

            if t >= 1:
                # stop when all finished
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                it = it * unfinished.type_as(it)
                seq[:,t-1] = it #seq[t] the input of t+2 time step
                seqLogprobs[:,t-1] = sampleLogprobs.view(-1)
                if unfinished.sum() == 0:
                    break

        return seq, seqLogprobs
