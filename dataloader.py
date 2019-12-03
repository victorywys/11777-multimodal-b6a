from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
import os
import numpy as np
import random

import torch
import torch.utils.data as data

import multiprocessing

class DataLoader(data.Dataset):

    def reset_iterator(self, split):
        del self._prefetch_process[split]
        self._prefetch_process[split] = BlobFetcher(split, self, split=='train')
        self.iterators[split] = 0

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.seq_per_img = opt.seq_per_img

        # feature related options
        self.use_att = getattr(opt, 'use_att', True)
        self.use_box = getattr(opt, 'use_box', 0)
        self.norm_att_feat = getattr(opt, 'norm_att_feat', 0)
        self.norm_box_feat = getattr(opt, 'norm_box_feat', 0)

        # load the json file which contains additional information about the dataset
        print('DataLoader loading json file: ', opt.input_json)
        self.info = json.load(open(self.opt.input_json))
        self.ix_to_word = self.info['ix_to_word']
        self.vocab_size = len(self.ix_to_word)
        print('vocab size is ', self.vocab_size)

        # open the hdf5 file
        print('DataLoader loading h5 file: ', opt.input_fc_dir, opt.input_att_dir, opt.input_box_dir, opt.input_label_h5)
        self.h5_label_file = h5py.File(self.opt.input_label_h5, 'r', driver='core')

        self.input_fc_dir = self.opt.input_fc_dir
        self.input_att_dir = self.opt.input_att_dir
        self.input_box_dir = self.opt.input_box_dir

        # REFCOCO
        self.ann_feats = np.load(self.opt.ref_ann_feats)
        self.image_feats = np.load(self.opt.ref_image_feats)
        self.ref_infos = json.load(open(self.opt.ref_infos))

        self.ann2idx = {item['ann_id']: i for i, item in enumerate(self.ref_infos['anns'])}
        self.img2idx = {item['image_id']: i for i, item in enumerate(self.ref_infos['images'])}
        self.ref2ann = {item['ref_id']: item['ann_id'] for item in self.ref_infos['refs']}
        self.ann2ref = {v: k for k, v in self.ref2ann.items()}
        self.ref2img = {item['ref_id']: item['image_id'] for item in self.ref_infos['refs']}
        self.img_info = {item['image_id']: item for item in self.ref_infos['images']}
        self.ref_info = {item['ref_id']: item for item in self.ref_infos['refs']}
        self.ann_info = {item['ann_id']: item for item in self.ref_infos['anns']}
        self.ref2infoidx = {item['id']: ix for ix, item in enumerate(self.info['images'])}

        # load in the sequence data
        seq_size = self.h5_label_file['labels'].shape
        self.seq_length = seq_size[1]
        print('max sequence length in data is', self.seq_length)
        # load the pointers in full to RAM (should be small enough)
        self.label_start_ix = self.h5_label_file['label_start_ix'][:]
        self.label_end_ix = self.h5_label_file['label_end_ix'][:]
        self.img_start_ix = self.h5_label_file['img_start_ix'][:]
        self.img_end_ix = self.h5_label_file['img_end_ix'][:]

        self.num_images = self.label_start_ix.shape[0]
        print('read %d image features' %(self.num_images))

        self.negative_sample_ix = []
        for i in range(self.num_images):
            self.negative_sample_ix.append([])
            for j in range(self.img_start_ix[i], self.img_end_ix[i] + 1):
                if j != i:
                    self.negative_sample_ix[-1].append(j)
            if (len(self.negative_sample_ix[-1]) == 0):
                for _ in range(3):
                    r = random.randint(1, len(self.img_start_ix)) - 1
                    while r == i:
                        r = random.randint(1, len(self.img_start_ix)) - 1
                    self.negative_sample_ix[-1].append(r)

        # separate out indexes for each of the provided splits
        self.split_ix = {'train': [], 'val': [], 'test': [], 'testA': [], 'testB': []}
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if img['split'] == 'train':
                self.split_ix['train'].append(ix)
            elif img['split'] == 'val':
                self.split_ix['val'].append(ix)
            elif img['split'] == 'test':
                self.split_ix['test'].append(ix)
            elif img['split'] == 'testA':
                self.split_ix['testA'].append(ix)
            elif img['split'] == 'testB':
                self.split_ix['testB'].append(ix)
            elif opt.train_only == 0: # restval
                self.split_ix['train'].append(ix)

        print('assigned %d images to split train' %len(self.split_ix['train']))
        print('assigned %d images to split val' %len(self.split_ix['val']))
        print('assigned %d images to split test' %len(self.split_ix['test']))
        print('assigned %d images to split testA' % len(self.split_ix['testA']))
        print('assigned %d images to split testB' % len(self.split_ix['testB']))

        self.iterators = {'train': 0, 'val': 0, 'test': 0, 'testA': 0, 'testB': 0}

        self._prefetch_process = {} # The three prefetch process
        for split in self.iterators.keys():
            self._prefetch_process[split] = BlobFetcher(split, self, split=='train')
            # Terminate the child process when the parent exists
        def cleanup():
            print('Terminating BlobFetcher')
            for split in self.iterators.keys():
                del self._prefetch_process[split]
        import atexit
        atexit.register(cleanup)

    def get_captions(self, ix, seq_per_img):
        # fetch the sequence labels
        ix1 = self.label_start_ix[ix] - 1 #label_start_ix starts from 1
        ix2 = self.label_end_ix[ix] - 1
        ncap = ix2 - ix1 + 1 # number of captions available for this image
        assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'

        if ncap < seq_per_img:
            # we need to subsample (with replacement)
            seq = np.zeros([seq_per_img, self.seq_length], dtype = 'int')
            for q in range(seq_per_img):
                ixl = random.randint(ix1,ix2)
                seq[q, :] = self.h5_label_file['labels'][ixl, :self.seq_length]
        else:
            ixl = random.randint(ix1, ix2 - seq_per_img + 1)
            seq = self.h5_label_file['labels'][ixl: ixl + seq_per_img, :self.seq_length]
        return seq

    def sample_neg_ids(self, pos_ann_id, seq_per_img, sample_ratio=0.5):
        neg_ann_ids, neg_ref_ids = [], []
        st_ref_ids, st_ann_ids, dt_ref_ids, dt_ann_ids = self.fetch_neighbour_ids(pos_ann_id)
        for k in range(seq_per_img):
            ### ann
            if len(st_ann_ids) > 0 and random.random() < sample_ratio:
                ix = random.randint(0, len(st_ann_ids) - 1)
                neg_ann_id = st_ann_ids[ix]
            elif len(dt_ann_ids) > 0:
                ix = random.randint(0, len(dt_ann_ids) - 1)
                neg_ann_id = dt_ann_ids[ix]
            else:
                ix = random.randint(0, len(self.ann_info) - 1)
                neg_ann_id = list(self.ann_info.keys())[ix]
            neg_ann_ids.append(neg_ann_id)
            ### ref
            if len(st_ref_ids) > 0 and random.random() < sample_ratio:
                ix = random.randint(0, len(st_ref_ids) - 1)
                neg_ref_id = st_ref_ids[ix]
            elif len(dt_ref_ids) > 0:
                ix = random.randint(0, len(dt_ref_ids) - 1)
                neg_ref_id = dt_ref_ids[ix]
            else:
                ix = random.randint(0, len(self.ref_info) - 1)
                neg_ref_id = list(self.ref_info.keys())[ix]
            infoidx = self.ref2infoidx[neg_ref_id]
            neg_ref_ids.append(infoidx)
        return neg_ann_ids, neg_ref_ids

    def fetch_neighbour_ids(self, ref_ann_id):
        # same image differnt id
        # same/different category and with/without ref annotation -- 4 combination
        ref_ann = self.ann_info[ref_ann_id]
        x, y, w, h = ref_ann['box']
        rx, ry = x + w / 2, y + h / 2

        def calc_distance_from_target(ann_id):
            x, y, w, h = self.ann_info[ann_id]['box']
            ax, ay = x + w / 2, y + h / 2
            return (rx - ax) ** 2 + (ry - ay) ** 2

        image = self.img_info[ref_ann['image_id']]
        ann_ids = image['ann_ids']
        ann_ids = sorted([[ann_id, calc_distance_from_target(ann_id)] for ann_id in ann_ids], key=lambda x: x[1])
        ann_ids = [ann_id[0] for ann_id in ann_ids]
        st_ref_ids, st_ann_ids, dt_ref_ids, dt_ann_ids = [], [], [], []
        for ann_id in ann_ids:
            if ann_id != ref_ann_id:
                if self.ann_info[ann_id]['category_id'] == ref_ann['category_id']:
                    st_ann_ids.append(ann_id)
                    if ann_id in self.ann2ref:
                        st_ref_ids.append(self.ann2ref[ann_id])
                else:
                    dt_ann_ids.append(ann_id)
                    if ann_id in self.ann2ref:
                        dt_ref_ids.append(self.ann2ref[ann_id])

        return st_ref_ids, st_ann_ids, dt_ref_ids, dt_ann_ids

    def fetch_dif_feats(self, ann_id):
        dif_ann_feats = np.zeros((2048), dtype=np.float32)
        dif_lfeats = np.zeros((5 * 5), dtype=np.float32)
        _, st_ann_ids, _, _ = self.fetch_neighbour_ids(ann_id)
        st_ann_ids = st_ann_ids[:5]
        if len(st_ann_ids) != 0:
            cand_ann_feats = self.ann_feats[[self.ann2idx[st_id_] for st_id_ in st_ann_ids]]
            ref_ann_feat = self.ann_feats[self.ann2idx[ann_id]].reshape(1, -1)
            dif_ann_feat = np.mean(cand_ann_feats - ref_ann_feat, axis=0)
            rbox = self.ann_info[ann_id]['box']
            rcx, rcy, rw, rh = rbox[0] + rbox[2] / 2, rbox[1] + rbox[3] / 2, rbox[2], rbox[3]
            dif_lfeat = []
            for st_ann_id in st_ann_ids:
                cbox = self.ann_info[st_ann_id]['box']
                cx1, cy1, cw, ch = cbox[0], cbox[1], cbox[2], cbox[3]
                dif_lfeat.extend(
                    [(cx1 - rcx) / rw, (cy1 - rcy) / rh, (cx1 + cw - rcx) / rw, (cy1 + ch - rcy) / rh, cw * ch / (rw * rh)])
            dif_ann_feats = dif_ann_feat
            dif_lfeats[:len(dif_lfeat)] = dif_lfeat
        return dif_ann_feats, dif_lfeats

    def fetch_feats(self, ann_id):
        ann_info = self.ann_info[ann_id]
        img_id = ann_info['image_id']
        ann_feats = self.ann_feats[self.ann2idx[ann_id]]
        ctx_feats = self.image_feats[self.img2idx[img_id]]
        x, y, w, h = ann_info['box']
        iw, ih = self.img_info[img_id]['width'], self.img_info[img_id]['height']
        l_feats = np.array([x / iw, y / ih, (x + w - 1) / iw, (y + h - 1) / ih, w * h / (iw * ih)])
        df, dlf = self.fetch_dif_feats(ann_id)
        tmp_att = np.hstack((ctx_feats, ann_feats, l_feats, df, dlf)).reshape((1, -1))
        return tmp_att

    def get_batch(self, split, batch_size=None, seq_per_img=None):
        batch_size = batch_size or self.batch_size
        seq_per_img = seq_per_img or self.seq_per_img

        fc_batch = [] # np.ndarray((batch_size * seq_per_img, self.opt.fc_feat_size), dtype = 'float32')
        att_batch = [] # np.ndarray((batch_size * seq_per_img, self.opt.att_feat_size), dtype = 'float32')
        negative_fc_batch = []
        negative_att_batch = []
        label_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype = 'int')
        mask_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype = 'float32')
        negative_label_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype = 'int')
        negative_mask_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype = 'float32')

        wrapped = False

        infos = []
        gts = []

        for i in range(batch_size):
            # fetch image
            _, _, ix, tmp_wrapped = self._prefetch_process[split].get()
            ref_id = self.info['images'][ix]['id']
            ann_id = self.ref2ann[ref_id]
            img_id = self.ref2img[ref_id]

            tmp_fc = self.image_feats[img_id]
            fc_batch.append(tmp_fc)

            tmp_att = self.fetch_feats(ann_id)
            att_batch.append(tmp_att)

            label_batch[i * seq_per_img : (i + 1) * seq_per_img, 1 : self.seq_length + 1] = self.get_captions(ix, seq_per_img)

            neg_ann_ids, neg_ref_ids = self.sample_neg_ids(ann_id, seq_per_img)
            for neg_ann_id in neg_ann_ids:
                # no use
                negative_fc_batch.append(tmp_fc)
                negative_att_batch.append(self.fetch_feats(neg_ann_id))
            # not ref ids, but ix in self.info['images']
            for id, neg_ix in enumerate(neg_ref_ids):
                negative_label_batch[i * seq_per_img + id: i * seq_per_img + id + 1, 1: self.seq_length + 1] = self.get_captions(neg_ix, 1)

            if tmp_wrapped:
                wrapped = True

            # Used for reward evaluation
            gts.append(self.h5_label_file['labels'][self.label_start_ix[ix] - 1: self.label_end_ix[ix]])

            # record associated info as well
            info_dict = {}
            info_dict['ix'] = ix
            info_dict['id'] = self.info['images'][ix]['id']
            info_dict['file_path'] = self.info['images'][ix]['file_path']
            infos.append(info_dict)

        # #sort by att_feat length
        # fc_batch, att_batch, label_batch, gts, infos = \
        #     zip(*sorted(zip(fc_batch, att_batch, np.vsplit(label_batch, batch_size), gts, infos), key=lambda x: len(x[1]), reverse=True))
        max_att_len = max([_.shape[0] for _ in negative_att_batch])
        negative_att_batch_tmp = np.zeros(
            [len(att_batch) * seq_per_img, max_att_len, negative_att_batch[0].shape[1]],
            dtype='float32')
        for i in range(len(negative_att_batch)):
            negative_att_batch_tmp[i * seq_per_img:(i + 1) * seq_per_img, :negative_att_batch[i].shape[0]] = negative_att_batch[i]

        fc_batch, att_batch, negative_fc_batch, negative_att_batch_tmp, label_batch, negative_label_batch, gts, infos = \
            zip(*sorted(zip(fc_batch, att_batch, negative_fc_batch, np.vsplit(negative_att_batch_tmp, batch_size), np.vsplit(label_batch, batch_size), np.vsplit(negative_label_batch, batch_size), gts, infos), key=lambda x: 0, reverse=True))
        data = {}
        data['fc_feats'] = np.stack(reduce(lambda x, y: x + y, [[_] * seq_per_img for _ in fc_batch]))
        # merge att_feats
        max_att_len = max([_.shape[0] for _ in att_batch])
        data['att_feats'] = np.zeros([len(att_batch) * seq_per_img, max_att_len, att_batch[0].shape[1]],
                                     dtype='float32')
        for i in range(len(att_batch)):
            data['att_feats'][i * seq_per_img:(i + 1) * seq_per_img, :att_batch[i].shape[0]] = att_batch[i]
        data['att_masks'] = np.zeros(data['att_feats'].shape[:2], dtype='float32')
        for i in range(len(att_batch)):
            data['att_masks'][i * seq_per_img:(i + 1) * seq_per_img, :att_batch[i].shape[0]] = 1
        # set att_masks to None if attention features have same length
        # if data['att_masks'].sum() == data['att_masks'].size:
        #     data['att_masks'] = None

        data['negative_fc_feats'] = np.stack(reduce(lambda x, y: x + y, [[_] * seq_per_img for _ in negative_fc_batch]))
        # merge att_feats
        # max_att_len = max([_.shape[0] for _ in negative_att_batch])
        data['negative_att_feats'] = np.vstack(negative_att_batch_tmp)
        data['negative_att_masks'] = np.zeros(data['negative_att_feats'].shape[:2], dtype='float32')
        for i in range(len(negative_att_batch)):
            data['negative_att_masks'][i, :negative_att_batch[i].shape[0]] = 1
        # set att_masks to None if attention features have same length
        # if data['negative_att_masks'].sum() == data['negative_att_masks'].size:
        #     data['negative_att_masks'] = None

        data['labels'] = np.vstack(label_batch)
        # generate mask
        nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 2, data['labels'])))
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1
        data['masks'] = mask_batch

        data['negative_labels'] = np.vstack(negative_label_batch)
        nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 2, data['negative_labels'])))
        for ix, row in enumerate(negative_mask_batch):
            row[:nonzeros[ix]] = 1
        data['negative_masks'] = negative_mask_batch

        data['gts'] = gts  # all ground truth captions of each images
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': len(self.split_ix[split]), 'wrapped': wrapped}
        data['infos'] = infos

        return data

    # It's not coherent to make DataLoader a subclass of Dataset, but essentially, we only need to implement the following to functions,
    # so that the torch.utils.data.DataLoader can load the data according the index.
    # However, it's minimum change to switch to pytorch data loading.
    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        ix = index #self.split_ix[index]
        return (None, None, ix)
        # return (np.load(os.path.join(self.input_fc_dir, str(self.info['images'][ix]['id']) + '.npy')),
        #         np.load(os.path.join(self.input_att_dir, str(self.info['images'][ix]['id']) + '.npz'))['feat'],
        #         ix)

    def __len__(self):
        return len(self.info['images'])

class SubsetSampler(torch.utils.data.sampler.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

class BlobFetcher():
    """Experimental class for prefetching blobs in a separate process."""
    def __init__(self, split, dataloader, if_shuffle=False):
        """
        db is a list of tuples containing: imcrop_name, caption, bbox_feat of gt box, imname
        """
        self.split = split
        self.dataloader = dataloader
        self.if_shuffle = if_shuffle

    # Add more in the queue
    def reset(self):
        """
        Two cases for this function to be triggered:
        1. not hasattr(self, 'split_loader'): Resume from previous training. Create the dataset given the saved split_ix and iterator
        2. wrapped: a new epoch, the split_ix and iterator have been updated in the get_minibatch_inds already.
        """
        # batch_size is 1, the merge is done in DataLoader class
        self.split_loader = iter(data.DataLoader(dataset=self.dataloader,
                                            batch_size=1,
                                            sampler=SubsetSampler(self.dataloader.split_ix[self.split][self.dataloader.iterators[self.split]:]),
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=4, # 4 is usually enough
                                            collate_fn=lambda x: x[0]))

    def _get_next_minibatch_inds(self):
        max_index = len(self.dataloader.split_ix[self.split])
        wrapped = False

        ri = self.dataloader.iterators[self.split]
        ix = self.dataloader.split_ix[self.split][ri]

        ri_next = ri + 1
        if ri_next >= max_index:
            ri_next = 0
            if self.if_shuffle:
                random.shuffle(self.dataloader.split_ix[self.split])
            wrapped = True
        self.dataloader.iterators[self.split] = ri_next

        return ix, wrapped

    def get(self):
        if not hasattr(self, 'split_loader'):
            self.reset()

        ix, wrapped = self._get_next_minibatch_inds()
        tmp = self.split_loader.next()
        if wrapped:
            self.reset()

        assert tmp[2] == ix, "ix not equal"

        return tmp + [wrapped]
