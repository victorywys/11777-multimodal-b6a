from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import misc.utils as utils

import sys
import copy
import math
import numpy as np

from .CaptionModel import CaptionModel
from .AttModel import sort_pack_padded_sequence, pad_unsort_packed_sequence, pack_wrapper, AttModel
from .TransformerModel import *


class SpeakerListener(AttModel):
    def __init__(self, opt, res6=None):
        super(SpeakerListener, self).__init__(opt)
        self.opt = opt
        # self.config = yaml.load(open(opt.config_file))
        # d_model = self.input_encoding_size # 512

        delattr(self, 'att_embed')
        # self.att_embed = nn.Sequential(*(
        #         ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
        #         (nn.Linear(self.att_feat_size, self.input_encoding_size),
        #          nn.ReLU(),
        #          nn.Dropout(self.drop_prob_lm)) +
        #         ((nn.BatchNorm1d(self.input_encoding_size),) if self.use_bn == 2 else ())))

        delattr(self, 'embed')
        self.embed = lambda x: x
        delattr(self, 'fc_embed')
        self.fc_embed = lambda x: x
        delattr(self, 'logit')
        del self.ctx2att
        c = copy.deepcopy

        self.feat_ind = [2048, 2048, 5, 2048, 25]
        self.res_dim = self.opt.res_dim
        self.res6_dim = self.opt.res6_dim
        self.dif_num = self.opt.dif_num
        # self.input_encoding_size = self.opt.res6_dim
        if res6 != None:
            self.cxt_enc = c(res6)
            self.ann_enc = c(res6)
            self.dif_ann_enc = c(res6)
        else:
            self.cxt_enc = nn.Linear(self.res_dim, self.res6_dim)
            self.ann_enc = nn.Linear(self.res_dim, self.res6_dim)
            self.dif_ann_enc = nn.Linear(self.res_dim, self.res6_dim)
        self.joint_enc = nn.Linear(self.res6_dim * 3 + 5 * (self.dif_num + 1), self.input_encoding_size)

        vocab = self.vocab_size + 1
        h = 8
        N = 2
        d_model = opt.input_encoding_size
        d_ff = opt.rnn_size
        dropout = 0.1
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        # self.speaker = EncoderDecoder(
        #     Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        #     Decoder(DecoderLayer(d_model, c(attn), c(attn),
        #                          c(ff), dropout), N),
        #     lambda x: x,  # nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        #     nn.Sequential(Embeddings(d_model, vocab), c(position)),
        #     Generator(d_model, vocab))
        self.speaker = EncoderDecoder(
            None,
            Decoder(DecoderLayer(d_model, c(attn), c(attn),
                                 c(ff), dropout), N),
            None,
            nn.Sequential(Embeddings(d_model, vocab), c(position)),
            Generator(d_model, vocab))
        self.listener = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            None,
            nn.Sequential(Embeddings(d_model, vocab), c(position)),
            None,
            None)

        self.d_model = opt.input_encoding_size
        for layers in [self.speaker, self.listener]:
            for p in layers.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def logit(self, x):  # unsafe way
        return self.speaker.generator.proj(x)

    def init_hidden(self, bsz):
        return None

    def _prepare_feature(self, fc_feats, att_feats, att_masks):

        att_feats, seq_encoder, seq, att_masks, seq_encoder_mask, seq_mask = self._prepare_feature_forward(att_feats, att_masks)
        # do not use transformer:
        memory = att_feats
        # if using transformer, keep the following line.
        # memory = self.speaker.encode(att_feats, att_masks)

        return fc_feats[..., :1], att_feats[..., :1], memory, att_masks

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None, init_norm=20):
        feats = att_feats
        cxt = self.cxt_enc(feats[:, :, :self.feat_ind[0]])
        ann = self.ann_enc(feats[:, :, sum(self.feat_ind[:1]):sum(self.feat_ind[:2])])
        loc = feats[:, :, sum(self.feat_ind[:2]):sum(self.feat_ind[:3])]
        diff_ann = self.dif_ann_enc(feats[:, :, sum(self.feat_ind[:3]):sum(self.feat_ind[:4])])
        diff_loc = feats[:, :, sum(self.feat_ind[:4]):]

        cxt = F.normalize(cxt, dim=2) * init_norm
        ann = F.normalize(ann, dim=2) * init_norm
        loc = F.normalize(loc + 1e-15, dim=2) * init_norm
        diff_ann = F.normalize(diff_ann, dim=2) * init_norm
        diff_loc = F.normalize(diff_loc + 1e-15, dim=2) * init_norm

        J = torch.cat([cxt, ann, loc, diff_ann, diff_loc], 2)
        att_feats = J = F.dropout(self.joint_enc(J), p=0.25) # batch * 1 * input_encoding_size

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        if seq is not None:
            seq_encoder = seq.clone()
            seq_mask_encoder = (seq_encoder.data > 0)
            seq_mask_encoder[:, 0] += 1
            seq_mask_encoder = seq_mask_encoder.unsqueeze(-2).float()

            # crop the last one
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] += 1

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_encoder = None
            seq_mask_encoder = None
            seq_mask = None

        return att_feats, seq_encoder, seq, att_masks, seq_mask_encoder, seq_mask

    def _forward(self, fc_feats, att_feats, seq, att_masks=None, use_gumbel=False):
        sys.stdout.flush()
        #att_feats: batch_size * 1 * (2048 + 2048 + 5 + 2048 + 25)
        #att_masks: batch_size * 1, all the elements are 1
        att_feats, seq_encoder, seq, att_masks, seq_encoder_mask, seq_mask = \
            self._prepare_feature_forward(att_feats, att_masks, seq)

        # do not use transformer for only 1 feature
        img_feat = att_feats
        # if use transformer, keep the line below:
        #img_feat = self.speaker.encode(att_feats, att_masks)
        out = self.speaker.decode(img_feat, att_masks, seq, seq_mask)
        outputs = self.speaker.generator(out, use_gumbel=use_gumbel)

        sent_feat = self.listener.encode(seq_encoder, seq_encoder_mask)
        return img_feat, sent_feat, outputs

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):
        """
        state = [ys.unsqueeze(0)]
        """
        if state is None:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        out = self.speaker.decode(memory, mask,
                                ys,
                                subsequent_mask(ys.size(1)).to(memory.device))
        return out[:, -1], [ys.unsqueeze(0)]
