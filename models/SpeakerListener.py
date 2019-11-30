from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import misc.utils as utils

import copy
import math
import numpy as np

from .CaptionModel import CaptionModel
from .AttModel import sort_pack_padded_sequence, pad_unsort_packed_sequence, pack_wrapper, AttModel
from .TransformerModel import *


class SpeakerListener(AttModel):
    def __init__(self, opt):
        super(SpeakerListener, self).__init__(opt)
        self.opt = opt
        # self.config = yaml.load(open(opt.config_file))
        # d_model = self.input_encoding_size # 512

        delattr(self, 'att_embed')
        self.att_embed = nn.Sequential(*(
                ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
                (nn.Linear(self.att_feat_size, self.input_encoding_size),
                 nn.ReLU(),
                 nn.Dropout(self.drop_prob_lm)) +
                ((nn.BatchNorm1d(self.input_encoding_size),) if self.use_bn == 2 else ())))

        delattr(self, 'embed')
        self.embed = lambda x: x
        delattr(self, 'fc_embed')
        self.fc_embed = lambda x: x
        delattr(self, 'logit')
        del self.ctx2att

        vocab = self.vocab_size + 1
        h = 8
        N = opt.num_layers
        d_model = opt.input_encoding_size
        d_ff = opt.rnn_size
        dropout = 0.1
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        self.speaker = EncoderDecoder(
            Encoder(EncoderLayer_2b(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn),
                                 c(ff), dropout), N),
            lambda x: x,  # nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
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
        memory = self.speaker.encode(att_feats, att_masks)

        return fc_feats[..., :1], att_feats[..., :1], memory, att_masks

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

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

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        att_feats, seq_encoder, seq, att_masks, seq_encoder_mask, seq_mask = \
            self._prepare_feature_forward(att_feats, att_masks, seq)

        img_feat = self.speaker.encode(att_feats, att_masks)
        out = self.speaker.decode(img_feat, att_masks, seq, seq_mask)
        outputs = self.speaker.generator(out)

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
