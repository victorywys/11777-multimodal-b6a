from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils
import copy

class EDiscriminator(nn.Module):
    def __init__(self, opt, res6=None):
        super(EDiscriminator, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.sub_feat_size = opt.fc_feat_size

        self.ss_prob = 0.0 # Schedule sampling probability

        self.feat_ind = [2048, 2048, 5, 2048, 25]
        self.res_dim = opt.res_dim
        self.res6_dim = opt.res6_dim
        self.dif_num = opt.dif_num
        c = copy.deepcopy

        if not res6 is None:
            self.cxt_enc = c(res6)
            self.ann_enc = c(res6)
            self.dif_ann_enc = c(res6)
        else:
            self.cxt_enc = nn.Linear(self.res_dim, self.res6_dim)
            self.ann_enc = nn.Linear(self.res_dim, self.res6_dim)
            self.dif_ann_enc = nn.Linear(self.res_dim, self.res6_dim)
        self.joint_enc = nn.Linear(self.res6_dim * 3 + 5 * (self.dif_num + 1), self.input_encoding_size)

        self.att_linear = nn.Linear(self.input_encoding_size, self.sub_feat_size)
        self.att_norm = nn.BatchNorm1d(self.sub_feat_size)
        self.emb_linear = nn.Linear(self.input_encoding_size, self.sub_feat_size)
        self.emb_norm = nn.BatchNorm1d(self.sub_feat_size)
        self.after_linear = nn.Linear(self.sub_feat_size * 2, self.sub_feat_size)
        self.logits = nn.Linear(self.sub_feat_size, 3)

    def forward(self, fc_feats, att_feats, emb, att_masks=None, init_norm=20):

        batch_size = fc_feats.size(0)

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
        att_feats = J = F.dropout(self.joint_enc(J), p=0.25).squeeze(1) # batch * 1 * input_encoding_size

        emb = emb.squeeze(1)

        att_after = F.dropout(F.relu(self.att_norm(self.att_linear(att_feats))), p=0.1)
        emb_after = F.dropout(F.relu(self.emb_norm(self.emb_linear(emb))), p=0.1)
        all_after = F.relu(self.after_linear(torch.cat((att_after, emb_after), -1)))
        output = F.log_softmax(self.logits(all_after), dim=1)
        return output

