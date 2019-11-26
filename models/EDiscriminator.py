from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils

class EDiscriminator(nn.Module):
    def __init__(self, opt):
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
        self.listener_embedding_size = opt.listener_embedding_size

        self.ss_prob = 0.0 # Schedule sampling probability

        self.fc_linear = nn.Linear(self.fc_feat_size, self.sub_feat_size)
        self.fc_norm = nn.BatchNorm1d(self.sub_feat_size)
        self.att_linear = nn.Linear(self.fc_feat_size, self.sub_feat_size)
        self.att_norm = nn.BatchNorm1d(self.sub_feat_size)
        self.emb_linear = nn.Linear(self.listener_embedding_size, self.sub_feat_size)
        self.emb_norm = nn.BatchNorm1d(self.sub_feat_size)
        self.after_linear = nn.Linear(self.sub_feat_size * 3, self.sub_feat_size)
        self.logits = nn.Linear(self.sub_feat_size, 3)

    def forward(self, fc_feats, att_feats, emb, att_masks=None):

        batch_size = fc_feats.size(0)
        fc_after = F.dropout(F.relu(self.fc_norm(self.fc_linear(fc_feats))), p=0.1)
        att_after = F.dropout(F.relu(self.att_norm(self.att_linear(att_feats))), p=0.1)
        emb_after = F.dropout(F.relu(self.emb_norm(self.emb_linear(emb))), p=0.1)
        all_after = F.relu(self.after_linear(torch.cat((fc_after, att_after, emb_after), -1)))
        output = F.log_softmax(self.logits(all_after), dim=1)
        return output

