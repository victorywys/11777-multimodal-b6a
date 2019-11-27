from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils

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

class GDiscriminator(nn.Module):
    def __init__(self, opt):
        super(GDiscriminator, self).__init__()
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

        self.core = LSTMCore(opt)
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)

        self.fc_linear = nn.Linear(self.fc_feat_size, self.sub_feat_size)
        self.fc_norm = nn.BatchNorm1d(self.sub_feat_size)
        self.att_linear = nn.Linear(self.fc_feat_size, self.sub_feat_size)
        self.att_norm = nn.BatchNorm1d(self.sub_feat_size)
        self.emb_linear = nn.Linear(self.rnn_size, self.sub_feat_size)
        self.emb_norm = nn.BatchNorm1d(self.sub_feat_size)
        self.after_linear = nn.Linear(self.sub_feat_size * 3, self.sub_feat_size)
        self.logits = nn.Linear(self.sub_feat_size, 3)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'lstm':
            return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                    weight.new_zeros(self.num_layers, bsz, self.rnn_size))
        else:
            return weight.new_zeros(self.num_layers, bsz, self.rnn_size)


    def forward(self, fc_feats, att_feats, seq, att_masks=None, use_prob=False, label_len=None):
        # if use_prob:
        #    seq: batch_size * seq_length
        # else:
        #    seq: batch_size * seq_length * vocab_size
        if use_prob:
            embeds = self.embed.weight # vocab_size * embedding_size
            batch_size, max_label_len, _ = seq.size()
        else:
            batch_size, max_label_len = seq.size()
        state = self.init_hidden(batch_size)
        if label_len is None:
            label_len = torch.sum(1 - torch.eq(seq, 0), 1)
        outputs = []
        for i in range(max_label_len):
            if use_prob:
                xt = torch.matmul(seq[:, i], embeds)
            else:
                xt = self.embed(seq[:, i])
            output, state = self.core(xt, state)
            outputs.append(output)
        sent_embed = []
        for i in range(batch_size):
            sent_embed.append(outputs[label_len[i]][i])
        emb = torch.stack(sent_embed)

        fc_after = F.dropout(F.relu(self.fc_norm(self.fc_linear(fc_feats))), p=0.1)
        att_after = F.dropout(F.relu(self.att_norm(self.att_linear(att_feats))), p=0.1)
        emb_after = F.dropout(F.relu(self.emb_norm(self.emb_linear(emb))), p=0.1)
        all_after = F.relu(self.after_linear(torch.cat((fc_after, att_after, emb_after), -1)))
        output = F.log_softmax(self.logits(all_after), dim=1)
        return output
