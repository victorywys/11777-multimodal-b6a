import torch
import torch.nn as nn
import torch.nn.functional as F
from const import global_consts as gc

class Net(nn.Module):
    def __init__(self, pretrained_vector):
        super(Net, self).__init__()
        self.reduceLinear = nn.Linear(gc.input_dim, gc.reduce_dim)
        self.inputLinear = nn.Linear(gc.input_dim, gc.cell_dim)
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(pretrained_vector))
        self.attenCombine = nn.Linear(gc.reduce_dim + gc.word_dim, gc.cell_dim)
        self.decoder = nn.GRUCell(gc.cell_dim, gc.cell_dim)
        self.outputLinear = nn.Linear(gc.cell_dim, gc.vocab_size)


    def forward(self, img_feature, box_feature, atten_mask, label=None, infer=False):
        if label is None or infer:
            return self.evaluate(img_feature, box_feature, atten_mask)
        else:
            return self.train_step(img_feature, box_feature, label, atten_mask)

    def train_step(self, img_feature, box_feature, label, atten_mask):
        #   Temporally using simple dot product as attention function. Therefore, the reduce_dim and cell_dim are required to be the same.
        #   input:
        #       img_feature: batch * max_box_num * feature_dim
        #       box_feature: batch * 1 * feature_dim
        #       box_num: batch
        #       label: batch * max_out_len
        #   output:
        #       prob: batch * max_out_len * vocab_size
        batch = img_feature.size()[0]
#        print("img_feature: ", img_feature.size())
#        print("box_feature: ", box_feature.size())
#        print("label: ", label.size())
#        print("atten_mask: ", atten_mask.size())
        feature_reduce = self.reduceLinear(img_feature) # batch * max_box_num * reduce_dim
#        print("feature_reduce: ", feature_reduce.size())
        box_reduce = self.reduceLinear(box_feature) # batch * 1 * reduce_dim
#        print("box_reduce: ", box_reduce.size())
        decoder_h0 = self.inputLinear(box_feature).squeeze(1) # batch * cell_dim
#        print("decoder_h0: ", decoder_h0.size())
        h = decoder_h0 # batch * cell_dim
        w = torch.ones((batch), dtype = torch.long) * gc.BOS_id
        w = w.to(gc.device)
        ret_prob = None
        for i in range(gc.output_padding):
            atten_w = F.softmax(torch.sum((h.unsqueeze(1) * feature_reduce), -1), -1) # batch * max_box_num
#            print("atten_w: ", atten_w.size())
            atten_w = atten_w * atten_mask
#            print("atten_w: ", atten_w.size())
            atten_w = atten_w / (torch.sum(atten_w, -1) + 1e-8).unsqueeze(-1)
#            print("atten_w: ", atten_w.size())
            cv = torch.sum((atten_w.unsqueeze(2) * feature_reduce), 1) # batch * reduce_dim
#            print("cv: ", cv.size())
            gru_input = self.attenCombine(torch.cat([cv, self.embedding(w)], -1))
            gru_input = F.relu(gru_input)
#            print("gru_input: ", gru_input.size())
            h = self.decoder(gru_input, h)
#            print("h: ", h.size())
            prob = self.outputLinear(h) # batch * vocab_size
#            print("prob: ", prob.size())

            if ret_prob is None:
                ret_prob = prob.clone().unsqueeze(1)
            else:
                ret_prob = torch.cat([ret_prob, prob.clone().unsqueeze(1)], 1)
#            print("ret_prob: ", ret_prob.size())
            w = label[:, i]
#            print("w: ", w.size())
        return ret_prob


    def evaluate(self, img_feature, box_feature, atten_mask):
        #   Temporally using simple dot product as attention function. Therefore, the reduce_dim and cell_dim are required to be the same.
        #   input:
        #       img_feature: batch * max_box_num * feature_dim
        #       box_feature: batch * 1 * feature_dim
        #   output:
        #       prob: batch * max_out_len * vocab_size
        batch = img_feature.size()[0]
        feature_reduce = self.reduceLinear(img_feature) # batch * max_box_num * reduce_dim
        box_reduce = self.reduceLinear(box_feature) # batch * 1 * reduce_dim
        decoder_h0 = self.inputLinear(box_feature).squeeze(1) # batch * cell_dim

        h = decoder_h0 # batch * cell_dim
        w = torch.ones((batch), dtype = torch.long) * gc.BOS_id
        w = w.to(gc.device)
        ret_prob = None
        for i in range(gc.output_padding):
            atten_w = F.softmax(torch.sum((h.unsqueeze(1) * feature_reduce), -1), -1) # batch * max_box_num
            atten_w = atten_w * atten_mask
            atten_w = atten_w / (torch.sum(atten_w, -1) + 1e-8).unsqueeze(-1)
            cv = torch.sum((atten_w.unsqueeze(2) * feature_reduce), 1) # batch * reduce_dim
            gru_input = self.attenCombine(torch.cat([cv, self.embedding(w)], -1))
            gru_input = F.relu(gru_input)
            h = self.decoder(gru_input, h)
            prob = self.outputLinear(h) # batch * vocab_size

            if ret_prob is None:
                ret_prob = prob.clone().unsqueeze(1)
            else:
                ret_prob = torch.cat([ret_prob, prob.clone().unsqueeze(1)], 1)
            w = torch.argmax(prob, -1)
        return torch.softmax(ret_prob, -1)

