import torch
import torch.nn as nn
import torch.nn.functional as F
from const import global_consts as gc

class Net(nn.Module):
    def __init__(self, pretrained_vector):
        super(Net, self).__init__()
        self.inputLinear = nn.Linear(gc.word_dim, gc.cell_dim)
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(pretrained_vector))
        if gc.cell_type == "LSTM":
            self.encoder = nn.LSTM(gc.word_dim, gc.cell_dim, gc.num_layers, batch_first=True)
            self.decoder = nn.LSTMCell(gc.word_dim, gc.cell_dim)
        elif gc.cell_type == "GRU":
            self.encoder = nn.GRU(gc.word_dim, gc.cell_dim, gc.num_layers, batch_first=True)
            self.decoder = nn.GRUCell(gc.word_dim, gc.cell_dim)
        elif gc.cell_type == "RNN":
            self.encoder = nn.RNN(gc.word_dim, gc.cell_dim, gc.num_layers, batch_first=True)
            self.decoder = nn.RNNCell(gc.word_dim, gc.cell_dim)

        self.outputLinear = nn.Linear(gc.cell_dim, gc.vocab_size)

    def forward(self, word, infer=False):
        inp = self.embedding(word)
        batch = word.size()[0]
        if gc.cell_type == "LSTM":
            output, (hn, cn) = self.encoder(inp)
        else:
            output, hn = self.encoder(inp)
        if gc.cell_type == "LSTM":
            h = torch.mean(hn, 0)
            c = torch.mean(cn, 0)
            w = torch.ones((batch), dtype=torch.long) * gc.BOS_id
            w = w.to(gc.device)
            ret_prob = None
            for i in range(gc.max_len):
                h, c = self.decoder(self.embedding(w), (h, c))
                prob = self.outputLinear(h)
                if ret_prob is None:
                    ret_prob = prob.clone().unsqueeze(1)
                else:
                    ret_prob = torch.cat([ret_prob, prob.clone().unsqueeze(1)], 1)
                if infer:
                    w = torch.argmax(prob, -1)
                else:
                    w = word[:, i]
            return ret_prob
        else:
            h = torch.mean(hn, 0)
            w = torch.ones((batch), dtype=torch.long) * gc.BOS_id
            w = w.to(gc.device)
            ret_prob = None
            for i in range(gc.max_len):
                h = self.decoder(self.embedding(w), h)
                prob = self.outputLinear(h)
                if ret_prob is None:
                    ret_prob = prob.clone().unsqueeze(1)
                else:
                    ret_prob = torch.cat([ret_prob, prob.clone().unsqueeze(1)], 1)
                if infer:
                    w = torch.argmax(prob, -1)
                else:
                    w = word[:, i]
            return ret_prob

