import torch
import torch.nn as nn
import torch.nn.functional as F
from const import global_consts as gc

global INPUTDIM
INPUTDIM = 4096
global NORMDIM
NORMDIM = 300
global CELLDIM
CELLDIM = 300
global VOCABSIZE
VOCABSIZE = 6000

class Net(nn.Module):
    def __init__(self, pretrained_vector):
        super(Net, self).__init__()
        self.inputLinear = nn.Linear(gc.input_dim, gc.cell_dim)
        self.embedding = nn.Embedding.from_pretrained(pretrained_vector)
        self.decoder = nn.LSTMCell(gc.word_dim, gc.cell_dim)
        self.outputLinear = nn.Linear(gc.cell_dim, gc.vocab_size)

    def forward(self, img):
        LSTM_input = self.inputLinear(img.squeeze())
        batch = img.size()[0]
        h = LSTM_input
        c = torch.zeros(batch, gc.cell_dim).to(gc.device)
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
            w = torch.argmax(prob, -1)
        return ret_prob

