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
    def __init__(self):
        super(Net, self).__init__()
        self.inputLinear = nn.Linear(gc.input_dim, gc.norm_dim)
        self.decoder = nn.LSTM(gc.norm_dim, gc.cell_dim, batch_first=True)
        self.outputLinear = nn.Linear(gc.cell_dim, gc.vocab_size)

    def forward(self, img):
        LSTM_input = self.inputLinear(img.squeeze())
        _, LSTM_output = self.decoder(LSTM_input)
        prob = F.softmax(self.outputLinear(LSTM_output), -1)
        return prob

