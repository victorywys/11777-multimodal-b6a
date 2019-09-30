import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.inputLinear = nn.Linear(INPUTDIM, NORMDIM)
        self.decoder = nn.LSTM(NORMDIM, CELLDIM, batch_first=True)
        self.outputLinear = nn.Linear(CELLDIM, VOCABSIZE)

    def forward(self, img):
        LSTM_input = self.inputLinear(img.squeeze())
        _, LSTM_output = self.decoder(LSTM_input)
        prob = F.softmax(self.outputLinear(LSTM_output), -1)
        return prob
