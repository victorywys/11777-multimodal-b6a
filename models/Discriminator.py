from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils
from .GDiscriminator import GDiscriminator
from .EDiscriminator import EDiscriminator

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.ed = EDiscriminator(opt)
        self.gd = GDiscriminator(opt)

    def forward(self, *args, **kwargs):
        mode = kwargs.get('dis_mode', 'gd')
        if 'dis_mode' in kwargs:
            del kwargs['dis_mode']
        if mode == "gd":
            return self.forward_gd(*args, **kwargs)
        elif mode == "ed":
            return self.forward_ed(*args, **kwargs)
        else:
            raise Exception("unrecogized discriminator mode")

    def forward_gd(self, *args, **kwargs):
        return self.gd(*args, **kwargs)

    def forward_ed(self, *args, **kwargs):
        return self.ed(*args, **kwargs)

