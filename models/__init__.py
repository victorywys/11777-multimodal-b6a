from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy

import numpy as np
import misc.utils as utils
import torch

from .ShowTellModel import ShowTellModel
from .FCModel import FCModel
from .OldModel import ShowAttendTellModel, AllImgModel
from .AttModel import *
from .TransformerModel import TransformerModel

from .SpeakerListenerModel import SpeakerListenerModel
from .GDiscriminator import GDiscriminator
from .EDiscriminator import EDiscriminator
from .Discriminator import Discriminator

def setup(opt):

    if opt.caption_model == 'fc':
        model = FCModel(opt)
    elif opt.caption_model == 'show_tell':
        model = ShowTellModel(opt)
    # Att2in model in self-critical
    elif opt.caption_model == 'att2in':
        model = Att2inModel(opt)
    # Att2in model with two-layer MLP img embedding and word embedding
    elif opt.caption_model == 'att2in2':
        model = Att2in2Model(opt)
    elif opt.caption_model == 'att2all2':
        model = Att2all2Model(opt)
    # Adaptive Attention model from Knowing when to look
    elif opt.caption_model == 'adaatt':
        model = AdaAttModel(opt)
    # Adaptive Attention with maxout lstm
    elif opt.caption_model == 'adaattmo':
        model = AdaAttMOModel(opt)
    # Top-down attention model
    elif opt.caption_model == 'topdown':
        model = TopDownModel(opt)
    # StackAtt
    elif opt.caption_model == 'stackatt':
        model = StackAttModel(opt)
    # DenseAtt
    elif opt.caption_model == 'denseatt':
        model = DenseAttModel(opt)
    # Transformer
    elif opt.caption_model == 'transformer':
        model = TransformerModel(opt)
    elif opt.caption_model == 'speakerlistener':
        model = SpeakerListenerModel(opt)
    elif opt.caption_model == 'gdiscriminator':
        model = GDiscriminator(opt)
    elif opt.caption_model == 'ediscriminator':
        model = EDiscriminator(opt)
    elif opt.caption_model == 'discriminator':
        model = Discriminator(opt)
    else:
        raise Exception("Caption model not supported: {}".format(opt.caption_model))

    # check compatibility if training is continued from previously saved model
    if vars(opt).get('start_from', None) is not None:
        # check if all necessary files exist
        assert os.path.isdir(opt.start_from)," %s must be a a path" % opt.start_from
        assert os.path.isfile(os.path.join(opt.start_from,"infos_"+opt.id+".pkl")),"infos.pkl file does not exist in path %s"%opt.start_from
        if opt.caption_model == "gdiscriminator":
            model.load_state_dict(torch.load(os.path.join(opt.start_from, 'gd_model.pth')))
        elif opt.caption_model == "ediscriminator":
            model.load_state_dict(torch.load(os.path.join(opt.start_from, 'ed_model.pth')))
        elif opt.caption_model == "discriminator":
            if opt.continue_mode == "separate":
                model.gd.load_state_dict(torch.load(os.path.join(opt.discriminator_start_from, 'gd_model.pth')))
                model.ed.load_state_dict(torch.load(os.path.join(opt.discriminator_start_from, 'ed_model.pth')))
            elif opt.continue_mode == "joint":
                model.load_state_dict(torch.load(os.path.join(opt.discriminator_start_from, 'dis_model.pth')))
        elif opt.caaption_model == "speakerlistener":
            if opt.continue_mode == "separate":
                model.load_state_dict(torch.load(os.path.join(opt.start, 'model.pth')))
            else:
                model.load_state_dict(torch.load(os.path.join(opt.start, 'gen_model.pth')))
        else:
            model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model.pth')))
    return model
