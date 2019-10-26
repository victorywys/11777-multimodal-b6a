#import chainer.functions as F
#from chainer import cuda
#from chainer import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def emb_crits(emb_flows, margin, vlamda=1, llamda=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #xp = torch.tensor(emb_flows['vis'][0], device=device)
    #xp = cuda.get_array_module(emb_flows['vis'][0])
    batch_size = emb_flows['vis'][0].shape[0]
    
    zeros = torch.zeros(batch_size).to(device)#, dtype=xp.float32))
    vis_loss = torch.mean(torch.max(zeros, margin+emb_flows['vis'][1]-emb_flows['vis'][0]))
    lang_loss = torch.mean(torch.max(zeros, margin+emb_flows['lang'][1]-emb_flows['lang'][0]))
    return vlamda*vis_loss + llamda*lang_loss
    
def lm_crits(lm_flows, num_labels, margin, vlamda=1, llamda=0, langWeight=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #xp = cuda.get_array_module(lm_flows['T'])
    ## language loss
    n = 0
    lang_loss = 0
    Tprob = lm_flows['T']
    lang_num = num_labels['T']
    lang_loss -= torch.sum(Tprob)/(sum(lang_num)+len(lang_num))
    if vlamda==0 and llamda==0:
        return lang_loss
    
    def triplet_loss(flow, num_label):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pairGenP = flow[0]
        unpairGenP = flow[1]
        zeros = torch.zeros(pairGenP.shape[1]).to(device)
        pairSentProbs = torch.sum(pairGenP,axis=0)/(num_label+1)
        unpairSentProbs = torch.sum(unpairGenP,axis=0)/(num_label+1)
        trip_loss = torch.mean(torch.max(zeros, margin+unpairSentProbs-pairSentProbs))
        return trip_loss
    
    vloss = triplet_loss(lm_flows['visF'], torch.tensor(np.array(num_labels['T'])).to(device))
    lloss = triplet_loss(lm_flows['langF'], torch.tensor(np.array(num_labels['F'])).to(device))
    #print(lang_loss, vloss, lloss)
    return langWeight*lang_loss + vlamda*vloss+llamda*lloss