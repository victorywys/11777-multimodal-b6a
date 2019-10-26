#import chainer
#import chainer.functions as F
#import chainer.links as L
#from chainer import cuda
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import LanguageEncoderAttn, LanguageEncoder, VisualEncoder, MetricNet
    
class ListenerReward(nn.Module):
    def __init__(self, vocab_size, attention=True, scale=1):
        if attention:
            le = LanguageEncoderAttn
        else:
            le = LanguageEncoder
            
        super(ListenerReward, self).__init__(
            ve = VisualEncoder(),
            le = le(vocab_size),
            me = MetricNet()
        )
        self.scale=scale
            
    def calc_score(self, feats, seq, lang_length):
        #with chainer.using_config('train', False):
        with torch.no_grad():
            vis_enc_feats = self.ve(feats)
            lang_enc_feats = self.le(seq, lang_length)
            lr_score = F.sigmoid(self.me(vis_enc_feats, lang_enc_feats))
        return lr_score
    
    def __call__(self, feats, seq, seq_prob, lang_length):#, baseline):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #xp = torch.tensor(feats, device=device)
        #xp = cuda.get_array_module(feats)
        
        lr_score = self.calc_score(feats, seq, lang_length).data[:,0]
        self.reward = lr_score*self.scale
        #torch.tensor(np.array(num_labels['T'])).to(device))
        #loss = -torch.mean(torch.sum(seq_prob, axis=0)/(xp.array(lang_length+1))*(self.reward-self.reward.mean()))
        loss = -torch.mean(torch.sum(seq_prob, axis=0)/torch.tensor(np.array(lang_length+1)).to(device)*(self.reward-self.reward.mean()))
        return loss