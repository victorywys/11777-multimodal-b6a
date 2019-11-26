import numpy as np
import math
# import chainer
# import chainer.functions as F
# import chainer.links as L
# from chainer import Variable, cuda
import torch
import torch.nn.functional as F
import torch.nn as nn
from misc.utils import softmax_sample


def vis_combine(vis_enc, vis_emb, init_norm=20):
    return torch.concat((vis_enc, F.dropout(vis_emb*init_norm, p=0.25)),dim=1)

class LanguageModel(nn.Module):
    def __init__(self,vocab_size, seq_length):
        super(LanguageModel, self).__init__(
            word_emb = nn.Embedding(vocab_size+2, 512),
            LSTM = MyLSTM(512+512, 512, 512, 0.5),
            out = nn.Linear(512, vocab_size+1),
        )
        self.vocab_size = vocab_size
        self.seq_length = seq_length
            
    def LSTM_initialize(self):
        self.LSTM.h_0 = None
        self.LSTM.c_0 = None
    
    def forward(self, feat, w, i):
        w = self.word_emb(w)
        if i==0:
            _, (h,c) = self.LSTM(vis=feat, sos=w)
        else:
            _, (h,c) = self.LSTM(vis=feat, word=w)
        return self.out(h)
            
    def __call__(self, vis_feats, seqz, lang_last_ind):
        seqz = seqz.data
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #xp = cuda.get_array_module(vis_feats)
        batch_size = vis_feats.shape[0]
        self.LSTM_initialize()
        log_probs = []
        for i in range(max(lang_last_ind)+1):
            if i==0:
                mask = torch.ones(batch_size, dtype=torch.float32).to(device)
                sos = torch.ones(batch_size,dtype=torch.int32)*(self.vocab_size+1).to(device)
                sos = self.word_emb(sos)
                _, (h,c) = self.LSTM(vis=vis_feats, sos=sos)
            else:
                mask = torch.where(seqz[:, i-1]!=0,torch.tensor(1),torch.tensor(0)).to(device)
                w = self.word_emb(seqz[:, i-1])
                _, (h,c) = self.LSTM(vis=vis_feats, word=w)
            h = self.out(h)
            logsoft = (F.log_softmax(h)*mask.reshape(batch_size, 1).repeat(h.data.shape[1], axis=1))[torch.arange(batch_size), seqz[:,i]]
                
            log_probs.append(logsoft.reshape(1,batch_size)) 
                
        return torch.concat(log_probs, dim=0) 
    
    def sample(self, vis_feats, temperature=1, stochastic=True):
        #xp = cuda.get_array_module(vis_feats)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_size = vis_feats.shape[0]
        self.LSTM_initialize()
        
        output = torch.zeros((batch_size, self.seq_length), dtype=xp.int32).to(device)
        log_probs = [] 
        mask = torch.ones(batch_size).to(device)
        
        with torch.no_grad():
            for i in range(self.seq_length):
                if i==0:
                    sos = self.word_emb(torch.ones(batch_size,dtype=torch.int32)*(self.vocab_size+1)).to(device)
                    _, (h,c) = self.LSTM(vis=vis_feats, sos=sos)
                else:
                    mask_ = torch.where(w!=0,torch.tensor(1),torch.tensor(0))
                    mask *= mask_
                    if mask.sum()==0:
                        break
                    w = self.word_emb(torch.tensor(w))
                    _, (h,c) = self.LSTM(vis=vis_feats, word=w)
                h = self.out(h)
                logsoft = F.log_softmax(h)*mask.reshape(batch_size, 1).repeat(h.data.shape[1], axis=1)# if input==eos then mask

                if stochastic:
                    prob_prev = torch.exp(logsoft/torch.tensor(temperature))
                    prob_prev /= torch.expand(F.sum(prob_prev, axis=1, keepdim=False), prob_prev.shape)
                    w = softmax_sample(prob_prev)
                else:
                    w = torch.argmax(logsoft.data, dim=1).to(device)
                output[:, i] = w
                log_probs.append(logsoft[np.arange(batch_size), w].reshape(1,batch_size))
        return output, torch.concat(log_probs, axis=0)

    def max_sample(self, vis_feats):
        #xp = cuda.get_array_module(vis_feats)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_size = vis_feats.shape[0]
        self.LSTM_initialize()
        
        output = torch.zeros((batch_size, self.seq_length), dtype=torch.int32).to(device)
        mask = torch.ones(batch_size).to(device)
        for i in range(self.seq_length):
            if i==0:
                sos = self.word_emb(torch.ones(batch_size,dtype=torch.int32)*(self.vocab_size+1).to(device))
                _, (h,c) = self.LSTM(vis=vis_feats, sos=sos)
            else:
                mask_ = torch.where(output[:,i-1]!=0,torch.tensor(1),torch.tensor(0))
                mask *= mask_
                if mask.sum()==0:
                    break
                w = self.word_emb(output[:,i-1])
                _, (h,c) = self.LSTM(word=w)
            h = self.out(h)
            output[:,i] = torch.argmax(h.data[:,:-1], dim=1)
            
        result = []
        for out in output:
            for i, w in enumerate(out):
                if w==0:
                    result.append(out[:i])
                    break
                
        return result


class MyLSTM(nn.Module):
    
    def __init__(self, vis_size, word_size, rnn_size, dropout_ratio):
         super(MyLSTM, self).__init__(
                vis2g = nn.Linear(vis_size, rnn_size),
                h2g = nn.Linear(rnn_size, rnn_size, bias = False),
                w2g = nn.Linear(word_size, rnn_size, bias = False),
                lstm = nn.LSTM(word_size+vis_size, rnn_size),
         )
         self.dropout_ratio = dropout_ratio
    
    def reset_state(self):
        self.lstm.h_0 = None
        self.lstm.c_0 = None
        
    def __call__(self, vis = None, sos = None, word = None):
        
        if sos is not None:
            input_emb = torch.concat((vis, sos), dim=1)
            g = self.vis2g(vis)+self.w2g(sos)
            h = F.dropout(self.lstm(input_emb), p = self.dropout_ratio)
            
        else:
            word = F.dropout(word, p = self.dropout_ratio)
            input_emb = F.concat((vis, word), dim=1)
            g = F.sigmoid(self.w2g(word) + self.vis2g(vis) + self.h2g(self.lstm.h))
            h = F.dropout(self.lstm(input_emb), p = self.dropout_ratio)
            
        s_t = F.dropout(g * F.tanh(self.lstm.c), p=self.dropout_ratio)
    
        return s_t, h