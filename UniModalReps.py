# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 12:44:52 2019

@author: pvsha
"""

import matplotlib
matplotlib.use('Agg')

from refer import REFER
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torchvision.models as  models
import os.path as osp

global PAD_id
PAD_id = 0
global BOS_id
BOS_id = 1
global UNK_id
UNK_id = 2
global EOS_id
EOS_id = 3

def load_statistics(refer):
    print('dataset [%s_%s] contains: ' % (dataset, splitBy))
    ref_ids = refer.getRefIds()
    image_ids = refer.getImgIds()
    print('%s expressions for %s refs in %s images.' % (len(refer.Sents), len(ref_ids), len(image_ids)))

def make_split(dataset):
    if dataset == 'refcoco':
        splits = ['train', 'val', 'test']
    elif dataset == 'refcoco+':
        splits = ['train', 'val', 'test']
    elif dataset == 'refcocog':
        splits = ['train', 'val']  # we don't have test split for refcocog right now.
    return splits

def get_refIds(refer,split):
    ref_ids = refer.getRefIds(split=split)
    return ref_ids

def get_refBox(refId):
    RefBox = refer.getRefBox(refId)
    return RefBox

def get_refslist(refer,refIds_dict):
    refs_dict = {}
    for i in refIds_dict.keys():
        for j in refIds_dict[i]:
            if(i not in refs_dict.keys()):
                refs_dict[i] = [refer.Refs[j]]
            else:
                refs_dict[i].append(refer.Refs[j])
    return refs_dict

def show_image_annoted(refer,ref_id,seg_box='box'):
    plt.figure()
    refer.showRef(ref_id,seg_box)
    plt.show()

def show_image(img_list):
    for i in img_list:
        plt.figure()
        plt.imshow(i)


def load_images(refer,ref_list):
    lst = []
    for i in range(len(ref_list)):
        lst.append(refer.loadImgs(ref_list[i]['image_id']))
    return lst

def get_refs_id_dict(refer, splits):
    ref_ids_dict = {}
    for split in splits:
        ref_ids_dict[split] = get_refIds(refer,split)
    return ref_ids_dict

def get_bounded_image(refer,ref_list,loaded_img_list):
    bounded_img_list = []
    for i in range(len(loaded_img_list)):
        I = io.imread(osp.join(refer.IMAGE_DIR, loaded_img_list[i][0]['file_name']))
        RefBox = get_refBox(ref_list[i])
        bounded_img_list.append(I[int(RefBox[1]):int(RefBox[1]+RefBox[3]),int(RefBox[0]):int(RefBox[0]+RefBox[2])])
    return bounded_img_list

def get_C4_vec(res50_C4,bounded_image_list):
    bounded_outputs = []
    for i in range(len(bounded_image_list)):
        cropped_tensor = torch.from_numpy(bounded_image_list[i])
        cropped_tensor = cropped_tensor.reshape(1,cropped_tensor.shape[2],cropped_tensor.shape[0],cropped_tensor.shape[1])
        cropped_tensor = cropped_tensor.type(dtype=torch.FloatTensor)
        bounded_outputs.append(res50_C4(cropped_tensor))
    return bounded_outputs


class Resnet_C4(nn.Module):
    def __init__(self, original_model):
        super(Resnet_C4, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
    def forward(self, x):
        x = self.features(x)
        return x

class Resnet_C3(nn.Module):
    def __init__(self,original_model):
        super(Resnet_C3, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-2])
    def forward(self,x):
        x = self.features(x)
        return x

def build_vocab(refs, min_occur = 5):
    d = dict()
    tr = refs['train']
    for ref in res_small:
        for w in ref.strip().split():
            if d.has_key(w):
                d[w] += 1
            else:
                d[w] = 1
    l = zip(d.keys(), d.values())
    l = filter(lambda x: x[1] > min_occur , sorted(l, lambda x, y: 1 if x[1] > y[1] else -1))
    return map(lambda x:x[0], l), map(lambda x:x[1], l)

if __name__ == '__main__':
    resnet_50 = models.resnet50(pretrained=True)
    res50_C4 = Resnet_C4(resnet_50)

    #Setting variables for data loading
    data_root = '../../data'  # contains refclef, refcoco, refcoco+, refcocog and images
    dataset = 'refcoco+'
    splitBy = 'unc'
    refer = REFER(data_root, dataset, splitBy)

    load_statistics(refer)
    splits = make_split(dataset)

    ref_ids_dict = get_refs_id_dict(refer,splits)
    refs_dict = get_refslist(refer,ref_ids_dict)

    refs_small = refs_dict['train'][:10]
    ref_id_small = ref_ids_dict['train'][:10]

    for i in refs_small:
        show_image_annoted(refer,i)

    img_list = load_images(refer,refs_small)
    bounded_image_list = get_bounded_image(refer,ref_id_small,img_list)


    bounded_outputs = get_C4_vec(res50_C4,bounded_image_list)
    print(bounded_outputs[0].size())
    show_image(bounded_image_list)

    vocab, freq = build_vocab(refs_dict)
    vocab = ['_PAD', '_BOS', '_UNK', '_EOS'] + vocab

    w2id = dict()
    for i, w in enumerate(vocab):
        w2id[w] = i

    labels = dict()
    labels['train'] = []
    labels['test'] = []
    labels['valid'] = []

    sen2id = lambda x: w2id[x] if w2id.has_key(x) else UNK_id
    for r in refs_dict['train']:
        labels['train'].append(map(sen2id, r.strip().split()))
    for r in refs_dict['test']:
        labels['test'].append(map(sen2id, r.strip().split()))
    for r in refs_dict['valid']:
        labels['valid'].append(map(sen2id, r.strip().split()))

