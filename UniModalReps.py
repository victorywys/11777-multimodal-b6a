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
import torch.optim as optim
import torch
import torchvision.models as  models
import os.path as osp
from RefcocoDataset import RefcocoDataset as RefD
import torch.utils.data as Data
from model import Net

global PAD_id
PAD_id = 0
global BOS_id
BOS_id = 1
global UNK_id
UNK_id = 2
global EOS_id
EOS_id = 3
global BATCH_SIZE
BATCH_SIZE = 32
global EPOCH_NUM
EPOCH_NUM = 10
global MAX_LEN
MAX_LEN = 25
global LEARNING_RATE
LEARNING_RATE = 1e-2

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
    print("bounded_image get")
    return bounded_img_list

def get_C4_vec(res50_C4,bounded_image_list):
    bounded_outputs = []
    for i in range(len(bounded_image_list)):
        cropped_tensor = torch.from_numpy(bounded_image_list[i])
        cropped_tensor = cropped_tensor.reshape(1,cropped_tensor.shape[2],cropped_tensor.shape[0],cropped_tensor.shape[1])
        cropped_tensor = cropped_tensor.type(dtype=torch.FloatTensor)
        bounded_outputs.append(res50_C4(cropped_tensor))
    print("C4_vec")
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
    print("start building vocabulary...")
    for i, ref in enumerate(tr):
        if i % 10000 == 0:
            print("processing image %d" % i)
        for sen in ref['sentences']:
            for w in sen['tokens']:
                if d.has_key(w):
                    d[w] += 1
                else:
                    d[w] = 1
    l = zip(d.keys(), d.values())
    l = filter(lambda x: x[1] > min_occur , sorted(l, lambda x, y: 1 if x[1] < y[1] else -1))
    return map(lambda x:x[0], l), map(lambda x:x[1], l)

if __name__ == '__main__':
    resnet_50 = models.resnet50(pretrained=True)
    res50_C4 = Resnet_C4(resnet_50)

    #Setting variables for data loading
    data_root = '../data'  # contains refclef, refcoco, refcoco+, refcocog and images
    dataset = 'refcoco+'
    splitBy = 'unc'
    refer = REFER(data_root, dataset, splitBy)

    load_statistics(refer)
    splits = make_split(dataset)

    ref_ids_dict = get_refs_id_dict(refer,splits)
    refs_dict = get_refslist(refer,ref_ids_dict)

    refs_small = refs_dict['train'][:10]
    ref_id_small = ref_ids_dict['train'][:10]
    print(len(refs_dict['train']))
    print(len(refs_dict['test']))
    print(len(refs_dict['val']))

    for i in refs_small:
        show_image_annoted(refer,i)

    img_list = dict()
    bounded_img_list = dict()
    bounded_outputs = dict()

    num = {'train': 50, 'test': 10, 'val':10}
    for key in ['train', 'test', 'val']:
        img_list[key] = load_images(refer,refs_dict[key][:num[key]])
    #    bounded_img_list[key] = get_bounded_image(refer, ref_ids_dict[key][:num[key]], img_list[key])
    #    bounded_outputs[key] = get_C4_vec(res50_C4,bounded_img_list[key])
#    print(len(bounded_outputs['train']))
#    print(bounded_outputs['train'][0].size())
#    print(bounded_outputs['train'][0])

    vocab, freq = build_vocab(refs_dict)
    fout = open("vocab.txt", 'w')
    for u, v in zip(vocab, freq):
        fout.write("%s\t%d\n" % (u, v))

    vocab = ['_PAD', '_BOS', '_UNK', '_EOS'] + vocab

    w2id = dict()
    for i, w in enumerate(vocab):
        w2id[w] = i

    labels = dict()
    labels['train'] = []
    labels['test'] = []
    labels['valid'] = []

    sen2id = lambda x: w2id[x] if w2id.has_key(x) else UNK_id
    for key in ['train', 'test', 'val']:
        labels[key] = []
        for r in refs_dict[key][:num[key]]:
            labels[key].append(map(sen2id, r['sentences'][0]['tokens']) + [EOS_id])

    feed_data = dict()
    for key in ['train', 'test', 'val']:
        feed_data[key] = RefD(bounded_outputs[key], labels[key])

    train_loader = Data.DataLoader(
            dataset=feed_data['train'],
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=1,
        )
    test_loader = Data.DataLoader(
            dataset=feed_data['test'],
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=1,
        )
    val_loader = Data.DataLoader(
            dataset=feed_data['val'],
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=1,
        )

    net = Net()
    print net

    running_loss = 0.0
    criterion = nn.CrossEntropyLoss(ignore_index=UNK_id)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCH_NUM):
        print("Training epoch %d" % epoch)
        for i, data in enumerate(train_loader):
            img, label = data
            outputs = net(img)
            output_flat = []
            label_flat = []
            for j in range(len(label)):
                output_flat += outputs[j][:min(len(label[j]), MAX_LEN)]
                label_flat += label[j][:min(len(label[j]), MAX_LEN)]
            loss = criterion(outputs_flat, label_flat)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 50 == 49:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0

        with torch.no_grad():
            log_per = 0.0
            for i, data in enumerate(test_loader):
                img, label = data
                outputs = net(img)
                output_flat = []
                label_flat = []
                for j in range(len(label)):
                    for k in range(min(len(label[j]), MAX_LEN)):
                        log_per += torch.log(outputs[j][k][label[j][k]])
                log_per /= len(label)
                print('Epoch %d: Perplexity on test: %.2f' % (epoch + 1, 2**(-log_per)))

            log_per = 0.0
            for i, data in enumerate(val_loader):
                img, label = data
                outputs = net(img)
                output_flat = []
                label_flat = []
                for j in range(len(label)):
                    for k in range(min(len(label[j]), MAX_LEN)):
                        log_per += torch.log(outputs[j][k][label[j][k]])
                log_per /= len(label)
                print('Epoch %d: Perplexity on valid: %.2f' % (epoch + 1, 2**(-log_per)))

