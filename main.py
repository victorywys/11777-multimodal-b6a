# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 12:44:52 2019

@author: pvsha
"""
from const import global_consts as gc
import sys
sys.path.append("/home/ubuntu/11777/yansen/11777-multimodal-b6a/refer/")
sys.path.append("../refer")
sys.path.append("../refer/evaluation")
sys.path.append("../refer/evaluation/bleu")
sys.path.append("../refer/evaluation/cider")

import matplotlib
matplotlib.use('Agg')

from refer import REFER
from evaluation.refEvaluation import *
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
from torch import autograd
import torchvision.models as  models
import os.path as osp
from RefcocoDataset import RefcocoDataset as RefD
import torch.utils.data as Data
import torch.nn.functional as F
from model import Net
from PIL import Image
from torchvision import transforms
import json

import util
def load_statistics(refer):
    print('dataset [%s_%s] contains: ' % (dataset, gc.split_by))
    ref_ids = refer.getRefIds()
    image_ids = refer.getImgIds()
    print('%s expressions for %s refs in %s images.' % (len(refer.Sents), len(ref_ids), len(image_ids)))

def make_split(dataset):
    if dataset == 'refcoco':
        splits = ['train', 'val', 'testA', 'testB']
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

def build_vocab(refs, min_occur = 5):
    d = dict()
    tr = refs['train']
    print("start building vocabulary...")
    for i, ref in enumerate(tr):
        if i % 10000 == 0:
            print("processing image %d" % i)
        for sen in ref['sentences']:
            for w in sen['tokens']:
                if w in d:
                    d[w] += 1
                else:
                    d[w] = 1
    l = zip(d.keys(), d.values())
    l = list(l)
    def takeSecond(elem):
        return elem[1]
    l.sort(key=takeSecond, reverse=True)
    l = filter(lambda x: x[1] >= min_occur , list(sorted(list(l), key=lambda x:x[1], reverse=True)))
    return list(l)

def load_wv(path):
    vectors = {}
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i % 50000 == 0:
                print("loading vector %d" % i)
            l = line.strip().split()
            vectors[l[0]] = map(lambda x: float(x), l[1:])
    return vectors



if __name__ == '__main__':
    gc.device = device = torch.device("cuda:%d" % gc.cuda if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(gc.cuda)
    print("running device: ", device)

    #Setting variables for data loading
    dataset = gc.dataset
    refer = REFER(gc.refer_data_path, gc.dataset, gc.split_by)

    load_statistics(refer)
    splits = make_split(dataset)

    gc.ref_ids_dict = ref_ids_dict = get_refs_id_dict(refer, splits)
    gc.refs_dict = refs_dict = get_refslist(refer,ref_ids_dict)

    print("load image summary...")
    img_sum = open("%s/%s" % (gc.image_path, "summary.csv"))
    box_sum = open("%s/%s" % (gc.box_path, "summary.csv"))
    gc.image_infos = image_infos = {}
    max_box_num = 0
    for line in img_sum:
        img_id, img_h, img_w, img_num_box = line.strip().split(',')
        image_infos[int(img_id)] = {'h': int(img_h), 'w': int(img_w), 'num_box': int(img_num_box)}
        max_box_num = max(max_box_num, int(img_num_box))
    img_sum.close()
    gc.box_infos = box_infos = {}
    for line in box_sum:
        ref_id, ref_h, ref_w, _ = line.strip().split(',')
        box_infos[int(ref_id)] = {'h': int(ref_h), 'w': int(img_w)}
    box_sum.close()

    if gc.debug:
        for key in ["train", "testA", "testB", "val"]:
            refs_dict[key] = refs_dict[key][:32]

    print(len(refs_dict['train']))
    print(len(refs_dict['testA']))
    print(len(refs_dict['testB']))
    print(len(refs_dict['val']))
    print("max box number: %d" % max_box_num)
    gc.input_padding = max_box_num + 1
    gc.input_dim = np.load("%s/features/%d.npy" % (gc.image_path, refs_dict['train'][0]['image_id'])).shape[-1]
    vf = build_vocab(refs_dict, gc.min_occur)
    vocab = []
    freq = []
    with open("vocab.txt", 'w') as fout:
        for i in range(len(vf)):
            vocab.append(vf[i][0])
            freq.append(vf[i][1])
            fout.write("%s %d\n" % (vocab[i], freq[i]))
    glove_vectors = load_wv(gc.wv_path)
    vocab = ['_PAD', '_BOS', '_UNK', '_EOS'] + list(vocab)
    gc.vocab_size = len(vocab)
    vocab_vectors = []
    for w in vocab:
        if w in glove_vectors:
            vocab_vectors.append(list(glove_vectors[w]))
        else:
            vocab_vectors.append(np.random.rand((gc.word_dim)).tolist())

    word_id = dict()
    for i, w in enumerate(vocab):
        word_id[w] = i

    w2id = lambda x: word_id[x] if x in word_id else gc.UNK_id
    gc.label = label = {}

    longest = 0
    for key in ['train', 'testA', 'testB', 'val']:
        label[key] = []
        for data in refs_dict[key]:
            for sen in data['sentences']:
                label[key].append(list(map(w2id, sen['tokens'])) + [gc.EOS_id])
                longest = max(longest, len(sen['tokens']))
    print("longest sentence: %d words" % longest)
    gc.output_padding = longest + 1

    train_loader = Data.DataLoader(
            dataset=RefD('train'),
            batch_size=gc.batch_size,
            shuffle=True,
            num_workers=1,
        )
    testA_loader = Data.DataLoader(
            dataset=RefD('testA'),
            batch_size=gc.batch_size,
            shuffle=False,
            num_workers=1,
        )
    testB_loader = Data.DataLoader(
            dataset=RefD('testB'),
            batch_size=gc.batch_size,
            shuffle=False,
            num_workers=1,
        )
    val_loader = Data.DataLoader(
            dataset=RefD('val'),
            batch_size=gc.batch_size,
            shuffle=False,
            num_workers=1,
        )

    net = Net(vocab_vectors)
    net.to(device)
    print(net)

    if gc.inference:
        net = torch.load('%s/model_checkpoint_%d.pt' % (gc.checkpoint_path, gc.checkpoint_id))
        fout = open("testA.txt", 'w')
        print("evaluating on testA")
        id_list = []
        Res = []
        if True:
#        with torch.no_grad():
            for i, data in enumerate(testA_loader):
                ref_id, box_num, img, box, out_len, label, atten_mask = data
                ref_id = ref_id.tolist()
                img = img.to(device)
                label = label.to(device)
                box = box.to(device)
                atten_mask = atten_mask.to(device)
                outputs = net(img, box, atten_mask)
                predict = outputs.argmax(-1)[:, :8].tolist()
                for j in range(len(label)):
                    if ref_id[j] in id_list:
                        continue
                    id_list.append(ref_id[j])
                    fout.write("golden:\n\t")
                    for k in range(out_len[j]):
                        if label[j][k] > 3:
                            fout.write("%s " % vocab[label[j][k]])
                    fout.write("\npredict:\n\t")
                    sen = ""
                    for k in range(len(predict[j])):
                        if predict[j][k] == gc.EOS_id:
                            break
                        fout.write("%s " % vocab[predict[j][k]])
                        sen = sen + vocab[predict[j][k]] + " "
                    fout.write("\n\n")
                    Res.append({'ref_id': ref_id[j], 'sent': sen.strip()})
        fout.close()
        json.dump(Res, open("testA.json", 'w'))

        print("evaluating on testB")
        fout = open("testB.txt", 'w')
        id_list = []
        Res = []
        if True:
#        with torch.no_grad():
            for i, data in enumerate(testB_loader):
                ref_id, box_num, img, box, out_len, label, atten_mask = data
                ref_id = ref_id.tolist()
                img = img.to(device)
                label = label.to(device)
                box = box.to(device)
                atten_mask = atten_mask.to(device)
                outputs = net(img, box, atten_mask)
                predict = outputs.argmax(-1).tolist()
                for j in range(len(label)):
                    if ref_id[j] in id_list:
                        continue
                    id_list.append(ref_id[j])
                    fout.write("golden:\n\t")
                    for k in range(out_len[j]):
                        if label[j][k] > 3:
                            fout.write("%s " % vocab[label[j][k]])
                    fout.write("\npredict:\n\t")
                    sen = ""
                    for k in range(len(predict[j])):
                        if predict[j][k] == gc.EOS_id:
                            break
                        fout.write("%s " % vocab[predict[j][k]])
                        sen = sen + vocab[predict[j][k]] + " "
                    fout.write("\n\n")
                    Res.append({'ref_id': ref_id[j], 'sent': sen.strip()})
        fout.close()
        json.dump(Res, open("testB.json", 'w'))
        sys.exit()

    running_loss = 0.0
    criterion = nn.CrossEntropyLoss(ignore_index=gc.UNK_id)
    optimizer = optim.Adam(net.parameters(), lr=gc.learning_rate)

    for epoch in range(gc.epoch_num):
        print("Training epoch %d" % epoch)
        for i, data in enumerate(train_loader):
            _, box_num, img, box, out_len, label, atten_mask = data
            img = img.to(device)
            label = label.to(device)
            box = box.to(device)
            atten_mask = atten_mask.to(device)
            with autograd.detect_anomaly():
                outputs = net(img, box, atten_mask, label)
                output_flat = None
                label_flat = None
                for j in range(len(label)):
                    if output_flat is None:
                        output_flat = outputs[j][:out_len[j]]
                        label_flat = label[j][:out_len[j]]
                    else:
                        output_flat = torch.cat([output_flat, outputs[j][:out_len[j]]], 0)
                        label_flat = torch.cat([label_flat, label[j][:out_len[j]]], 0)
                loss = criterion(output_flat, label_flat)
                loss.backward()
                optimizer.step()
            running_loss += loss.item()
            if gc.debug:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0
            else:
                if i % 50 == 49:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 50))
                    running_loss = 0.0
        torch.save(net, "%s/model_checkpoint_%d.pt" % (gc.checkpoint_path, epoch + 1))

        with torch.no_grad():
            log_per = 0.0
            for i, data in enumerate(testA_loader):
                _, box_num, img, box, out_len, label, atten_mask = data
                img = img.to(device)
                label = label.to(device)
                box = box.to(device)
                atten_mask = atten_mask.to(device)
                outputs = net(img, box, atten_mask)
                for j in range(len(label)):
                    for k in range(out_len[j]):
                        log_per += torch.log(outputs[j][k][label[j][k]])
                log_per2 = log_per / sum(out_len)
                log_per /= len(label)
            print('Epoch %d: Perplexity on testA: %.2f, %.2f' % (epoch + 1, 2**(-log_per), 2**(-log_per2)))

            log_per = 0.0
            for i, data in enumerate(testB_loader):
                _, box_num, img, box, out_len, label, atten_mask = data
                img = img.to(device)
                label = label.to(device)
                box = box.to(device)
                atten_mask = atten_mask.to(device)
                outputs = net(img, box, atten_mask)
                for j in range(len(label)):
                    for k in range(out_len[j]):
                        log_per += torch.log(outputs[j][k][label[j][k]])
                log_per2 = log_per / sum(out_len)
                log_per /= len(label)
            print('Epoch %d: Perplexity on testB: %.2f, %.2f' % (epoch + 1, 2**(-log_per), 2**(-log_per2)))

            log_per = 0.0
            for i, data in enumerate(val_loader):
                _, box_num, img, box, out_len, label, atten_mask = data
                img = img.to(device)
                label = label.to(device)
                box = box.to(device)
                atten_mask = atten_mask.to(device)
                outputs = net(img, box, atten_mask)
                for j in range(len(label)):
                    for k in range(out_len[j]):
                        log_per += torch.log(outputs[j][k][label[j][k]])
                log_per2 = log_per / sum(out_len)
                log_per /= len(label)
            print('Epoch %d: Perplexity on val: %.2f, %.2f' % (epoch + 1, 2**(-log_per), 2**(-log_per2)))

