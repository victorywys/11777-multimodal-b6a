# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 12:44:52 2019

@author: pvsha
"""
from const import global_consts as gc

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
from RefcocoDataset import LMDataset
import torch.utils.data as Data
import torch.nn.functional as F
from model import Net
from PIL import Image
from torchvision import transforms

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
    print("%d images to load" % len(loaded_img_list))
    for i in range(len(loaded_img_list)):
        if (i % 100 == 0):
            print("working on image %d" % i)
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
        bounded_outputs.append(res50_C4(cropped_tensor.to(gc.device)).squeeze())
    print("C4_vec")
    return bounded_outputs

def get_C4_vec_from_ref(res50_C4, refer, ref_list, loaded_img_list):
    print("%d images to load" % len(loaded_img_list))
    bounded_outputs = None
    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    for i in range(len(loaded_img_list)):
#    for i in range(500):
        if (i % 100 == 0):
            print("working on image %d" % i)

        I = io.imread(osp.join(refer.IMAGE_DIR, loaded_img_list[i][0]['file_name']))
        RefBox = get_refBox(ref_list[i])
        bounded_img = I[int(RefBox[1]):int(RefBox[1]+RefBox[3]),int(RefBox[0]):int(RefBox[0]+RefBox[2])]
        if torch.tensor(bounded_img).dim() != 3:
            bounded_img = torch.tensor(bounded_img).unsqueeze(2)
            bounded_img = torch.cat([bounded_img, bounded_img, bounded_img], 2).numpy()
        cropped_tensor = transform(Image.fromarray(bounded_img))
        if cropped_tensor.dim() != 3:
            cropped_tensor = cropped_tensor.unsqueeze(0)
            cropped_tensor = torch.cat([croppped_tensor, cropped_tensor, cropped_tensor], 0)
        cropped_tensor = cropped_tensor.unsqueeze(0)
        cropped_tensor = cropped_tensor.to(torch.float).to(device)
        output = res50_C4(cropped_tensor).squeeze().unsqueeze(0)

        if bounded_outputs is None:
            bounded_outputs = output.clone()
        else:
            bounded_outputs = torch.cat([bounded_outputs, output.clone()], 0)
#        print(bounded_outputs.size())
    print("C4_vec get")
    return bounded_outputs.to("cpu")


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
#    tr = refs['train'][:500]
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
    gc.device = device = torch.device("cuda:%d" % gc.cuda if torch.cuda.is_available() else "cpu")

    #Setting variables for data loading
    data_root = '../../data'  # contains refclef, refcoco, refcoco+, refcocog and images
    dataset = 'refcoco+'
    splitBy = 'unc'
    refer = REFER(data_root, dataset, splitBy)

    load_statistics(refer)
    splits = make_split(dataset)

    ref_ids_dict = get_refs_id_dict(refer,splits)
    refs_dict = get_refslist(refer,ref_ids_dict)

    vocab, freq = build_vocab(refs_dict)
    fout = open("vocab.txt", 'w')
    for u, v in zip(vocab, freq):
        fout.write("%s\t%d\n" % (u, v))

    glove_vectors = load_wv(gc.wv_path)

    vocab = ['_PAD', '_BOS', '_UNK', '_EOS'] + vocab
    gc.vocab_size = len(vocab)
    vocab_vectors = []
    for w in vocab:
        if glove_vectors.has_key(w):
            vocab_vectors.append(glove_vectors[:])
        else:
            vocab_vectors.append([0 for _ in range(gc.word_dim)])

    w2id = dict()
    for i, w in enumerate(vocab):
        w2id[w] = i

    labels = dict()
    labels['train'] = []
    labels['test'] = []
    labels['valid'] = []

    sen2id = lambda x: w2id[x] if w2id.has_key(x) else gc.UNK_id
    for key in ['train', 'test', 'val']:
        labels[key] = []
#        for r in refs_dict[key][:500]:
        for r in refs_dict[key]:
            labels[key].append(map(sen2id, r['sentences'][0]['tokens']) + [gc.EOS_id])

    feed_data = dict()
    for key in ['train', 'test', 'val']:
        feed_data[key] = LMDataset(labels[key])

    train_loader = Data.DataLoader(
            dataset=feed_data['train'],
            batch_size=gc.batch_size,
            shuffle=True,
            num_workers=1,
        )
    test_loader = Data.DataLoader(
            dataset=feed_data['test'],
            batch_size=gc.batch_size,
            shuffle=False,
            num_workers=1,
        )
    val_loader = Data.DataLoader(
            dataset=feed_data['val'],
            batch_size=gc.batch_size,
            shuffle=False,
            num_workers=1,
        )

    net = Net(vocab_vectors)
    net.to(device)
    print net

    running_loss = 0.0
    criterion = nn.CrossEntropyLoss(ignore_index=gc.UNK_id)
    optimizer = optim.Adam(net.parameters(), lr=gc.learning_rate)

    for epoch in range(gc.epoch_num):
        print("Training epoch %d" % epoch)
        for i, data in enumerate(train_loader):
            word, label, length = data
            word = word.to(device)
            label = label.to(device)
            outputs = net(word, length)
            output_flat = None
            label_flat = None
            for j in range(len(label)):
                if output_flat is None:
                    output_flat = outputs[j][:length[j]]
                    label_flat = label[j][:length[j]]
                else:
                    output_flat = torch.cat([output_flat, outputs[j][:length[j]]], 0)
                    label_flat = torch.cat([label_flat, label[j][:length[j]]], 0)
            loss = criterion(output_flat, label_flat)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 50 == 49:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0

        with torch.no_grad():
            log_per = 0.0
            for i, data in enumerate(test_loader):
                word, label, length = data
                word = word.to(device)
                label = label.to(device)
                outputs = F.softmax(net(word), -1)
                output_flat = []
                label_flat = []
                for j in range(len(label)):
                    for k in range(length[j]):
                        log_per += torch.log(outputs[j][k][label[j][k]])
                log_per /= len(label)
            print('Epoch %d: Perplexity on test: %.2f' % (epoch + 1, 2**(-log_per)))

            log_per = 0.0
            for i, data in enumerate(val_loader):
                word, label, length = data
                word = word.to(device)
                label = label.to(device)
                outputs = F.softmax(net(word), -1)
                output_flat = []
                label_flat = []
                for j in range(len(label)):
                    for k in range(length[j]):
                        log_per += torch.log(outputs[j][k][label[j][k]])
                log_per /= len(label)
            print('Epoch %d: Perplexity on valid: %.2f' % (epoch + 1, 2**(-log_per)))

