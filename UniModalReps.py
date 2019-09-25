# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 12:44:52 2019

@author: pvsha
"""

from refer import REFER
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import torch.nn as nn 
import torch
import torchvision.models as  models
import os.path as osp

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
        print(i)
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



if __name__ == '__main__':
    resnet_50 = models.resnet50(pretrained=True)
    res50_C4 = Resnet_C4(resnet_50)
    
    #Setting variables for data loading
    data_root = '.\data'  # contains refclef, refcoco, refcoco+, refcocog and images
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
    show_image(bounded_image_list)
    
    #Loading Resnet pretained model
    resnet_50 = models.resnet50(pretrained=True)
    params =resnet_50.state_dict()
    
    

    
