import torch
from torch.utils.data import Dataset
from const import global_consts as gc

import numpy as np

class RefcocoDataset(Dataset):
    def __init__(self, cls):
        print("building Refcoco Dataset, class: %s" % cls)
        self.cls = cls
        self.img_id = []
        self.box_id = []
        self.label = []
        now = 0
        for d in gc.refs_dict[cls]:
            for _ in d['sentences']:
                self.label.append(gc.label[cls][now])
                self.img_id.append(d['image_id'])
                self.box_id.append(d['ref_id'])
                now += 1
        print("%s set contains %d pieces of data" % (cls, now))

    def __len__(self):
        return len(self.box_id)

    def __getitem__(self, idx):
        '''
        ret:
            box_num(int): number of boxes used in the image features
            img_feature(box_num padded to gc.input_padding * gc.input_dim): image features
            box_feature(gc.input_dim): bounding box feature
            out_len(int): length of the caption
            label(out_len padded to gc.output_padding): caption of the bounding box
        '''
        img_id = self.img_id[idx]
        ref_id = self.box_id[idx]
        box_num = gc.image_infos[img_id]['num_box']
        atten_mask = torch.cat([torch.ones(box_num), torch.zeros(gc.input_padding-box_num)], 0)
        img_feature = torch.tensor(np.load("%s/features/%d.npy" % (gc.image_path, img_id)))
        img_feature = torch.cat((img_feature, torch.zeros((gc.input_padding-box_num, gc.input_dim))), 0)
        box_feature = torch.tensor(np.load("%s/features/%d.npy" % (gc.box_path, ref_id)))
        out_len = len(self.label[idx])
        label = torch.tensor(self.label[idx] + [0] * (gc.output_padding - out_len))
        return box_num, img_feature, box_feature, out_len, label, atten_mask
