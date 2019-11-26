from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import base64
import numpy as np
import csv
import sys
import zlib
import time
import mmap
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()

# output_dir
parser.add_argument('--downloaded_feats', default='/usr0/home/yansenwa/mscoco_feature/boxes/bu_feature_gt/refcoco_unc', help='downloaded feature directory')
parser.add_argument('--output_dir', default='/usr0/home/yansenwa/mscoco_feature/boxes/bu_feature_gt/refcoco_unc/cocobu_ref', help='output feature files')

args = parser.parse_args()

csv.field_size_limit(sys.maxsize)


FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']
infiles = ['refcoco_unc.tsv.0', 'refcoco_unc.tsv.1', 'refcoco_unc.tsv.2', 'refcoco_unc.tsv.3']

if not os.path.exists(args.output_dir + '_att'):
    os.makedirs(args.output_dir+'_att')
if not os.path.exists(args.output_dir + '_fc'):
    os.makedirs(args.output_dir+'_fc')
if not os.path.exists(args.output_dir + '_box'):
    os.makedirs(args.output_dir+'_box')

import sys
sys.path.append('/usr0/home/yansenwa/11777/refer2/refer')
from refer import REFER
refer = REFER('/usr0/home/yansenwa/11777/data/', dataset='refcoco', splitBy='unc')
feats_root_dir = '/usr0/home/yansenwa/mscoco_feature/cocobu'

for infile in infiles:
    print('Reading ' + infile)
    with open(os.path.join(args.downloaded_feats, infile), "r+b") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        for i, item in enumerate(tqdm(reader)):
            item['image_id'] = int(item['image_id'])
            ref = refer.loadRefs(item['image_id'])[0]
            image_id = ref['image_id']
            att_feat = np.load(os.path.join(feats_root_dir, 'cocobu_att', str(image_id) + '.npy'))
            # fc_feat = np.load(os.path.join(feats_root_dir, 'cocobu_fc', str(image_id) + '.npy'))
#            box_feat = np.load(os.path.join(feats_root_dir, 'cocobu_box', str(image_id) + '.npy'))

            item['num_boxes'] = int(item['num_boxes'])
            for field in ['boxes', 'features']:
                item[field] = np.frombuffer(base64.decodestring(item[field]),
                        dtype=np.float32).reshape((item['num_boxes'],-1))

            assert (item['features'].shape[0] == 1)
            assert (item['boxes'].shape[0] == 1)
#            item['features'] = np.vstack((item['features'], att_feat))
            #item['boxes'] = np.vstack((item['boxes'], box_feat))

            np.save(os.path.join(args.output_dir+'_att', str(item['image_id'])), item['features'].mean(0))
            np.save(os.path.join(args.output_dir+'_fc', str(item['image_id'])), att_feat.mean(0))
            np.save(os.path.join(args.output_dir+'_box', str(item['image_id'])), item['boxes'])
