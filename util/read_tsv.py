#!/usr/bin/env python


import base64
import numpy as np
import csv
import sys
import zlib
import time
import mmap

csv.field_size_limit(sys.maxsize)

FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']

def readTSV(path):
    # Verify we can read a tsv
    in_data = {}
    print("reading %s..." % path)
    with open(path, "r+") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        line = 0
        for item in reader:
            line += 1
            if line % 1000 == 0:
                print("reading line %d" % line)
            item['image_id'] = int(item['image_id'])
            item['image_h'] = int(item['image_h'])
            item['image_w'] = int(item['image_w'])
            item['num_boxes'] = int(item['num_boxes'])
            for field in ['boxes', 'features']:
                item[field] = np.frombuffer(base64.decodestring(bytes(item[field], 'utf-8')),
                      dtype=np.float32).reshape((item['num_boxes'],-1))
            in_data[item['image_id']] = item
            break
    return in_data

