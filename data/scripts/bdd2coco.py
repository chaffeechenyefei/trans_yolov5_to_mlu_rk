import os
import numpy as np
import glob
import cv2
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

data_dir = '/data/dataset/yolo_dataset/bdd100k'
test_set = 'val'
with open(os.path.join(data_dir, f'det_{test_set}.json'), 'r') as f:
    label_json = json.load(f)

class2id = {'car': 2, 'pedestrian': 0, 'bus': 5, 'truck': 7, 'bicycle': 1, 'motorcycle': 3, 'train': 6}
pic_list = os.listdir(os.path.join(data_dir, 'images', '100k', test_set))
total_pic_fp = os.path.join(data_dir, f'{test_set}.txt')
file = open(total_pic_fp, 'w')

for pic_infos in tqdm(label_json):
    if 'labels' in pic_infos.keys():
        pic_labels = pic_infos['labels']
        pic_name = pic_infos['name']
        if pic_name in pic_list:
            pic_fp = os.path.join(data_dir, 'images', '100k', test_set, pic_name)
            img = cv2.imread(pic_fp)

            boxes = []
            for label in pic_labels:
                if label['category'] in class2id.keys():
                    class_id = class2id[label['category']]
                    x1 = float(label['box2d']['x1'])
                    y1 = float(label['box2d']['y1'])
                    x2 = float(label['box2d']['x2'])
                    y2 = float(label['box2d']['y2'])

                    x = (x2 + x1) / 2 / img.shape[1]
                    y = (y2 + y1) / 2 / img.shape[0]
                    w = (x2 - x1) / img.shape[1]
                    h = (y2 - y1) / img.shape[0]
                    box = [class_id, x, y, w, h]
                    boxes.append(box)
            if len(boxes):
                label_fp = os.path.join(data_dir, 'labels', '100k', test_set, pic_name.split('.')[0] + '.txt')
                np.savetxt(label_fp, boxes)
            file.writelines(pic_fp + '\n')
file.close()