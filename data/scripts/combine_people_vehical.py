import os
import cv2
import tqdm
import torch
import numpy as np
from utils.datasets import LoadListImages
from evaluate import Detect
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.plots import plot_bbox

names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]

target_names = [ 'person', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motorcycle' ]


mode = 'train'
save_dir = '/data/dataset/yolo_dataset/people_vehical_detection/wework_cocopeople_UAV_VisDrone_bdd100k'
bboximg_save_dir = os.path.join(save_dir, 'bboximg', mode)

coco_file = f'/mnt/ActionRecog/dataset/coco/coco/{mode}2017.txt'
coco_list = [os.path.basename(fp) for fp in np.loadtxt(coco_file, dtype=str)]
coco_dir = f'/mnt/ActionRecog/dataset/coco/coco/labels/{mode}2017'

device = torch.device('cuda:0')
img_dir = os.path.join(save_dir, 'images', mode)
img_list = []
for fn in os.listdir(img_dir):
    if os.path.splitext(fn)[-1] in ['.jpg', '.png']:
        img_list.append(os.path.join(img_dir, fn))
dataset = LoadListImages(img_list, img_size=1280, stride=64, print_path=False)

fp_file = open(os.path.join(save_dir, f'{mode}.txt'), 'w')
for batch_i, (img_path, img, im0s, vid_cap) in tqdm.tqdm(enumerate(dataset)):
    if os.path.basename(img_path) in coco_list:
        labels_fp = img_path.replace('images', 'labels').replace('jpg', 'txt')

        coco_labels_fp = os.path.join(coco_dir, os.path.basename(img_path).replace('jpg', 'txt'))
        coco_labels = np.loadtxt(coco_labels_fp).reshape(-1, 5)

        t_labels = []
        for t in target_names:
            if t not in names:
                continue
            temp = coco_labels[coco_labels[:, 0] == names.index(t)]
            temp[:, 0] = target_names.index(t)
            t_labels.append(temp)
        labels = np.concatenate(t_labels, axis=0).reshape(-1, 5)
        np.savetxt(labels_fp, labels)

    labels_fp = img_path.replace('images', 'labels').replace('jpg', 'txt')
    if os.path.isfile(labels_fp):
        boxes = np.loadtxt(labels_fp)
        bbox_img = plot_bbox(im0s, boxes.reshape(-1, 5).tolist())
        cv2.imwrite(os.path.join(bboximg_save_dir, os.path.basename(img_path)), bbox_img)
        fp_file.writelines(img_path + '\n')