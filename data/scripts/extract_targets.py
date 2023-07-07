import os
import numpy as np
import glob
import shutil
import cv2
from tqdm import tqdm
import yaml

targets_names = ['cat', 'dog', 'bicycle', 'motorcycle']
save_dir = '/mnt/ActionRecog/dataset/custom_dataset/targets'

def crop_targets(img, boxes, trans=False):
    if np.max(boxes) < 1:
        boxes[:, [0, 2]] *= img.shape[1]
        boxes[:, [1, 3]] *= img.shape[0]
    if trans:
        xyxy = np.zeros_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2
        boxes = xyxy.astype(np.int16)
    cropImg_list = []
    for box in boxes:
        cropImg = img[box[1]:box[3], box[0]:box[2], :]
        cropImg_list.append(cropImg)
    return cropImg_list

def extract_from_coco(data_fp):
    with open(data_fp) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    names = data['names']
    path_list = []
    with open(data['train'], 'r') as f:
        train_list = f.read().strip().splitlines()
        path_list.extend([os.path.join(os.path.dirname(data_fp), p) for p in train_list])
    with open(data['val'], 'r') as f:
        val_list = f.read().strip().splitlines()
        path_list.extend([os.path.join(os.path.dirname(data_fp), p) for p in val_list])
    for target in targets_names:
        os.makedirs(os.path.join(save_dir, target), exist_ok=True)
    for pic_path in tqdm(path_list, desc='Generating...'):
        if os.path.isfile(pic_path):
            label_path = pic_path.replace('images', 'labels').replace('jpg', 'txt')
            img = cv2.imread(pic_path) #(height, width, channel)
            if os.path.isfile(label_path):
                labels = np.loadtxt(label_path).astype(np.float32) #(cls, center_x, center_y, width, height)
                labels = labels.reshape(-1, labels.shape[-1])

                for target in targets_names:
                    target_index = names.index(target)
                    target_labels = labels[labels[:, 0] == target_index]
                    if len(target_labels) > 0:
                        cropImg_list = crop_targets(img, target_labels[:, 1:], trans=True)
                        for cropImg in cropImg_list:
                            pic_save_dir = os.path.join(save_dir, target)
                            pic_index = len(os.listdir(pic_save_dir))
                            pic_save_fp = os.path.join(pic_save_dir, str(pic_index).rjust(6, '0')+'.jpg')
                            if min(cropImg.shape[:2]) < 30:
                                continue
                            try:
                                cv2.imwrite(pic_save_fp, cropImg)
                            except:
                                continue

if __name__ == '__main__':
    data_type = 'coco'
    data_fp = '/mnt/ActionRecog/dataset/coco/coco/data.yaml'

    if data_type == 'coco':
        extract_from_coco(data_fp)