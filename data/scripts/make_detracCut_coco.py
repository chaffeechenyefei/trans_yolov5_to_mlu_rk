import os
import sys
sys.path.append('/mnt/ActionRecog/yolov5')
import glob
import cv2
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
from data.scripts.augmentation import Pipeline

pipeline = Pipeline()
target2id = {'bicycle': 1, 'motorcycle': 3, 'dog': 16, 'cat': 15}
id2target = {value: key for key, value in target2id.items()}

def check_iou(rect1, rect2_list, thresh=0.1):
    # rect = (top_left_x, top_left_y, w, h)
    if len(rect2_list) == 0:
        return True
    else:
        for ii, rect2 in enumerate(rect2_list):
            top_line = max(rect1[1], rect2[1])
            bottom_line = min(rect1[1]+rect1[3], rect2[1]+rect2[3])
            left_line = max(rect1[0], rect2[0])
            right_line = min(rect1[0]+rect1[2], rect2[0]+rect2[2])
            if left_line >= right_line or top_line >= bottom_line:
                continue
            else:
                intersect = (right_line - left_line) * (bottom_line - top_line)
                sum_area = rect1[2]*rect1[3] + rect2[2]*rect2[3]
                iou = intersect / (sum_area - intersect)
                if iou < thresh:
                    continue
                else:
                    return False
        return True

def get_mask(cls_list, mask_dict):
    # According class name list, random chioce class image, return [img, class_id]
    imgs_info = []
    for cls in cls_list:
        mask_path = random.choice(mask_dict[cls])
        img = Image.open(mask_path)  # img.size = (width, height)
        while min(img.size) < 30:
            mask_path = random.choice(mask_dict[cls])
            img = Image.open(mask_path) # img.size = (width, height)
        imgs_info.append([img, target2id[cls]])
    return imgs_info

def crop_img(img, edge, ratio, training=False):
    color_map = np.linspace(0, 255, edge).astype(np.int16)

    width, height = img.size
    scale = random.randint(ratio[0], ratio[1]) / min(width, height)
    scale = scale if scale < 1 else 1
    new_width, new_height = int(width * scale), int(height * scale)
    img = img.resize((new_width, new_height))

    img = img.convert('RGBA')
    img_data = np.array(img)

    img_data[0, :, -1] = 0
    img_data[-1, :, -1] = 0
    img_data[:, 0, -1] = 0
    img_data[:, -1, -1] = 0
    for x in range(1, edge):
        img_data[x, x:-x, -1] = color_map[x]
        img_data[-1-x, x:-x, -1] = color_map[x]
        img_data[x:-x, x, -1] = color_map[x]
        img_data[x:-x, -1-x, -1] = color_map[x]
    aug_img_data = img_data.copy()
    if training:
        aug_img_data[:, :, :3] = pipeline(img_data[:, :, :3])
    img = Image.fromarray(aug_img_data.astype('uint8'))

    return img

def paste_img(mask_infos, img, ratio, edge, label):
    merged_img = img.copy().convert('RGBA')
    boxes = []
    # check_list = label[:, 1:].tolist() if len(label) else []
    check_list = label[:, 1:].tolist()
    for qqq, mask_info in enumerate(mask_infos):
        mask_img = mask_info[0]
        croped_img = crop_img(mask_img, edge, ratio[mask_info[1]], training=True if opt.task=='train' else False)
        mask_width, mask_height = croped_img.size
        index_x = random.randint(0, img.size[0]-mask_width)
        index_y = random.randint(0, img.size[1]-mask_height)
        paste_flg = True
        cycle_time = 0
        while paste_flg:
            if check_iou([index_x, index_y, mask_width, mask_height], check_list):
                merged_img.paste(croped_img, (index_x, index_y), croped_img)
                check_list.append([index_x, index_y, mask_width, mask_height])
                paste_flg = False
            else:
                index_x = random.randint(0, img.size[0] - mask_width)
                index_y = random.randint(0, img.size[1] - mask_height)
                cycle_time += 1
                if cycle_time >= 200:
                    break
        x = round((index_x + mask_width/2) / merged_img.size[0], 6)
        y = round((index_y + mask_height/2) / merged_img.size[1], 6)
        w = round(mask_width / merged_img.size[0], 6)
        h = round(mask_height / merged_img.size[1], 6)
        box = [mask_info[1], x, y, w, h]
        boxes.append(box)
    for l_box in label:
        x = round((l_box[1] + l_box[3] / 2) / merged_img.size[0], 6)
        y = round((l_box[2] + l_box[4] / 2) / merged_img.size[1], 6)
        w = round(l_box[3] / merged_img.size[0], 6)
        h = round(l_box[4] / merged_img.size[1], 6)
        box = [l_box[0], x, y, w, h]
        boxes.append(box)
    return merged_img, boxes

def cluster_boxes(img, boxes, edge=30):
    if len(boxes) == 0:
        cluster_img = img
        cluster_boxes = boxes
    else:
        img = np.array(img)
        img_H, img_W, _ = img.shape
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * img_W
        boxes[:, [2, 4]] = boxes[:, [2, 4]] * img_H
        left_line = max(int(np.min(boxes[:, 1])) - edge, 0)
        top_line = max(int(np.min(boxes[:, 2])) - edge, 0)
        right_line = int(np.max(boxes[:, 1] + boxes[:, 3])) + edge
        bottom_line = int(np.max(boxes[:, 2] + boxes[:, 4])) + edge
        cluster_img = img[top_line: bottom_line, left_line: right_line, :]
        new_H, new_W, _ = cluster_img.shape
        boxes[:, 1] = boxes[:, 1] - left_line
        boxes[:, 2] = boxes[:, 2] - top_line
        cluster_boxes = boxes
        cluster_img = Image.fromarray(cluster_img.astype('uint8'))
    return cluster_img, cluster_boxes

def plot_bbox(img, bboxes):
    H, W, C = img.shape
    color=[0, 0, 255]
    tl = 2
    for bbox in bboxes:
        c1 = int((bbox[1]-bbox[3]/2) * W), int((bbox[2]-bbox[4]/2) * H)
        c2 = int((bbox[1]+bbox[3]/2) * W), int((bbox[2]+bbox[4]/2) * H)
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='test')
    opt = parser.parse_args()
    task = opt.task
    edge = 8
    ratio = {15: [30, 50], 16: [30, 50],
             1: [30, 100], 3: [30, 100]}
    mask_dir = '/mnt/ActionRecog/dataset/custom_dataset/targets'
    data_path = f'/mnt/ActionRecog/dataset/DETRAC_coco/{task}2021.txt'
    save_dir = f'/data/dataset/yolo_dataset/detracCut_coco'
    paste_cls = ['bicycle', 'motorcycle', 'dog', 'cat']

    with open(data_path, 'r') as f:
        img_paths = f.read().strip().splitlines()

    mask_dict = {'train': {}, 'val': {}, 'test': {}}
    for cls in paste_cls:
        mask_list = glob.glob(os.path.join(mask_dir, cls, '*.jpg'))
        print('{}: {}'.format(cls, len(mask_list)))
        mask_dict['train'][cls] = mask_list[: int(0.6*len(mask_list))]
        mask_dict['val'][cls] = mask_list[int(0.6*len(mask_list)): int(0.8*len(mask_list))]
        mask_dict['test'][cls] = mask_list[int(0.8*len(mask_list)): ]
    mask_dict = mask_dict[task]

    txt_save_path = os.path.join(save_dir, f'{task}.txt')
    with open(txt_save_path, 'w') as t:
        for idx, img_path in tqdm(enumerate(img_paths), desc='Generating...'):
            try:
                bn = f"{img_path.split('/')[-1].split('.')[0]}"
                img_save_path = os.path.join(save_dir, 'images', task, bn + '.jpg')
                img = Image.open(img_path) # img.size = Width, Height
                ori_fp = os.path.join('/mnt/ActionRecog/dataset/DETRAC_coco/labels/train', bn + '.txt')
                # if os.path.isfile(ori_fp):
                ori_label = np.loadtxt(ori_fp).reshape(-1, 5)
                ori_label[:, 0] = 2
                # else:
                #     ori_label = np.array([])

                cluster_img, label = cluster_boxes(img, ori_label)
                mask_infos = get_mask(cls_list=random.choices(paste_cls, k=random.randint(3, 7)), mask_dict=mask_dict)
                merged_img, boxes = paste_img(mask_infos, cluster_img, ratio, edge, label)

                merged_img = merged_img.convert('RGB')
                merged_img.save(img_save_path, quality=95)
                t.writelines(img_save_path + '\n')

                bbox_img_save_path = os.path.join(save_dir, 'img_bbox', task, bn + '.jpg')
                bbox_img = np.array(merged_img)
                bbox_img = plot_bbox(bbox_img, boxes)
                cv2.imwrite(bbox_img_save_path, bbox_img)

                labels_save_path = os.path.join(save_dir, 'labels', task, bn+'.txt')
                with open(labels_save_path, 'w') as f:
                    for box in boxes:
                        cls, x, y, w, h = box
                        log = '{} {:.6f} {:.6f} {:.6f} {:.6f}'.format(cls, x, y, w, h)
                        f.writelines(log + '\n')

            except:
                continue