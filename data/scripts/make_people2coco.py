import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import shutil
import glob
from scipy.io import loadmat
from tqdm import tqdm


def parse_xml(xml_fp):
    tree = ET.parse(xml_fp)
    root = tree.getroot()
    objs = root.findall('object')
    coords = list()
    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)
    img_size = [img_width, img_height]

    for ix, obj in enumerate(objs):
        name = obj.find('name').text
        name = 0
        box = obj.find('bndbox')
        x_min = int(box[0].text)
        y_min = int(box[1].text)
        x_max = int(box[2].text)
        y_max = int(box[3].text)
        coords.append([name, x_min, y_min, x_max, y_max])
    return np.array(coords).astype(np.float32).reshape(-1, 5)


def plot_bbox(img, bboxes):
    H, W, C = img.shape
    color = [0, 0, 255]
    tl = 2
    for bbox in bboxes:
        c1 = int((bbox[1] - bbox[3] / 2) * W), int((bbox[2] - bbox[4] / 2) * H)
        c2 = int((bbox[1] + bbox[3] / 2) * W), int((bbox[2] + bbox[4] / 2) * H)
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    return img


def parse_seq(seq_fp, jpeg_save_dir):
    f = open(seq_fp, 'rb+')
    # 将seq文件的内容转化成str类型
    string = f.read().decode('latin-1')

    # splitstring是图片的前缀，可以理解成seq是以splitstring为分隔的多个jpg合成的文件
    splitstring = "\xFF\xD8\xFF\xE0\x00\x10\x4A\x46\x49\x46"

    strlist = string.split(splitstring)
    f.close()

    for idx, img in enumerate(strlist[1:]):
        # abandon the first one, which is filled with .seq header
        jpeg_fn = f"{seq_fp.split('/')[-2]}_{seq_fp.split('/')[-1].split('.')[0]}_{idx}.jpg"
        jpeg_fp = os.path.join(jpeg_save_dir, jpeg_fn)

        with open(jpeg_fp, 'wb+') as f:
            f.write(splitstring.encode('latin-1'))
            f.write(img.encode('latin-1'))


def parse_vbb(vbb_fp, anno_save_dir):
    vbb = loadmat(vbb_fp)
    # object info in each frame: id, pos, occlusion, lock, posv
    objLists = vbb['A'][0][0][1][0]
    for idx, obj in enumerate(objLists):
        bbox = []
        if len(obj) > 0:
            for x in obj[0]:
                bbox.append(x[1][0])
        bbox = np.array(bbox)

        anno_fn = f"{vbb_fp.split('/')[-2]}_{vbb_fp.split('/')[-1].split('.')[0]}_{idx}.txt"
        anno_fp = os.path.join(anno_save_dir, anno_fn)
        np.savetxt(anno_fp, bbox)


def parse_txt(txt_fp):
    bboxes = np.loadtxt(txt_fp).reshape(-1, 4).tolist()
    coords = []
    for bbox in bboxes:
        name = 0
        x_min, y_min, w, h = bbox
        coords.append([name, x_min, y_min, w, h])

    return np.array(coords).astype(np.float32)


# upper_body
dataset = 'train'

data_dir = '/data/dataset/yolo_dataset/people_detection/upper_body'
anno_dir = os.path.join(data_dir, 'dataset', 'Annotations')
pic_dir = os.path.join(data_dir, 'dataset', 'JPEGImages')

img_dir = os.path.join(data_dir, 'images', dataset)
labels_dir = os.path.join(data_dir, 'labels', dataset)
bboximg_dir = os.path.join(data_dir, 'bboximg', dataset)
os.makedirs(img_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)
os.makedirs(bboximg_dir, exist_ok=True)

pic_list = os.listdir(pic_dir)
train_val_split = int(len(pic_list) * 0.8)
pic_list = pic_list[:train_val_split] if dataset == 'train' else pic_list[train_val_split:]
print(len(pic_list))
fp_file = open(os.path.join(data_dir, f'{dataset}.txt'), 'w')
for pic_fn in tqdm(pic_list):
    pic_fp = os.path.join(pic_dir, pic_fn)
    labels_fp = os.path.join(anno_dir, pic_fn.split('.')[0] + '.xml')

    img = cv2.imread(pic_fp)
    img_width, img_height = img.shape[1], img.shape[0]
    bboxes = parse_xml(labels_fp)

    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 4] = bboxes[:, 4] - bboxes[:, 2]
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] / 2
    bboxes[:, 2] = bboxes[:, 2] + bboxes[:, 4] / 2

    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] / img_width
    bboxes[:, [2, 4]] = bboxes[:, [2, 4]] / img_height

    cv2.imwrite(os.path.join(img_dir, pic_fn), img)

    bbox_img = plot_bbox(img, bboxes.tolist())
    cv2.imwrite(os.path.join(bboximg_dir, pic_fn), bbox_img)

    np.savetxt(os.path.join(labels_dir, pic_fn.replace('jpg', 'txt')), bboxes)

    fp_file.writelines(os.path.join(img_dir, pic_fn) + '\n')

fp_file.close()


# Caltech
dataset = 'val'

data_dir = '/data/dataset/yolo_dataset/people_detection/Caltech'
vbb_dir = os.path.join(data_dir, 'dataset', 'annotations')
seq_dir = os.path.join(data_dir, 'dataset', 'images')

anno_dir = os.path.join(data_dir, 'dataset', 'Annotations')
pic_dir = os.path.join(data_dir, 'dataset', 'JPEGImages')

img_dir = os.path.join(data_dir, 'images', dataset)
labels_dir = os.path.join(data_dir, 'labels', dataset)
bboximg_dir = os.path.join(data_dir, 'bboximg', dataset)
os.makedirs(img_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)
os.makedirs(bboximg_dir, exist_ok=True)

pic_list = os.listdir(pic_dir)[::10]  # 视频抽帧
train_val_split = int(len(pic_list) * 0.8)
pic_list = pic_list[:train_val_split] if dataset == 'train' else pic_list[train_val_split:]
print(len(pic_list))

# parse seq and vbb
seq_list = glob.glob(os.path.join(seq_dir, '*/*.seq'))
jpeg_save_dir = os.path.join(data_dir, 'dataset', 'JPEGImages')
anno_save_dir = os.path.join(data_dir, 'dataset', 'Annotations')
for seq_fp in tqdm(seq_list):
    #     print(seq_fp)
    parse_seq(seq_fp, jpeg_save_dir)

    vbb_fp = seq_fp.replace('images', 'annotations').replace('.seq', '.vbb')
    parse_vbb(vbb_fp, anno_save_dir)

fp_file = open(os.path.join(data_dir, f'{dataset}.txt'), 'w')
for pic_fn in tqdm(pic_list):
    pic_fp = os.path.join(pic_dir, pic_fn)
    #     print(pic_fp)
    labels_fp = os.path.join(anno_dir, pic_fn.split('.')[0] + '.txt')

    img = cv2.imread(pic_fp)
    img_width, img_height = img.shape[1], img.shape[0]
    if os.path.getsize(labels_fp):
        bboxes = parse_txt(labels_fp)

        bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] / 2
        bboxes[:, 2] = bboxes[:, 2] + bboxes[:, 4] / 2

        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] / img_width
        bboxes[:, [2, 4]] = bboxes[:, [2, 4]] / img_height

        cv2.imwrite(os.path.join(img_dir, pic_fn), img)

        bbox_img = plot_bbox(img, bboxes.tolist())
        cv2.imwrite(os.path.join(bboximg_dir, pic_fn), bbox_img)

        np.savetxt(os.path.join(labels_dir, pic_fn.replace('jpg', 'txt')), bboxes)

        fp_file.writelines(os.path.join(img_dir, pic_fn) + '\n')

fp_file.close()


# wework_fmt
dataset = 'val'

data_dir = '/data/dataset/yolo_dataset/people_detection/wework_fmt'
pic_dir = os.path.join(data_dir, 'dataset', 'Data')

img_dir = os.path.join(data_dir, 'images', dataset)
labels_dir = os.path.join(data_dir, 'labels', dataset)
bboximg_dir = os.path.join(data_dir, 'bboximg', dataset)
os.makedirs(img_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)
os.makedirs(bboximg_dir, exist_ok=True)

pic_list = glob.glob(os.path.join(pic_dir, '*/*/JPEGImages/*.jpg'))
anno_list = glob.glob(os.path.join(pic_dir, '*/*/labels/*.txt'))
print(len(pic_list), len(anno_list))
train_val_split = int(len(pic_list) * 0.8)
pic_list = pic_list[:train_val_split] if dataset == 'train' else pic_list[train_val_split:]
print(len(pic_list))

fp_file = open(os.path.join(data_dir, f'{dataset}.txt'), 'w')
for pic_fp in tqdm(pic_list):
    #     print(pic_fp)
    pic_fn = os.path.basename(pic_fp)
    labels_fp = pic_fp.replace('JPEGImages', 'labels').replace('jpg', 'txt')

    img = cv2.imread(pic_fp)
    bboxes = np.loadtxt(labels_fp).reshape(-1, 5)
    bboxes[:, 0] = 0

    cv2.imwrite(os.path.join(img_dir, pic_fn), img)

    bbox_img = plot_bbox(img, bboxes.tolist())
    cv2.imwrite(os.path.join(bboximg_dir, pic_fn), bbox_img)

    np.savetxt(os.path.join(labels_dir, pic_fn.replace('jpg', 'txt')), bboxes)

    fp_file.writelines(os.path.join(img_dir, pic_fn) + '\n')

fp_file.close()