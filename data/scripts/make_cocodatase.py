import os
import sys
sys.path.append('/mnt/ActionRecog/yolov5')
import glob
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
target2id = {'bicycle': 1, 'motorcycle': 3, 'dog': 16, 'cat': 15}
id2target = {value: key for key, value in target2id.items()}

def check_iou(rect1, rect2_list, thresh=0.3):
    # rect = (x, y, w, h)
    if len(rect2_list) == 0:
        return True
    else:
        for rect2 in rect2_list:
            top_line = max(rect1[1], rect2[1])
            bottom_line = min(rect1[1]+rect1[3], rect2[1]+rect2[3])
            left_line = max(rect1[0], rect2[0])
            right_line = min(rect1[0]+rect1[2], rect2[0]+rect2[2])
            if left_line >= right_line or top_line >= bottom_line:
                continue
            else:
                intersect = (right_line - left_line) * (bottom_line - top_line)
                sum_area = rect1[2]*rect1[3] +rect2[2]*rect2[3]
                iou = intersect / (sum_area - intersect)
                if iou < thresh:
                    continue
                else:
                    return False
        return True

def augmentation(img, aug=0.3):
    if random.random() < aug:
        img = np.fliplr(img)
    if random.random() < aug:
        img = np.flipud(img)
    if random.random() < aug:
        noise = np.random.normal(loc=0, scale=0.3, size=img.shape)
        img = img + noise
    if random.random() < aug:
        bright = random.randint(-10, 10)
        img[:, :, :3] = img[:, :, :3] + bright
    img = img.astype(np.float32)
    return img

def get_mask(cls_list, mask_dict):
    imgs_info = []
    for cls in cls_list:
        mask_path = random.choice(mask_dict[cls])
        img = Image.open(mask_path)  # img.size = (width, height)
        while min(img.size) < 30:
            mask_path = random.choice(mask_dict[cls])
            img = Image.open(mask_path) # img.size = (width, height)
        imgs_info.append([img, target2id[cls]])
    return imgs_info

def crop_img(img, edge, ratio):
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
    aug_img_data = augmentation(img_data)
    img = Image.fromarray(aug_img_data.astype('uint8'))

    return img

def paste_img(mask_infos, img, ratio, edge):
    merged_img = img.copy().convert('RGBA')
    boxes = []
    check_list = []
    for mask_info in mask_infos:
        mask_img = mask_info[0]
        croped_img = crop_img(mask_img, edge, ratio[mask_info[1]])
        mask_width, mask_height = croped_img.size
        index_x = random.randint(0, img.size[0]-mask_width)
        index_y = random.randint(0, img.size[1]-mask_height)
        paste_flg = True
        while paste_flg:
            if check_iou([index_x, index_y, mask_width, mask_height], check_list):
                merged_img.paste(croped_img, (index_x, index_y), croped_img)
                check_list.append([index_x, index_y, mask_width, mask_height])
                paste_flg = False
            else:
                index_x = random.randint(0, img.size[0] - mask_width)
                index_y = random.randint(0, img.size[1] - mask_height)
        x = round((index_x + mask_width/2) / merged_img.size[0], 6)
        y = round((index_y + mask_height/2) / merged_img.size[1], 6)
        w = round(mask_width / merged_img.size[0], 6)
        h = round(mask_height / merged_img.size[1], 6)
        box = [mask_info[1], x, y, w, h]
        boxes.append(box)
    return merged_img, boxes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='train')
    opt = parser.parse_args()
    task = opt.task
    edge = 8
    ratio = {15: [30, 50], 16: [30, 50],
             1: [70, 120], 3: [70, 120]}
    mask_dir = '/mnt/ActionRecog/dataset/custom_dataset/targets'
    data_path = f'/mnt/ActionRecog/dataset/DETRAC_coco/{task}2021.txt'
    save_dir = f'/mnt/ActionRecog/dataset/custom_dataset/'
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
            bn = f"{img_path.split('/')[-2]}_{img_path.split('/')[-1].split('.')[0]}"
            img_save_path = os.path.join(save_dir, 'images', task, bn + '.jpg')
            img = Image.open(img_path)
            mask_infos = get_mask(cls_list=random.choices(paste_cls, k=random.randint(3, 7)), mask_dict=mask_dict)
            merged_img, boxes = paste_img(mask_infos, img, ratio, edge)

            merged_img =merged_img.convert('RGB')
            merged_img.save(img_save_path, quality=95)
            t.writelines(img_save_path + '\n')

            ori_fp = os.path.join('/mnt/ActionRecog/dataset/DETRAC_coco/labels/train', bn + '.txt')
            if os.path.isfile(ori_fp):
                ori_label = np.loadtxt(ori_fp).reshape(-1, 5)
                ori_label[:, 1] = ori_label[:, 1] + 0.5 * ori_label[:, 3]
                ori_label[:, 2] = ori_label[:, 2] + 0.5 * ori_label[:, 4]
                ori_label[:, 0] = 2
                boxes = np.concatenate([np.array(boxes), ori_label], axis=0)
            labels_save_path = os.path.join(save_dir, 'labels', task, bn+'.txt')
            with open(labels_save_path, 'w') as f:
                for box in boxes:
                    cls, x, y, w, h = box
                    log = '{} {:.6f} {:.6f} {:.6f} {:.6f}'.format(cls, x, y, w, h)
                    f.writelines(log + '\n')