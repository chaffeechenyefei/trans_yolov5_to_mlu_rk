import os
import cv2
import glob
from tqdm import tqdm
import random
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from utils.plots import plot_one_box
names2id = {'cat': 0, 'dog': 1}

def get_box(xml_path):
    xml_tree = ET.parse(xml_path)
    xml_root = xml_tree.getroot()
    target = xml_root.find('object').find('bndbox')
    xmin, ymin = target.find('xmin').text, target.find('ymin').text,
    xmax, ymax = target.find('xmax').text, target.find('ymax').text
    xyxy = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
    return xyxy

def make_origdataset(data_dir, save_dir):
    pic_list = glob.glob(os.path.join(data_dir, '*.jpg'))
    for idx, pic_path in enumerate(pic_list):
        print(idx, pic_path)
        cls = os.path.basename(pic_path).split('.')[0]
        img = cv2.imread(pic_path) #(height, width, channel)
        xml_path = pic_path.replace('jpg', 'xml')
        xyxy = get_box(xml_path)

        plot_one_box(xyxy, img, color=[0, 255, 0], line_thickness=1)
        cv2.imwrite(os.path.join(save_dir, os.path.basename(pic_path)), img)
        x, w = xyxy[0]/img.shape[1], (xyxy[2] - xyxy[0])/img.shape[1]
        y, h = xyxy[1]/img.shape[0], (xyxy[3] - xyxy[1])/img.shape[0]
        log = '{} {:.6f} {:.6f} {:.6f} {:.6f}'.format(names2id[cls], x, y, w, h)

def get_dogcat(pic_list, num=1):
    imgs_info = []
    random.shuffle(pic_list)
    for idx, pic_path in enumerate(pic_list[:num]):
        cls = os.path.basename(pic_path).split('.')[0]
        img = Image.open(pic_path) # img.size = (width, height)
        xyxy = get_box(pic_path.replace('jpg', 'xml'))
        imgs_info.append([img, names2id[cls], xyxy])
    return imgs_info

def crop_img(img, edge, xyxy, pixels):
    color_map = np.linspace(0, 255, edge).astype(np.int16)

    xyxy[0] = max(0, xyxy[0] - edge)
    xyxy[1] = max(0, xyxy[1] - edge)
    xyxy[2] = min(img.size[0], xyxy[2] + edge)
    xyxy[3] = min(img.size[1], xyxy[3] + edge)
    img = img.crop(xyxy)

    width, height = img.size
    scale = mask_pixels / min(width, height)
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
    img = Image.fromarray(img_data.astype('uint8'))

    return img

def paste_img(mask_infos, img, mask_pixels, edge):
    merged_img = img.copy().convert('RGBA')
    boxes = []
    for mask_info in mask_infos:
        mask_img = mask_info[0]
        xyxy = mask_info[2].astype(np.int16)
        croped_img = crop_img(mask_img, edge, xyxy, mask_pixels)
        mask_width, mask_height = croped_img.size

        index_x = random.randint(0, img.size[0]-mask_width)
        index_y = random.randint(0, img.size[1]-mask_height)
        merged_img.paste(croped_img, (index_x, index_y), croped_img)
        x = round(index_x / merged_img.size[0], 6)
        y = round(index_y / merged_img.size[1], 6)
        w = round(mask_width / merged_img.size[0], 6)
        h = round(mask_height / merged_img.size[1], 6)
        box = [mask_info[1], x, y, w, h]
        boxes.append(box)
    return merged_img, boxes

if __name__ == '__main__':
    edge = 12
    mask_pixels = 30 + 2 * edge
    task = 'train'
    mask_dir = '/mnt/ActionRecog/dataset/DETRAC_dogcat/Asirra: cat vs dogs'
    data_path = '/mnt/ActionRecog/dataset/DETRAC_coco/val.txt'
    save_dir = f'/mnt/ActionRecog/dataset/DETRAC_dogcat'
    mask_list = glob.glob(os.path.join(mask_dir, '*.jpg'))

    with open(data_path, 'r') as f:
        img_paths = f.read().strip().splitlines()

    txt_save_path = os.path.join(save_dir, f'{task}.txt')
    with open(txt_save_path, 'w') as t:
        for idx, img_path in tqdm(enumerate(img_paths), desc='Generating...'):
            bn = os.path.basename(img_path).split('.')[0]
            img = Image.open(img_path)
            mask_infos = get_dogcat(pic_list=mask_list, num=random.randint(1, 5))
            merged_img, boxes = paste_img(mask_infos, img, mask_pixels, edge)
            img_save_path = os.path.join(save_dir, 'images', task, bn+'.jpg')
            merged_img =merged_img.convert('RGB')
            merged_img.save(img_save_path, quality=95)
            t.writelines(img_save_path + '\n')

            labels_save_path = os.path.join(save_dir, 'labels', task, bn+'.txt')
            with open(labels_save_path, 'w') as f:
                for box in boxes:
                    cls, x, y, w, h = box
                    log = '{} {:.6f} {:.6f} {:.6f} {:.6f}'.format(cls, x, y, w, h)
                    f.writelines(log + '\n')