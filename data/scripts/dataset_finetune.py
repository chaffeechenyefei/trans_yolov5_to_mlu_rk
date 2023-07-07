import os
import sys
sys.path.append('/mnt/ActionRecog/yolov5')
import cv2
import glob
import random
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
import argparse
from utils.torch_utils import select_device, load_classifier, time_synchronized

target2id = {'bicycle': 1, 'motorcycle': 3, 'dog': 16, 'cat': 15}
id2target = {value: key for key, value in target2id.items()}
imgsz = 640
conf_thres = 0.25
iou_thres = 0.45
devicel = '0'
device = select_device(devicel)

def get_box(txt_path):
    label_list = np.loadtxt(txt_path).astype(np.int16)
    if len(label_list.shape) == 1:
        c, *xyxy = label_list
    else:
        c, *xyxy = random.choice(label_list)
    return xyxy

def get_mask(cls_list, mask_dict):
    imgs_info = []
    for cls in cls_list:
        mask_path = random.choice(mask_dict[cls])
        img = Image.open(mask_path) # img.size = (width, height)
        xyxy = get_box(mask_path.replace('jpg', 'txt'))
        imgs_info.append([img, target2id[cls], xyxy])
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
        xyxy = mask_info[2]
        croped_img = crop_img(mask_img, edge, xyxy, mask_pixels)
        mask_width, mask_height = croped_img.size

        index_x = random.randint(0, img.size[0]-mask_width)
        index_y = random.randint(0, img.size[1]-mask_height)
        merged_img.paste(croped_img, (index_x, index_y), croped_img)
        x = round((index_x + mask_width/2) / merged_img.size[0], 6)
        y = round((index_y + mask_height/2) / merged_img.size[1], 6)
        w = round(mask_width / merged_img.size[0], 6)
        h = round(mask_height / merged_img.size[1], 6)
        box = [mask_info[1], x, y, w, h]
        boxes.append(box)
    return merged_img, boxes

def model_init():
    weights = './yolov5s.pt'
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Load model
    model = attempt_load(weights, map_location=device)
      # check img_size
    if half:
        model.half()  # to FP16

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    return model

def pred(img_fp, model):
    stride = int(model.stride.max())
    dataset = LoadImages(img_fp, img_size=imgsz, stride=stride, print_path=False, toy=False, get_label=False)
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        pred = model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)[0]
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], im0s.shape).round()
        pred = pred.cpu().numpy()
        t_preds = []
        for *xyxy, conf, cls in reversed(pred):
            x = (xyxy[0] + xyxy[2]) / 2 / im0s.shape[1]
            y = (xyxy[1] + xyxy[3]) / 2 / im0s.shape[0]
            w = (xyxy[2] - xyxy[0]) / im0s.shape[1]
            h = (xyxy[3] - xyxy[1]) / im0s.shape[0]
            t_preds.append([cls, x, y, w, h])
        return t_preds

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='test')
    opt = parser.parse_args()
    task = opt.task
    edge = 12
    mask_pixels = 30 + 2 * edge
    mask_dir = '/mnt/ActionRecog/dataset/coco_class'
    data_path = f'/mnt/ActionRecog/dataset/DETRAC_coco/{task}2021.txt'
    save_dir = f'/mnt/ActionRecog/dataset/finetune_coco/'
    paste_cls = ['bicycle', 'motorcycle', 'dog', 'cat']

    with open(data_path, 'r') as f:
        img_paths = f.read().strip().splitlines()

    mask_dict = {'train': {}, 'val': {}, 'test': {}}
    for cls in paste_cls:
        mask_list = glob.glob(os.path.join(mask_dir, cls, '*.jpg'))
        mask_dict['train'][cls] = mask_list[: int(0.6*len(mask_list))]
        mask_dict['val'][cls] = mask_list[int(0.6*len(mask_list)): int(0.8*len(mask_list))]
        mask_dict['test'][cls] = mask_list[int(0.8*len(mask_list)): ]
    mask_dict = mask_dict[task]

    txt_save_path = os.path.join(save_dir, f'{task}.txt')
    with open(txt_save_path, 'w') as t:
        for idx, img_path in tqdm(enumerate(img_paths), desc='Generating...'):
            bn = f"{img_path.split('/')[-2]}_{img_path.split('/')[-1].split('.')[0]}"
            img_save_path = os.path.join(save_dir, 'images', task, bn + '.jpg')
            t.writelines(img_save_path + '\n')

            img = Image.open(img_path)
            mask_infos = get_mask(cls_list=random.choices(paste_cls, k=random.randint(3, 7)), mask_dict=mask_dict)
            merged_img, boxes = paste_img(mask_infos, img, mask_pixels, edge)

            model = model_init()
            preds = pred(img_path, model)
            boxes.extend(preds)

            merged_img =merged_img.convert('RGB')
            merged_img.save(img_save_path, quality=95)

            labels_save_path = os.path.join(save_dir, 'labels', task, bn+'.txt')
            with open(labels_save_path, 'w') as f:
                for box in boxes:
                    cls, x, y, w, h = box
                    log = '{} {:.6f} {:.6f} {:.6f} {:.6f}'.format(cls, x, y, w, h)
                    f.writelines(log + '\n')