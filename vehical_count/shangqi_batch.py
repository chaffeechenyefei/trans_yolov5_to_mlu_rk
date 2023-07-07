#!/usr/bin/env python
# coding: utf-8

# In[14]:


import os
import cv2
import xlwt
import torch
import numpy as np
from shangqi_utils.img_stitcher import Image_Stitching

from models.experimental import attempt_load
from utils.datasets import LoadPairImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box, plot_car_num

# In[15]:


font = xlwt.Font()
font.name = 'Times New Roman'
font.height = 20 * 14

alignment = xlwt.Alignment()
alignment.horz = 0x02
alignment.vert = 0x01

style_normal = xlwt.XFStyle()
style_normal.font = font
style_normal.alignment = alignment


def xls_save(path, num_list, car_num):
    current_time = os.path.basename(path)[:-10]

    workbook = xlwt.Workbook(encoding='utf-8')

    worksheet = workbook.add_sheet('sheet1')
    worksheet.col(0).width = 25 * 256
    for i in range(1, 4):
        worksheet.col(i).width = 20 * 256

    worksheet.write(0, 0, label='Time', style=style_normal)
    worksheet.write(0, 1, label='Camera 1', style=style_normal)
    worksheet.write(0, 2, label='Camera 2', style=style_normal)
    worksheet.write(0, 3, label='Camera merge', style=style_normal)

    worksheet.write(1, 0, label=current_time, style=style_normal)
    worksheet.write(1, 1, label=num_list[0], style=style_normal)
    worksheet.write(1, 2, label=num_list[1], style=style_normal)
    worksheet.write(1, 3, label=car_num, style=style_normal)

    workbook.save(os.path.join(os.path.dirname(path), current_time + '.xls'))


# In[16]:


test_dir = '/mnt/ActionRecog/dataset/car_count/shangqi_test/'
save_dir = '/mnt/ActionRecog/dataset/car_count/shangqi_test_results/'
os.makedirs(save_dir, exist_ok=True)

names = ['people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
target_names = ['car', 'van', 'truck', 'bus']

img_size = 640 * 2
conf_thresh = 0.4
iou_thresh = 0.45

# In[17]:


device = torch.device('cuda:0')
weights_fp = '/mnt/ActionRecog/weights/yolo_weights/UAV_VisDrone_bdd100k/weights/best.pt'

half = device.type != 'cpu'  # half precision only supported on CUDA
# Load model
model = attempt_load(weights_fp, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride 64
img_size = check_img_size(img_size, s=stride)  # check img_size
if half:
    model.half()  # to FP16
if device.type != 'cpu':
    model(torch.zeros(1, 3, 960, 1280).to(device).type_as(next(model.parameters())))  # run once
    img = torch.zeros((1, 3, img_size, img_size), dtype=torch.float32).to(device).half()
    yy = model(img)


dataset = LoadPairImages(test_dir, img_size=img_size, stride=stride, merge=False, mask=True)
stitcher = Image_Stitching()


for frame_id, (img_list, img0_list, path_list) in enumerate(dataset):
    if frame_id <= 34:
        continue
    car_num, merge_list, num_list = 0, [], []
    for img_id in range(len(img_list)):
        img, im0s, img_path = img_list[img_id], img0_list[img_id], path_list[img_id]
        img = img.astype(np.float32)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        output = model(img)[0]
        pred = non_max_suppression(output, conf_thresh, iou_thresh)
        det = pred[0]
        t_det = []
        for t in target_names:
            t_det.append(det[det[:, -1] == names.index(t)])
        det = torch.cat(t_det, dim=0)

        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

            for *xyxy, conf, cls in reversed(det):
                color = [0, 0, 255]
                plot_one_box(xyxy, im0s, label=None, color=color, line_thickness=2)

        save_path = os.path.join(save_dir, os.path.basename(img_path))
        if img_id == 0:
            merge_list.append(im0s.copy())
            num_list.append(len(det))
            car_num += len(det)
            plot_car_num(im0s, len(det), color=[255, 0, 0])
            cv2.imwrite(save_path, im0s)
        elif img_id == 1:
            merge_list.append(im0s.copy())
            car_num += len(det)
        elif img_id == 2:
            num_list.append(len(det))
            plot_car_num(im0s, len(det), color=[255, 0, 0])
            cv2.imwrite(save_path, im0s)

    pano_img = stitcher.blending(merge_list[0], merge_list[1])
    save_path = os.path.join(save_dir, os.path.basename(path_list[0].replace('_cam1.jpg', '_merge.png')))
    plot_car_num(pano_img, car_num, color=[255, 0, 0])
    cv2.imwrite(save_path, pano_img)

    xls_save(save_path, num_list, car_num)
