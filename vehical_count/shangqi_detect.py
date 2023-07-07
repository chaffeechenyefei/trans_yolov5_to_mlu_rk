import os
import cv2
import torch
import numpy as np
import shutil
import copy
from shangqi_utils.img_stitcher import Image_Stitching
from shangqi_utils.xls_utils import xlsx_save

from models.experimental import attempt_load
from utils.datasets import LoadPairImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box, plot_car_num

class Detect():
    def __init__(self, img_dir, save_dir, save_dir_bk=None):
        self.img_dir = img_dir
        self.save_dir = save_dir
        self.save_dir_bk = save_dir_bk

        self.names = ['people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
        self.target_names = ['car', 'van', 'truck', 'bus']

        self.img_size = 640 * 2
        self.conf_thresh = 0.4
        self.iou_thresh = 0.45

        self.device = torch.device('cuda:0')
        weights_fp = '/mnt/ActionRecog/weights/yolo_weights/UAV_VisDrone_bdd100k/weights/best.pt'

        self.model = attempt_load(weights_fp, map_location=self.device)
        self.stride = int(self.model.stride.max())
        self.img_size = check_img_size(self.img_size, s=self.stride)
        if self.device.type != 'cpu':
            self.model.half()
            self.model(torch.zeros(1, 3, 960, 1280).to(self.device).type_as(next(self.model.parameters())))  # run once
            img = torch.zeros((1, 3, self.img_size, self.img_size), dtype=torch.float32).to(self.device).half()
            yy = self.model(img)

        self.stitcher = Image_Stitching()
        self.clear()

    def inference(self, prefix='', saveImg=False, saveOri=False, save_xlsx=True):
        dataset = LoadPairImages(self.img_dir, img_size=self.img_size, stride=self.stride,
                                 merge=False, mask=True, suffix=['1', '2'], print_path=False)
        for frame_id, (img_list, img0_list, path_list) in enumerate(dataset):
            car_num, merge_list, num_list, merge0_list = 0, [], [], []
            for img_id in range(len(img_list)):
                img, im0s, img_path = img_list[img_id], img0_list[img_id], path_list[img_id]
                img = img.astype(np.float32)
                img = torch.from_numpy(img).to(self.device)
                img = img.half()
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                output = self.model(img)[0]
                pred = non_max_suppression(output, self.conf_thresh, self.iou_thresh)
                det = pred[0]
                t_det = []
                for t in self.target_names:
                    t_det.append(det[det[:, -1] == self.names.index(t)])
                det = torch.cat(t_det, dim=0)

                if saveOri:
                    im0s_cp = copy.deepcopy(im0s)
                    merge0_list.append(im0s_cp)

                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

                    for *xyxy, conf, cls in reversed(det):
                        plot_one_box(xyxy, im0s, label=None, color=[0, 0, 255], line_thickness=2)

                save_path = os.path.join(self.save_dir, os.path.basename(img_path))
                if img_id == 0:
                    merge_list.append(im0s.copy())
                    num_list.append(len(det))
                    car_num += len(det)
                    plot_car_num(im0s, len(det), color=[255, 255, 255])
                    cv2.imwrite(save_path, im0s)
                elif img_id == 1:
                    merge_list.append(im0s.copy())
                    car_num += len(det)
                elif img_id == 2:
                    num_list.append(len(det))
                    plot_car_num(im0s, len(det), color=[255, 255, 255])
                    cv2.imwrite(save_path, im0s)

            num_list.append(car_num)
            merge_img = self.stitcher.blending(merge_list[0], merge_list[1])
            plot_car_num(merge_img, car_num, color=[255, 255, 255])
            save_path = os.path.join(self.save_dir, os.path.basename(path_list[0].replace('_1.jpg', '_merge.png')))
            cv2.imwrite(save_path, merge_img)

            if saveImg:
                save_path = os.path.join(self.save_dir_bk, prefix+os.path.basename(path_list[0].replace('_1.jpg', '_merge.png')))
                cv2.imwrite(save_path, merge_img)

                if saveOri:
                    merge0_img = self.stitcher.blending(merge0_list[0], merge0_list[1])
                    plot_car_num(merge0_img, car_num, color=[255, 255, 255])
                    save_path = os.path.join(self.save_dir_bk,
                                             prefix+os.path.basename(path_list[0].replace('_1.jpg', '_merge_ori.png')))
                    cv2.imwrite(save_path, merge0_img)

            if save_xlsx:
                xlsx_save(save_path, num_list)

    def clear(self):
        for fn in os.listdir(self.img_dir):
            os.remove(os.path.join(self.img_dir, fn))
        for fn in os.listdir(self.save_dir):
            os.remove(os.path.join(self.save_dir, fn))

    def send_email(self, event_time):
        os.makedirs(os.path.join('/mnt/ActionRecog/dataset/car_count/shangqi_email', event_time))
        for fn in os.listdir(self.save_dir):
            shutil.move(os.path.join(self.save_dir, fn),
                        os.path.join('/mnt/ActionRecog/dataset/car_count/shangqi_email', event_time, fn))
        print('Send success')
