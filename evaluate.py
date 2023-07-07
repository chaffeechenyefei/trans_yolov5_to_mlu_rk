import os
import cv2
import yaml
import torch
import numpy as np
import argparse
from models.experimental import attempt_load
from utils.datasets import LoadListImages
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.plots import plot_one_box

class Detect():
    def __init__(self, weights_fp, target_names, conf_thresh=0.4, iou_thresh=0.4, img_size=1280, device='cuda:0'):
        super(Detect, self).__init__()
        self.weights_fp = weights_fp
        self.img_size = img_size[0] if isinstance(img_size, list) else img_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = torch.device(device)
        self.model, self.names, self.stride = self.load_model(self.weights_fp, self.device, self.img_size)
        self.target_names = target_names if target_names else self.names

    def load_model(self, weights_fp, device, img_size):
        model = attempt_load(weights_fp, map_location=device)
        names = model.names
        stride = int(model.stride.max())
        img_size = check_img_size(img_size, s=stride)
        if device.type != 'cpu':
            self.half = True
            model = model.half()
        model.eval()
        img = torch.zeros((1, 3, img_size, img_size), dtype=torch.float32).to(device)
        img = img.half() if self.half else img
        yy = model(img)
        return model, names, stride

    @staticmethod
    def load_dataset(img_dir, img_size, stride):
        if img_dir.endswith('.txt'):
            img_list = np.loadtxt(img_dir, dtype=str)
        else:
            img_list = []
            for fn in os.listdir(img_dir):
                if os.path.splitext(fn)[-1] in ['.jpg', '.png']:
                    img_list.append(os.path.join(img_dir, fn))
        dataset = LoadListImages(img_list, img_size=img_size, stride=stride, print_path=False)
        return dataset

    def inference(self, dataset, bboximg_save_dir=None, txt_save_dir=None, file_save_fp=None):
        if file_save_fp:
            file = open(file_save_fp, 'w')
        for batch_i, (img_path, img, im0s, vid_cap) in enumerate(dataset):
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

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape)

            if bboximg_save_dir:
                for *xyxy, conf, cls in det.round():
                    color = [0, 0, 255]
                    plot_one_box(xyxy, im0s, label=None, color=color, line_thickness=2)
                bboximg_save_path = os.path.join(bboximg_save_dir, os.path.basename(img_path))
                cv2.imwrite(bboximg_save_path, im0s)

            if txt_save_dir:
                gn = np.array([im0s.shape[1], im0s.shape[0], im0s.shape[1], im0s.shape[0]])
                det = det.cpu().numpy()
                det[:, :4] = xyxy2xywh(det[:, :4])/gn
                det = det[:, [5, 0, 1, 2, 3]]

                txt_save_path = os.path.join(txt_save_dir, os.path.basename(img_path)).replace('jpg', 'txt')
                np.savetxt(txt_save_path, det)

            if file_save_fp:
                file.writelines(img_path + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='config path', default='./vehical_count/config_v5x6.yaml')
    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    img_size = 1280
    detect = Detect(weights_fp=os.path.join(config['project'], config['name'], 'weights/best.pt'),
                 target_names=[],
                 img_size=img_size,
                 conf_thresh=config['conf_thresh'],
                 iou_thresh=config['iou_thresh'],
                 device='cuda:0')
    dataset = detect.load_dataset(img_size=img_size, stride=detect.stride,
                                  img_dir='/data/dataset/yolo_dataset/UAV/UAV_VisDrone_bdd100k_val.txt',)
    detect.inference(
        dataset=dataset,
        bboximg_save_dir='/data/dataset/yolo_dataset/people_vehical_detection/wework_cocopeople_UAV_VisDrone_bdd100k',
        txt_save_dir='/data/dataset/yolo_dataset/people_vehical_detection/wework_cocopeople_UAV_VisDrone_bdd100k',
        file_save_fp='/data/dataset/yolo_dataset/people_vehical_detection/wework_cocopeople_UAV_VisDrone_bdd100k/val.txt')