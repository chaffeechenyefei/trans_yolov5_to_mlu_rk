import argparse
import os
import cv2
import torch
import random
from tqdm import tqdm
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.plots import plot_one_box, plot_car_num
from utils.torch_utils import select_device
# from safety_hat.classifier import Classifier
from fall.classifier import Classifier


def detect(save_img=True, save_num=False, save_txt=False, save_conf=False, plot_label=True, plot_conf=True,
           classify=False, target_names=None):
    if classify:
        classifier = Classifier()
    source, save_dir, weights, imgsz = opt.source, opt.save_dir, opt.weights, opt.img_size
    os.makedirs(save_dir, exist_ok=True)

    device = select_device(opt.device)
    half = device.type != 'cpu'
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    print('stride = ', stride)
    # imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    dataset = LoadImages(source, img_size=imgsz, stride=stride, print_path=True)
    vid_path, vid_writer = None, None
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = {n:[random.randint(0, 255) for _ in range(3)] for n in names}

    if device.type != 'cpu':
        if isinstance(imgsz, list):
            model(torch.zeros(1, 3, imgsz[0], imgsz[1]).to(device).type_as(next(model.parameters())))
        else:
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    for path, img, im0s, vid_cap in dataset:
        im0s_copy = im0s.copy()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        output = model(img)[0]
        pred = non_max_suppression(output, opt.conf_thres, opt.iou_thres)
        det = pred[0]
        if target_names:
            t_det = []
            for t in target_names:
                t_det.append(det[det[:, -1] == names.index(t)])
            det = torch.cat(t_det, dim=0)
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
            if save_txt:
                txt_path = os.path.join(save_dir, os.path.basename(path).split('.jpg')[0]) + \
                           ('' if dataset.mode == 'image' else f'_{dataset.frame}')
                f = open(txt_path + '.txt', 'w')
            for *xyxy, conf, cls in reversed(det):
                label = names[int(cls)]
                gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                if save_txt:  # Write to file
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

                if classify:
                    if int(cls) == 0:
                        label = classifier.inference\
                            (im0s_copy, (cls, *xywh))

                if save_img:
                    if plot_label:
                        if plot_conf:
                            label = label + f'{conf:.2f}'
                    color = colors[names[int(cls)]]
                    # color = [0, 0, 255]
                    if 'fall' in label:
                        color = [0, 0, 255]
                    if 'stand' in label:
                        color = [0, 255, 0]
                    plot_one_box(xyxy, im0s, color=color, label=label, line_thickness=2)
                    if save_num:
                        plot_car_num(im0s, len(det), color=[0, 0, 255], line_thickness=1, label=label)
            if save_txt:
                f.close()

        if save_img:
            save_path = os.path.join(save_dir, os.path.basename(path).split('.')[0])
            if dataset.mode == 'image':
                cv2.imwrite(save_path + '.jpg', im0s)
            else:  # 'video' or 'stream'
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0s.shape[1], im0s.shape[0]
                    save_path += '.mp4'
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0s)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        # default='/mnt/projects/ObjectDetection/weights/yolo_weights/MovingObject_v5s_conv/weights/best.pt')
                        default='/mnt/projects/ObjectDetection/weights/yolo_weights/SafetyHat_v5s_conv2/weights/best.pt')
                        # default='/mnt/projects/ObjectDetection/weights/yolo_weights/Fall_v5s_conv15/weight0.7409_0.5001.pt')
    parser.add_argument('--source', type=str,
                        default='/mnt/projects/ObjectDetection/dataset/SafetyHat/orig_dataset/')
                        # default='/mnt/projects/ObjectDetection/dataset/fall/shangti/')
    parser.add_argument('--save_dir', type=str,
                        default='/mnt/projects/ObjectDetection/dataset/SafetyHat/orig_results_5s_conv/')
                        # default='/mnt/projects/ObjectDetection/dataset/fall/shangti_results')
    parser.add_argument('--img_size', type=int, nargs='+', default=[416, 736])
    # parser.add_argument('--img_size', type=int, nargs='+', default=[1280,1280])
    parser.add_argument('--conf_thres', type=float, default=0.2)
    parser.add_argument('--iou_thres', type=float, default=0.5)
    parser.add_argument('--plot_label', action='store_true')
    parser.add_argument('--device', default='1')
    opt = parser.parse_args()
    detect()