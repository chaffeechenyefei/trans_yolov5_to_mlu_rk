
import cv2
import torch
import numpy as np

from models.experimental import attempt_load, attempt_load_v2
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, \
    scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

from utils.activations import *
from models.yolo import Model,Model_yolov5x6
from models.common import Conv

"""
detect_sz = [w,h]
"""
def load_yolov5x6_model(detect_sz=[1024,1024], stride=[8., 16., 32., 64] ,cfg = './models/hub/yolov5x6.yaml',weights = './weights/yolov5x6.pth',preTrained=False):
    # cfg = './models/hub/yolov5x6.yaml'
    yolo_net = Model_yolov5x6(cfg=cfg, input_sz=detect_sz ,stride=stride)
    if preTrained:
        ckpt = torch.load(weights, map_location='cpu')  # load
        yolo_net.load_state_dict(ckpt, strict=False)
    yolo_net.eval()
    return yolo_net


def load_yolo_model(cfg='./models/yolov5s.yaml', weights='./weights/yolov5s.pt', preTrained=False):
    if preTrained:
        if weights.endswith('.pth'):
            yolo_net = Model(cfg=cfg)
            ckpt = torch.load(weights, map_location='cpu')
            yolo_net.load_state_dict(ckpt, strict=False)
        else:
            yolo_net_adv = attempt_load(weights, map_location='cpu', fuse=False)
            return yolo_net_adv
    else:
        yolo_net = Model(cfg=cfg)
        yolo_net.eval()
        return yolo_net

def initial_weights(model):
    for k, m in model.named_modules():
        if isinstance(m, Conv):
            m.conv.weight = nn.init.kaiming_normal_(m.conv.weight,mode='fan_out')
            m.bn.weight.data.fill_(1)
            m.bn.bias.data.zero_()

def update_models(model):
    for k, m in model.named_modules():
        # m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
    return model


#
# class yolov5_net(object):
#     COCO_CLASSES = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
#          'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
#          'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
#          'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
#          'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#          'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
#          'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
#          'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
#          'hair drier', 'toothbrush' ]
#
#     def __init__(self, threshold=0.5, iou_threshold=0.45, detect_sz=1024, class_name=['person'],
#                  use_device='cpu', torchLowVersion=True, fuse=True, singleOutput=False):
#         if torchLowVersion:
#             self.weights = './weights/yolov5x6{}'.format('.pth')
#         else:
#             self.weights = './weights/yolov5x6{}'.format('.pt')
#
#         self.torchLowVersion = torchLowVersion
#         self.singleOutput = singleOutput
#         self.img_size = detect_sz
#         self.conf_threshold = threshold
#         self.iou_threshold = iou_threshold
#         self.device = select_device('') if use_device is None else torch.device(use_device)
#         self.load(fuse)
#         dict_name_id = dict(zip(self.COCO_CLASSES, range(len(self.COCO_CLASSES))))
#         self.classes = [ dict_name_id[phs] for phs in class_name ]
#         class_det_set = [ phs for phs in class_name ]
#         print('{} will be detected'.format(class_det_set))
#
#     def load(self,fuse):
#         if self.torchLowVersion:
#             self.model = attempt_load_v2(self.weights, cfg='models/hub/yolov5x6.yaml' ,
#                                          map_location=self.device, singleOutput=self.singleOutput)  # load FP32 model
#         else:
#             self.model = attempt_load(self.weights, map_location=self.device, fuse=fuse)  # load FP32 model
#         self.stride = int(self.model.stride.max())  # model stride
#         self.img_size = check_img_size(self.img_size, s=self.stride)  # check img_size
#         if self.device.type != 'cpu':
#             self.model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(next(self.model.parameters())))  # run once
#         self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
#         self.model.eval()
#
#     @torch.no_grad()
#     def inference_v2(self, img_cv2, threshold=None, iou_threshold=None):
#         if threshold is not None:
#             self.threshold = threshold
#         if iou_threshold is not None:
#             self.iou_threshold = iou_threshold
#         # Padded resize
#         img_cv = img_cv2.copy()
#         img = letterbox(img_cv, self.img_size, stride=self.stride)[0]
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2,0,1)
#         img = torch.from_numpy(img).to(self.device)
#         img = img.float()  # uint8 to fp16/32
#         img /= 255.0  # 0 - 255 to 0.0 - 1.0
#         if len(img.shape) == 3:
#             img = img.unsqueeze(0)
#         # print(img.shape)
#         # Inference
#         # t1 = time_synchronized()
#         pred = self.model(img, augment=False)[0]
#         # print(pred.shape) = [Batch, w*h*,85] 85=80(cls)+1(obj)+4(xyxy)
#
#         # Apply NMS
#         pred = non_max_suppression(pred, self.conf_threshold, self.iou_threshold, classes=self.classes, agnostic=False)
#         # t2 = time_synchronized()
#
#         det = pred[0] #[*,85] torch
#         xyxys = []
#         if len(det):
#             # Rescale boxes from img_size to im0 size
#             det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_cv2.shape).round()
#             # Write results
#             for *xyxy, conf, cls in reversed(det):
#                 label = '{} {:.2f}'.format(self.names[int(cls)],conf)
#                 plot_one_box(xyxy, img_cv, label=None, color=(0,0,255), line_thickness=3)
#                 xyxy = [ int(c.data.cpu().item()) for c in xyxy ]
#                 xyxys.append(xyxy)
#
#         return xyxys, img_cv
#
#
#     @torch.no_grad()
#     def inference(self, img_cv2, threshold=None, iou_threshold=None):
#         if threshold is not None:
#             self.threshold = threshold
#         if iou_threshold is not None:
#             self.iou_threshold = iou_threshold
#         # Padded resize
#         img = letterbox(img_cv2, self.img_size, stride=self.stride)[0]
#         # Convert
#         img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
#         img = np.ascontiguousarray(img)
#         # to torch
#         img = torch.from_numpy(img).to(self.device)
#         img = img.float()  # uint8 to fp16/32
#         img /= 255.0  # 0 - 255 to 0.0 - 1.0
#         if img.ndimension() == 3:
#             img = img.unsqueeze(0)
#
#         # Inference
#         t1 = time_synchronized()
#         pred = self.model(img, augment=False)[0]
#
#         # Apply NMS
#         pred = non_max_suppression(pred, self.conf_threshold, self.iou_threshold, classes=self.classes, agnostic=False)
#         t2 = time_synchronized()
#
#         det = pred[0]
#         xyxys = []
#         if len(det):
#             # Rescale boxes from img_size to im0 size
#             det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_cv2.shape).round()
#             # Write results
#             for *xyxy, conf, cls in reversed(det):
#                 label = '{} {:.2f}'.format(self.names[int(cls)],conf)
#                 plot_one_box(xyxy, img_cv2, label=None, color=(0,0,255), line_thickness=3)
#                 xyxy = [ int(c.data.cpu().item()) for c in xyxy ]
#                 xyxys.append(xyxy)
#
#         return xyxys, img_cv2, det.cpu()


if __name__ == '__main__':
    print("nothing")
    # import io,os,shutil
    # import decord
    # pj = os.path.join
    #
    # net = yolov5_net(detect_sz=640)
    #
    # vid_path = '/project/data/education_video/classroom'
    # vidlist = [c for c in os.listdir(vid_path) if c.endswith('.mp4')]
    # for vidfile in vidlist:
    #     print(vidfile)
    #
    #     filename = os.path.basename(pj(vid_path,vidfile))
    #     filename = filename.replace('.', '_')
    #     savedir = pj(vid_path,filename)
    #     if os.path.exists(savedir):
    #         shutil.rmtree(savedir, ignore_errors=True)
    #     os.makedirs(savedir)
    #
    #     container = decord.VideoReader(pj(vid_path,vidfile), num_threads=1)
    #     total_frames = len(container)
    #     print('total {:d} frames inside'.format(total_frames))
    #
    #     step = 10
    #     idx_list = range(0, total_frames - 1, step)
    #     for k, idx in enumerate(idx_list):
    #         if idx == 0:
    #             continue
    #         print('{:.2f}%...'.format(100 * k / len(idx_list)), end='\r')
    #         try:
    #             image = container[idx].asnumpy()
    #         except:
    #             print('Error caught in decord')
    #             break
    #         image_rgb = image.copy()
    #         image_bgr = cv2.cvtColor(image_rgb,cv2.COLOR_RGB2BGR)
    #         # image_rgb = cv2.transpose(image_rgb)
    #         # print(image_rgb.shape)
    #
    #         xyxys, img_cv2 = net.inference_v2(image_bgr)
    #         if len(xyxys) > 0:
    #             # print(xyxys) if idx < 10 else None
    #             cv2.imwrite(pj(savedir,'{:04d}.jpg'.format(idx)), img_cv2)
    #
    #     del container
