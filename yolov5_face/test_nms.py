import numpy as np
import torch
import cv2
from utils.general import non_max_suppression, \
    scale_coords
from utils.plots import plot_one_box
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--w', type=int, default=1024)
    parser.add_argument('--h', type=int ,default=640)
    args = parser.parse_args()

    COCO_CLASSES = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]

    # img_cv2 = cv2.imread('data/images/zidane.jpg')
    img_cv2 = cv2.imread('./tmp_img/test.jpg')
    img_cva = img_cv2.copy()
    img_cvb = img_cv2.copy()
    a = np.load('cpu_pred.npy')
    b = np.load('mlu_pred.npy')

    IMG_SIZE = [args.w,args.h]

    conf_threshold = 0.4
    iou_threshold = 0.4

    class_name = COCO_CLASSES#['person']
    dict_name_id = dict(zip(COCO_CLASSES, range(len(COCO_CLASSES))))
    classes = [ dict_name_id[phs] for phs in class_name ]
    class_det_set = [ phs for phs in class_name ]
    print('{} will be detected'.format(class_det_set))

    pred = torch.FloatTensor(a)
    pred = non_max_suppression(pred, conf_threshold, iou_threshold, classes=classes, agnostic=False)

    det = pred[0]  # [*,85] torch
    xyxys = []
    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(IMG_SIZE[::-1], det[:, :4], img_cv2.shape).round()
        # Write results
        for *xyxy, conf, cls in reversed(det):
            plot_one_box(xyxy, img_cva, label=None, color=(0, 0, 255), line_thickness=3)
            xyxy = [int(c.data.cpu().item()) for c in xyxy]
            xyxys.append(xyxy)

    cv2.imwrite('test_a.jpg', img_cva)

    predb = torch.FloatTensor(b)
    predb = non_max_suppression(predb, conf_threshold, iou_threshold, classes=classes, agnostic=False)
    print(predb)
    det = predb[0]  # [*,85] torch
    xyxys = []
    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(IMG_SIZE[::-1], det[:, :4], img_cv2.shape).round()
        # Write results
        for *xyxy, conf, cls in reversed(det):
            plot_one_box(xyxy, img_cvb, label=None, color=(0, 0, 255), line_thickness=3)
            xyxy = [int(c.data.cpu().item()) for c in xyxy]
            xyxys.append(xyxy)

    cv2.imwrite('test_b.jpg', img_cvb)