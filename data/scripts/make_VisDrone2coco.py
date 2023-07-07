import os
import glob
import cv2
import tqdm
import argparse
import matplotlib.pyplot as plt
import numpy as np
def plot_bbox(img, bboxes):
    H, W, C = img.shape
    color=[0, 0, 255]
    tl = 2
    for bbox in bboxes:
        c1 = int((bbox[1]-bbox[3]/2) * W), int((bbox[2]-bbox[4]/2) * H)
        c2 = int((bbox[1]+bbox[3]/2) * W), int((bbox[2]+bbox[4]/2) * H)
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    return img

target2id = {'pedestrian': 0, 'people': 0, 'bicycle': 1, 'car': 2, 'van': 3, 'truck': 4, 'tricycle':5,
             'awning-tricycle': 6, 'bus': 7, 'motor': 8}

UAV_id2target = {1: 'car', 2: 'truck', 3: 'bus'}
VisDrone_id2target = {0: 'ignored regions', 1: 'pedestrian', 2: 'people', 3: 'bicycle', 4: 'car', 5: 'van',
                      6: 'truck', 7: 'tricycle', 8: 'awning-tricycle', 9: 'bus', 10: 'motor', 11: 'others'}

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='train')
opt = parser.parse_args()
task = opt.task
data_dir = '/data/dataset/yolo_dataset/VisDrone_DET'
pic_list = glob.glob(os.path.join(data_dir, f'VisDrone2019-DET-{task}/images/*.jpg'))
print(len(pic_list))

images_save_dir = os.path.join(data_dir, 'images', task)
os.makedirs(images_save_dir, exist_ok=True)

labels_save_dir = os.path.join(data_dir, 'labels', task)
os.makedirs(labels_save_dir, exist_ok=True)

bboximg_save_dir = os.path.join(data_dir, 'bboximg', task)
os.makedirs(bboximg_save_dir, exist_ok=True)

fp_file = open(os.path.join(data_dir, f'{task}.txt'), 'w')
for pic_fp in tqdm.tqdm(pic_list, desc='Generating...'):
    pic_name = pic_fp.split('/')[-1]

    img = cv2.imread(pic_fp)

    gt_fp = os.path.join(data_dir, f'VisDrone2019-DET-{task}/annotations/{pic_name.split(".")[0]}.txt')
    try:
        gt = np.loadtxt(gt_fp, delimiter=',')
    except:
        gt = []
        with open(gt_fp, 'r') as gt_file:
            lines = gt_file.readlines()
            for line in lines:
                line = line.strip()
                if line[-1] == ',':
                    line = line[:-1]
                tmp = [float(l) for l in line.split(',') ]
                gt.append(tmp)
        gt = np.array(gt, dtype=np.float32)
    gt = gt.reshape(-1, 8)
    try:
        gt[:, 1:5] = gt[:, 0:4]
        gt[:, 0] = gt[:, 5]
        boxes = gt[:, :5]
    except:
        print(gt)
        break

    ignore = boxes[boxes[:, 0] == 0, :]
    ignore = ignore.astype(np.int32)

    boxes = boxes[boxes[:, 0] != 0, :]
    boxes = boxes[boxes[:, 0] != 11, :]
    boxes_copy = boxes.copy()

    for id, target in VisDrone_id2target.items():
        if target not in ['ignored regions', 'others']:
            boxes[:, 0][boxes_copy[:, 0] == id] = target2id[target]

    boxes[:, 1] = (boxes[:, 1] + boxes[:, 3] / 2) / img.shape[1]
    boxes[:, 2] = (boxes[:, 2] + boxes[:, 4] / 2) / img.shape[0]
    boxes[:, 3] = boxes[:, 3] / img.shape[1]
    boxes[:, 4] = boxes[:, 4] / img.shape[1]

    ignored_img = img.copy()
    for each_region in ignore:
        ignored_img[each_region[2]: each_region[2] + each_region[4], each_region[1]: each_region[1] + each_region[3],
        :] = 0

    cv2.imwrite(os.path.join(images_save_dir, f'{pic_name}'), ignored_img)

    bbox_img = plot_bbox(ignored_img, boxes.tolist())
    cv2.imwrite(os.path.join(bboximg_save_dir, f'{pic_name}'), bbox_img)

    np.savetxt(os.path.join(labels_save_dir, f'{pic_name}'.replace('jpg', 'txt')), boxes)

    fp_file.writelines(os.path.join(images_save_dir, f'{pic_name}') + '\n')
