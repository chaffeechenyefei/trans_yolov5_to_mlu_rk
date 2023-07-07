import os
import shutil
from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2
import yaml
from utils.plots import plot_one_box

target2id = {'bicycle': 1, 'motorcycle': 3, 'dog': 16, 'cat': 15}
id2target = {value: key for key, value in target2id.items()}

if __name__ == '__main__':
    data_path = '/mnt/ActionRecog/yolov5/data/coco.yaml'
    save_dir = '/mnt/ActionRecog/dataset/coco_class'
    with open(data_path) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    for tn in target2id.keys():
        if not os.path.exists(os.path.join(save_dir, tn)):
            os.makedirs(os.path.join(save_dir, tn))
        if not os.path.exists(os.path.join(save_dir, 'labeled', tn)):
            os.makedirs(os.path.join(save_dir, 'labeled', tn))

    with open(data['val'], 'r') as f:
        path_list = f.read().strip().splitlines()
        path_list = [os.path.join('/mnt/ActionRecog/dataset/coco/coco', p) for p in path_list]

    for pic_path in tqdm(path_list, desc='Generating...'):
        if os.path.isfile(pic_path):
            pic_name = os.path.basename(pic_path)
            img = cv2.imread(pic_path)
            img_shape = img.shape # height, width, channel
            label_path = pic_path.replace('images', 'labels').replace('jpg', 'txt')
            if os.path.isfile(label_path):
                labels = np.loadtxt(label_path).astype(np.float32)
                if len(labels.shape) == 1:
                    labels = np.expand_dims(labels, 0)

                for target in target2id.keys():
                    target_labels = labels[labels[:, 0] == target2id[target]]
                    if len(target_labels) > 0:
                        save_path = os.path.join(save_dir, target, pic_name)
                        shutil.copy(pic_path, save_path)
                        label_save_path = save_path.replace('jpg', 'txt')
                        with open(label_save_path, 'w') as f:
                            for target_label in target_labels:
                                x, y, w, h = target_label[1:]
                                xmin = int((x-w/2) * img_shape[1])
                                ymin = int((y-h/2) * img_shape[0])
                                xmax = int(xmin + w * img_shape[1])
                                ymax = int(ymin + h * img_shape[0])
                                log = '{} {} {} {} {}'.format(target2id[target], xmin, ymin, xmax, ymax)
                                f.writelines(log + '\n')
                                plot_one_box([xmin, ymin, xmax, ymax], img, label=target, line_thickness=1)
                        rect_save_path = os.path.join(save_dir, 'labeled', target, pic_name)
                        cv2.imwrite(rect_save_path, img)