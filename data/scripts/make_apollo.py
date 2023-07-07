import os
import glob
from tqdm import tqdm
import numpy as np
import cv2
import threading

classes_id = {
    'car': [33, 161, 38, 166, 39, 167],
    'bicycle': [34, 162, 40, 168],
    'motorcycle': [35, 163, 37, 165],
}
detect_targets = {'car': 2, 'bicycle': 1, 'motorcycle': 3}

def make_label(n, pic_fp_list):
    txt_save_path = os.path.join(dataset_dir, f'test_sub{n}.txt')
    with open(txt_save_path, 'w') as t:
        # for pic_fp in tqdm(pic_fp_list, desc='Generating...'):
        for pic_fp in pic_fp_list:
            label_fp = pic_fp.replace('.jpg', '_bin.png').replace('ColorImage', 'Label')
            label = cv2.imread(label_fp)[:, :, 0]  # height, width
            img = cv2.imread(pic_fp)

            index_dict = {}
            boxes = []
            for target, target_id in detect_targets.items():
                index_dict[target] = label == classes_id[target][0]
                if len(classes_id[target]) > 1:
                    for class_id in classes_id[target][1:]:
                        index_dict[target] = (index_dict[target]) | (label == class_id)

                # find each target label
                target_label = np.zeros_like(label)
                target_label[index_dict[target]] = 1

                # get target box
                contours, hierarchy = cv2.findContours(target_label, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if contour.shape[0] >= 4:
                        contour = np.squeeze(contour)
                        min_x, max_x = min(contour[:, 0]), max(contour[:, 0])
                        min_y, max_y = min(contour[:, 1]), max(contour[:, 1])
                        w, h = (max_x - min_x) / img.shape[1], (max_y - min_y) / img.shape[0]
                        x, y = min_x / img.shape[1] + w/2, min_y / img.shape[0] + h/2
                        cv2.rectangle(img, (min_x, min_y), (max_x, max_y), color=(0, 255, 0), thickness=5)
                        boxes.append([target_id, x, y, w, h])
            if len(boxes) > 0:
                boxes = np.array(boxes, np.float32).reshape(-1, 5)
                box_save_fp = pic_fp.replace('images', 'labels').replace('.jpg', '.txt')
                os.makedirs(os.path.dirname(box_save_fp), exist_ok=True)
                np.savetxt(box_save_fp, boxes)
                t.writelines(pic_fp + '\n')

if __name__ == '__main__':
    dataset_dir = '/mnt/ActionRecog/dataset/apollo/'
    # pic_list = glob.glob(os.path.join(dataset_dir, 'images', '*/*/*/*/*.jpg'))
    # print(len(pic_list))
    #
    thread_num = 10
    # div_num = int(len(pic_list)/thread_num)
    # for id in range(thread_num):
    #     if id == thread_num-1:
    #         sub_list = pic_list[id * div_num:]
    #     else:
    #         sub_list = pic_list[id*div_num: (id+1)*div_num]
    #     threading.Thread(target=make_label, args=(id, sub_list)).start()

    boxes = []

    lines = []
    for id in range(thread_num):
        sub_save_path = os.path.join(dataset_dir, f'test_sub{id}.txt')
        with open(sub_save_path, 'r') as f:
            sub_lines = f.read().strip().splitlines()
            lines.extend(sub_lines)
            print(len(lines))
    #     sub_label = np.loadtxt(sub_save_path)
    #     boxes.append(np.array(sub_label))
    # boxes = np.concatenate(boxes, axis=0)
    txt_save_path = os.path.join(dataset_dir, f'test_split.txt')
    lines = lines[::10]
    print(len(lines))
    with open(txt_save_path, 'w') as t:
        for line in lines:
            t.writelines(line + '\n')
    # np.savetxt(txt_save_path, lines)