import os
import glob
import numpy as np
import cv2
import shutil
import threading

classes_id = {
    'car': [33, 161, 38, 166, 39, 167],
    'motorcycle': [34, 162],
    'bicycle': [35, 163, 40, 168],
}
# detect_targets = {'car': 2, 'bicycle': 1, 'motorcycle': 3}
# id2target = {id: t for t, id in detect_targets.items()}
dataset_dir = '/mnt/ActionRecog/dataset/apollo/'
# pic_fp_list = glob.glob(os.path.join(dataset_dir, 'images', '*/*/*/*/*.jpg'))
# pic_save_dir = '/mnt/ActionRecog/dataset/apollo_class'
# # shutil.rmtree(pic_save_dir)
# os.system(f'rm -r {pic_save_dir}')
# os.makedirs(os.path.join(pic_save_dir, 'car'), exist_ok=True)
# os.makedirs(os.path.join(pic_save_dir, 'bicycle'), exist_ok=True)
# os.makedirs(os.path.join(pic_save_dir, 'motorcycle'), exist_ok=True)

def extract_targets(thread_idx, pic_fp_list):
    txt_save_path = os.path.join(dataset_dir, 'sub_txt', f'test_sub{thread_idx}.txt')
    with open(txt_save_path, 'w') as t:
        for pic_idx, pic_fp in enumerate(pic_fp_list):
            label_fp = pic_fp.replace('.jpg', '_bin.png').replace('ColorImage', 'Label')
            try:
                label = cv2.imread(label_fp)[:, :, 0]  # height, width
                img = cv2.imread(pic_fp)
            except:
                continue

            index_dict = {}
            boxes, box_label = [], []
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
                        x, y = min_x / img.shape[1] + w / 2, min_y / img.shape[0] + h / 2
                        if max_y - min_y > 50 and max_x - min_x > 50:
                            if cv2.cvtColor(img[min_y: max_y, min_x: max_x, :], cv2.COLOR_BGR2GRAY).mean() > 50:
                                boxes.append([target_id, min_x, min_y, max_x, max_y])
                                box_label.append([target_id, x, y, w, h])

            for target_idx, box in enumerate(boxes):
                target_img = img[box[2]:box[4], box[1]:box[3], :]
                target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2BGRA)

                target_label = np.zeros_like(label)
                target_label[index_dict[id2target[box[0]]]] = 1
                label_img = target_label[box[2]:box[4], box[1]:box[3]]
                label_img = label_img * 255

                target_img[:, :, -1] = label_img

                img_save_fp = os.path.join(pic_save_dir, id2target[box[0]], f'{thread_idx}_{pic_idx}_{target_idx}.png')
                print(img_save_fp)
                cv2.imwrite(img_save_fp, target_img)

            if len(box_label) > 0:
                boxes = np.array(box_label, np.float32).reshape(-1, 5)
                box_save_fp = pic_fp.replace('images', 'labels').replace('.jpg', '.txt')
                os.makedirs(os.path.dirname(box_save_fp), exist_ok=True)
                np.savetxt(box_save_fp, boxes)
                t.writelines(pic_fp + '\n')

if __name__ == '__main__':
    thread_num = 100
    # div_num = int(len(pic_fp_list)/thread_num)
    # for id in range(thread_num):
    #     if id == thread_num-1:
    #         sub_list = pic_fp_list[id * div_num:]
    #     else:
    #         sub_list = pic_fp_list[id*div_num: (id+1)*div_num]
    #     threading.Thread(target=extract_targets, args=(id, sub_list)).start()

    total_lines = []
    for thread_idx in range(thread_num):
        sub_txt_save_path = os.path.join(dataset_dir, 'sub_txt', f'test_sub{thread_idx}.txt')
        with open(sub_txt_save_path, 'r') as f:
            lines = f.readlines()
        total_lines.extend(lines)

    import random
    random.shuffle(total_lines)
    print(len(total_lines))  # 55194

    txt_save_path = os.path.join(dataset_dir, f'total.txt')
    with open(txt_save_path, 'w') as f:
        for line in total_lines:
            f.writelines(line)

    train_save_path = os.path.join(dataset_dir, f'train.txt')
    with open(train_save_path, 'w') as f:
        for line in total_lines[:44155]:
            f.writelines(line)

    test_save_path = os.path.join(dataset_dir, f'test.txt')
    with open(test_save_path, 'w') as f:
        for line in total_lines[44155:]:
            f.writelines(line)