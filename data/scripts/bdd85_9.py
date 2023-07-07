import os
import tqdm
import numpy as np
from glob import glob


names85 = ['people', 'bicycle', 'car', 'motor', 'airplane', 'bus', 'train', 'truck']
names9 = [ 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor' ]

dataset = 'train'
labels85_dir = f'/data/dataset/yolo_dataset/bdd100k/labels_85/100k/{dataset}'
labels9_dir = f'/data/dataset/yolo_dataset/bdd100k/labels/100k/{dataset}'
labels_list = os.listdir(labels85_dir)
for each_fn in tqdm.tqdm(labels_list):
    each_fp = os.path.join(labels85_dir, each_fn)
    labels = np.loadtxt(each_fp)
    if len(labels.shape) == 1:
        labels = np.expand_dims(labels, axis=0)
    # print(labels)

    labels = labels[labels[:, 0] != 6, :]
    labels = labels[labels[:, 0] != 4, :]
    labels_cp = labels.copy()
    for id in [0,1,2,3,5,7]:
        labels[labels_cp[:, 0] == id, 0] = names9.index(names85[id])

    np.savetxt(os.path.join(labels9_dir, each_fn), labels)
    # print(labels)
    # print()