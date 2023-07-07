import os

dataset = 'val'
data1_fp = f'/data/dataset/yolo_dataset/UAV/UAV_VisDrone_{dataset}.txt'
data2_fp = f'/data/dataset/yolo_dataset/bdd100k/{dataset}.txt'

save_fp = f'/data/dataset/yolo_dataset/UAV/UAV_VisDrone_bdd100k_{dataset}.txt'

with open(data1_fp, 'r') as f:
    lines1 = f.readlines()
print(len(lines1))

with open(data2_fp, 'r') as f:
    lines2 = f.readlines()
print(len(lines2))

lines1.extend(lines2)
print(len(lines1))

with open(save_fp, 'w') as f:
    f.writelines(lines1)