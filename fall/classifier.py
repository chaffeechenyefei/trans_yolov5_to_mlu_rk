import os
import cv2
import torch
import numpy as np
from models.resnet import resnet34
class Classifier():
    def __init__(self, num_classes=2,
                 weights_fp='/mnt/projects/PracticeProjects/FallClassifier/weights/bestmodel_resnet34.pth'):
        self.resnet = resnet34(pretrained=False, num_classes=num_classes)
        save_dict = torch.load(weights_fp, map_location=None)
        self.resnet.load_state_dict(save_dict)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resnet.to(self.device)
        self.resnet.eval()
        self.id2cls = {0: 'fall', 1: 'stand'}

    def data_normalize(self, x, dsize):
        h, w, c = x.shape
        max_len = max(x.shape[:2])
        img_pad = np.zeros((max_len, max_len, 3), np.float)
        paste_pos = [int((max_len - w) / 2), int((max_len - h) / 2)]
        img_pad[paste_pos[1]:paste_pos[1] + h, paste_pos[0]:paste_pos[0] + w] = x

        x = cv2.resize(img_pad, dsize=(dsize, dsize))
        mean, std = 127.5, 127.5
        x = (x - mean) / std
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, axis=0)
        x = torch.tensor(x, dtype=torch.float32)
        x = x.to(self.device)
        return x

    def detect_head(self, img):
        H, W, C = img.shape
        head_len = int(0.25 * max(H, W))
        if H >= W:
            head_left = max(0, int((W-head_len) / 2))
            head_right = min(W, head_left + head_len)
            head_img = img[:head_len, head_left:head_right, :]
        else:
            head_top = max(0, int((H - head_len) / 2))
            head_bottom = min(H, head_top + head_len)
            head_img = img[head_top:head_bottom, :head_len, :]
        return head_img

    def img_preprocess(self, img, label):
        bbox = self.label2bbox(img, label)
        img_bbox = img[bbox[2]: bbox[4], bbox[1]: bbox[3], :]
        img_normalized = self.data_normalize(img_bbox, dsize=112)
        return img_normalized

    def label2bbox(self, img, label):
        H, W, C = img.shape
        bbox = np.zeros_like(label)
        bbox[0] = label[0]
        bbox[1] = np.clip((label[1] - label[3] / 2) * W, 0, W)
        bbox[2] = np.clip((label[2] - label[4] / 2) * H, 0, H)
        bbox[3] = np.clip((label[1] + label[3] / 2) * W, 0, W)
        bbox[4] = np.clip((label[2] + label[4] / 2) * H, 0, H)
        bbox = bbox.astype(np.int)
        return bbox

    def inference(self, img, label):
        bbox_img = self.img_preprocess(img, label)
        output = self.resnet(bbox_img)
        pred = np.argmax(output.cpu().data.numpy(), axis=1)[0]
        cls = self.id2cls[pred]
        return cls

if __name__ == '__main__':
    classifier = Classifier()
    img = cv2.imread('/data/dataset/PracticeClassifier/HatClassifier/test/hat/000054_0.jpg')
    label = [0, 0.5, 0.5, 1, 1]
    cls = classifier.inference(img, label)
    print(cls)