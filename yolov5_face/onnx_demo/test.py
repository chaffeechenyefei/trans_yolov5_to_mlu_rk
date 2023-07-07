import os, sys
pj = os.path.join
curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curPath)
from onnx_demo.onnx_utils import *

import cv2
import numpy as np



model_path = '/project/yolov5/weights/fire_detection/yolov5s.onnx'
model = ONNXModel(model_path)

img_url = '/project/data/Fire/dunnings-2018/images-224x224/test/fire/WaterMistFireDemonstration6942.png'
img = cv2.imread(img_url)

data = cv2_to_1chw(img,[640,640],[0,0,0],[255,255,255], False)

aspect_ratio = data['aspect_ratio']
input_data = data['data']

ret = model.forward(input_data)

print(ret[0].shape) #1, 25200, 85





