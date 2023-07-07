import torch_mlu
import torch_mlu.core.mlu_model as ct
import torch_mlu.core.mlu_quantize as mlu_quantize
import argparse
import torch
import cv2
import numpy as np
import random,math
import os
torch.set_grad_enabled(False)
pj = os.path.join

from torch_mlu.core.utils import dump_utils

"""
yolov5
"""
from yolov5_api import yolov5_net, load_yolov5x6_model

def str2bool(v):
     return v.lower() in ("yes", "true", "t", "1")


def trans_mean_std_py2mlu(mean, std):
    """
    py: x = (x-mean)/std
    mlu: x = (x/255-m)/s = (x-255m)/(255s)
    :return: 
    """
    m = [ c/255 for c in mean]
    s = [ c/255 for c in std]
    return {
        "mean": m,
        "std": s
    }

def trans_mean_std_mlu2py(mean,std):
    """
    py: x = (x-m)/s
    mlu: x = (x/255-mean)/std = (x-255mean)/(255std)
    :return: 
    """
    s = [ 255*c for c in std ]
    m = [ 255*c for c in mean ]
    return {
        "mean":m,
        "std":s
    }

def fetch_cpu_data(x,use_half_input=False):
    if use_half_input:
        output = x.cpu().type(torch.FloatTensor)
    else:
        output = x.cpu()
    return output.detach().numpy()


def preprocess_yolov5(img_cv, dst_size, mean, std, mlu=False):
    h,w = img_cv.shape[:2]
    dstW,dstH = dst_size
    aspect_ratio = min([dstW/w,dstH/h])
    _h = min([ int(h*aspect_ratio), dstH])
    _w = min([ int(w*aspect_ratio), dstW])

    padh = dstH - _h
    padw = dstW - _w

    left_w_pad = int(padw/2)
    up_h_pad = int(padh/2)

    img_resized = cv2.resize(img_cv, (_w,_h) )

    img_dst = np.zeros([dstH,dstW,3],np.uint8)
    img_dst[up_h_pad:up_h_pad+_h,left_w_pad:left_w_pad+_w] = img_resized*1

    img_dst = cv2.cvtColor(img_dst, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    img_dst = torch.from_numpy(img_dst)
    img_dst = img_dst.float()  # uint8 to fp16/32

    if not mlu:
        mean = torch.FloatTensor(mean).reshape(-1,1,1)
        std = torch.FloatTensor(std).reshape(-1,1,1)
        img_dst -= mean
        img_dst /= std

    return img_dst, aspect_ratio

parser = argparse.ArgumentParser()
# Check
parser.add_argument('--check', action='store_true')
# Quant and Infer
parser.add_argument('--data',help='data path to the images used for quant')
parser.add_argument('--ext',default='.jpg')

parser.add_argument('--mlu', default=True, type=str2bool,
                    help='Use mlu to train model')
parser.add_argument('--jit', default=True, type=str2bool,
                    help='Use jit for inference net')
parser.add_argument('--quantization', default=False, type=str2bool,
                    help='Whether to quantize, set to True for quantization')

parser.add_argument("--batch_size", dest="batch_size", help="batch size for one inference.",
                    default=1, type=int)

# Advance
parser.add_argument("--quantized_mode", dest='quantized_mode', help=
"the data type, 0-float16 1-int8 2-int16, default 1.",
                    default=1, type=int)
parser.add_argument("--half_input", dest='half_input', help=
"the input data type, 0-float32, 1-float16/Half, default 1.",
                    default=1, type=int)

# Useless
parser.add_argument('--core_number', default=4, type=int,
                    help='Core number of mfus and offline model with simple compilation.')
parser.add_argument('--mcore', default='MLU270', type=str,
                    help="Set MLU Architecture")

args = parser.parse_args()
"""
Environment
"""
ct.set_core_number(args.core_number)
ct.set_core_version(args.mcore)
print("batch_size is {:d}, core number is {:d}".format(args.batch_size, args.core_number))
"""
ATT. Parameters
"""
model_path = './weights'
model_name = 'yolov5x6.pth'
model_online_fullname = pj(model_path, 'mlu_int8_{}'.format(model_name))
IMG_SIZE = [1024, 640]  # [w,h]
py_mean = [0, 0, 0]
py_std = [255, 255, 255]
mlu_mean_std = trans_mean_std_py2mlu(py_mean, py_std)
mlu_mean = mlu_mean_std['mean']
mlu_std = mlu_mean_std['std']
print("")

"""
Import data
"""
image_list = [pj(args.data, c) for c in os.listdir(args.data) if c.endswith(args.ext)]
K = min([len(image_list), args.batch_size])
image_list = image_list[:K]
print('sampled %d data' % len(image_list))
print(image_list[0])
input_img = [cv2.imread(c) for c in image_list]
# set mlu=False to always trigger normalization(mean,std)
data = [preprocess_yolov5(c, dst_size=IMG_SIZE, mean=py_mean, std=py_std, mlu=False) for c in input_img]
data = [c[0] for c in data]
print('len of data: %d' % len(data))
data = torch.stack(data, dim=0)
print('data shape =', data.shape)
data_mlu = 1*data
# if args.mlu:
#     if args.half_input:
#         data = data.type(torch.HalfTensor)
#     data = data.to(ct.mlu_device())

"""
Import pytorch model on cpu first
"""
print('==pytorch==')
use_device = 'cpu'
# yolov5_obj = yolov5_net(detect_sz=IMG_SIZE[0], torchLowVersion=True, fuse=False, singleOutput=True)
# model = yolov5_obj.model
model = load_yolov5x6_model(1024,preTrained=True)
print('==end==')
model = model.eval().float()


dump_utils.register_dump_hook(model)
pred = model(data)
pred_cpu = fetch_cpu_data(pred)
dump_utils.save_data('output/',"cpu")


# yolov5_obj = yolov5_net(detect_sz=IMG_SIZE[0], torchLowVersion=True, fuse=False, singleOutput=True)
# model = yolov5_obj.model
model_mlu = load_yolov5x6_model(1024, preTrained=False)
model_mlu = mlu_quantize.quantize_dynamic_mlu(model_mlu)
checkpoint = torch.load(model_online_fullname, map_location='cpu')
model_mlu.load_state_dict(checkpoint, strict=False)
model_mlu = model_mlu.eval().float().to(ct.mlu_device())
data_mlu = data_mlu.to(ct.mlu_device())

dump_utils.register_dump_hook(model_mlu)
pred = model_mlu(data_mlu)
pred_mlu = fetch_cpu_data(pred)
dump_utils.save_data('output/',"mlu")

dump_utils.diff_data("output/dump_cpu_data.pth", "output/dump_mlu_data.pth")

diff = np.sqrt(np.sum((pred_cpu[:,123,:5] - pred_mlu[:,123,:5])**2))
print(diff)
print(pred_cpu[:,123,:5],pred_mlu[:,123,:5])