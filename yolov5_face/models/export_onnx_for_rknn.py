"""Exports a YOLOv5 *.pt model to ONNX and TorchScript formats

Usage:
    $ export PYTHONPATH="$PWD" && python models/export.py --weights ./weights/yolov5s.pt --img 640 --batch 1
"""

import argparse
import sys
import time

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
sys.path.append('../')  # to run '$ python *.py' files in subdirectories

import torch
import torch.nn as nn
import numpy as np
import os
pj = os.path.join

import models
# from models.experimental import attempt_load

from yolov5_api import load_yolov5x6_model

from utils.activations import Hardswish, SiLU
from utils.general import set_logging, check_img_size
from utils.torch_utils import select_device
from utils.onnx_infer import ONNXModel
from config import model_context, MODELROOT

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_key',type=str, help='{}'.format(list(model_context.keys())))
    parser.add_argument('--mode', type=int, default=0)
    # Useless cmd -----------------
    parser.add_argument('--dynamic', action='store_true', help='dynamic ONNX axes')
    parser.add_argument('--grid', action='store_true', help='export Detect() layer grid')
    opt = parser.parse_args()
    print(opt)

    model_key = opt.model_key
    print('model_key = {}'.format(model_key))
    if not model_key in model_context.keys():
        print('model_key:{} not founded...'.format(model_key))
        exit(-1)
    pt_weight_full_name = pj(MODELROOT, model_context[model_key]['weights'])
    onnx_weight_full_name = pt_weight_full_name.replace('.pt', '_mode{:d}_rknn.onnx'.format(opt.mode))

    # Load PyTorch model -------------------------
    device = select_device('cpu')
    w = model_context[model_key]['w'] if 'w' in model_context[model_key].keys() else 736
    h = model_context[model_key]['h'] if 'h' in model_context[model_key].keys() else 416

    # model = attempt_load(pt_weight_full_name, map_location=device)  # load FP32 model
    IMG_SIZE = [w , h]
    model = load_yolov5x6_model(detect_sz=IMG_SIZE, stride=model_context[model_key]['stride'],
                                cfg=model_context[model_key]['cfg'],
                                weights=model_context[model_key]['weights'].replace('.pt', '.pth'),
                                preTrained=True,
                                yolov5face=True)
    model = model.eval().float()

    labels = model.names



    # Checks
    gs = int(max(model.stride))  # grid size (max stride)
    h,w = [check_img_size(x, gs) for x in [h,w]]  # verify img_size are gs-multiples
    print('Input Tensor: [h:{:d}, w:{:d}]'.format(h,w))
    # Input
    img = torch.rand(1, 3, h , w , dtype=torch.float32).to(device)  # image size(1,3,320,192) iDetection
    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        # elif isinstance(m, models.yolo.Detect):
        #     m.forward = m.forward_export  # assign forward (optional)
    model.model[-1].export = True  # set Detect() layer grid export
    model.model[-1].export_mode = opt.mode
    if opt.mode == 3:
        ### special part for mode 3, a proxy conv for anchor_grid
        model.model[-1].conv1x1 = nn.Conv2d(1,1,1,1,0, bias=False)
        model.model[-1].conv1x1.weight = nn.Parameter(torch.ones(1, 1, 1, 1).float())
        model.model[-1].conv1x1.eval()

    y = model(img)  # dry run
    print([i.shape for i in y])
    # exit(0)

    # ONNX export -------------------------
    try:
        # import onnx
        # print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
          # filename
        if opt.dynamic:
            dynamic_axes = {'images': {0: 'batch', 2: 'height', 3: 'width'},  # size(1,3,640,640)
                                        'output0': {0: 'batch', 2: 'y', 3: 'x'},
                                        'output1': {0: 'batch', 2: 'y', 3: 'x'},
                                        'output2': {0: 'batch', 2: 'y', 3: 'x'}}
        else:
            dynamic_axes = None

        torch.onnx.export(model, img, onnx_weight_full_name, verbose=False, opset_version=11, input_names=['images'],
                          # output_names=['classes', 'boxes'] if y is None else ['output0', 'output1', 'output2'],
                          dynamic_axes=dynamic_axes)
        print('checking')
        # Checks
        # onnx_model = onnx.load(f)  # load onnx model
        # onnx.checker.check_model(onnx_model)  # check onnx model
        # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('ONNX export success, saved as %s' % onnx_weight_full_name)

        print('--> onnx_net loading')
        onnx_net = ONNXModel(onnx_weight_full_name)
        print('<-- onnx_net loading')
        inputs = img.cpu().data.numpy().astype(np.float32)
        print('--> onnx_net forward')
        outputs = onnx_net.forward(inputs)
        print('<-- onnx_net forward')
        if isinstance(outputs,list):
            for o in outputs:
                print(o.shape)
                print(o.reshape(-1)[:10])
        else:
            print(outputs.shape)

    except Exception as e:
        print('ONNX export failure: %s' % e)

    print('==SUCCESS==')

    # CoreML export
    """
    try:
        import coremltools as ct

        print('\nStarting CoreML export with coremltools %s...' % ct.__version__)
        # convert model from torchscript and apply pixel scaling as per detect.py
        model = ct.convert(ts, inputs=[ct.ImageType(name='image', shape=img.shape, scale=1 / 255.0, bias=[0, 0, 0])])
        f = opt.weights.replace('.pt', '.mlmodel')  # filename
        model.save(f)
        print('CoreML export success, saved as %s' % f)
    except Exception as e:
        print('CoreML export failure: %s' % e)

    # Finish
    print('\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron.' % (time.time() - t))
    """
