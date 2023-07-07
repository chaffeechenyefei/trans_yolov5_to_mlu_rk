"""
mlu head
"""
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
"""
yolov5
"""
from yolov5_api import yolov5_net
from yolov5_api import load_yolov5x6_model
from trans_weights_to_old_pth import model_context

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

    # print(dstH,dstW,h,w, _h,_w)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_key', default='yolov5x6')
    parser.add_argument('--w',default=736, type=int)
    parser.add_argument('--h',default=416, type=int)
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

    if args.check:
        print('==Checking==')
        cpu_res = np.load('cpu_pred.npy')#[:,:,:5]
        mlu_res = np.load('mlu_pred.npy')#[:,:,:5]
        quant_res = np.load('cpu_quant_pred.npy')#[:mlu_res.shape[0],:,:5]

        print('cpu:', cpu_res.shape)
        print('mlu', mlu_res.shape)
        print('quant:', quant_res.shape)

        diff = np.sqrt(np.sum((cpu_res - mlu_res)**2))/cpu_res.shape[0]
        print('Sqrt Diff cpu vs mlu: {:.3f}'.format(diff))

        diff = np.sqrt(np.sum((quant_res - mlu_res)**2))/quant_res.shape[0]
        print('Sqrt Diff quant vs mlu: {:.3f}'.format(diff))

        diff = np.sqrt(np.sum((quant_res - cpu_res)**2))/cpu_res.shape[0]
        print('Sqrt Diff quant vs cpu: {:.3f}'.format(diff))

        print('==Done==')
        exit(0)

    # if args.check:
    #     print('==Checking==')
    #     cpu_res = np.load('cpu_pred.npy')
    #     quant_res = np.load('cpu_quant_pred.npy')

    #     print('cpu:', cpu_res.shape)
    #     print('quant:', quant_res.shape)

    #     diff = np.sqrt(np.sum((quant_res - cpu_res)**2))/cpu_res.shape[0]
    #     print('Sqrt Diff quant vs cpu: {:.3f}'.format(diff))

    #     print('==Done==')
    #     exit(0)

    """
    Environment
    """
    ct.set_core_number(args.core_number)
    ct.set_core_version(args.mcore)
    print("batch_size is {:d}, core number is {:d}".format(args.batch_size, args.core_number))
    dtype = 'int8'
    if args.quantized_mode == 0:
        dtype = 'float16'
    elif args.quantized_mode == 1:
        dtype = 'int8'
    elif args.quantized_mode == 2:
        dtype = 'int16'
    else:
        pass
    print('using dtype = {}'.format(dtype))
    """
    ATT. Parameters
    """
    model_path = './weights'
    model_key = args.model_key
    model_name = '{}.pth'.format(model_key)
    print('@@ Model: {}'.format(model_name))
    model_online_fullname = pj(model_path, 'mlu_{}_{}_{:d}x{:d}.pth'.format(dtype,model_key,args.w,args.h))
    IMG_SIZE = [args.w, args.h]  # [w,h]
    py_mean = [0, 0, 0]
    py_std = [255, 255, 255]
    mlu_mean_std = trans_mean_std_py2mlu(py_mean,py_std)
    mlu_mean = mlu_mean_std['mean']
    mlu_std = mlu_mean_std['std']
    use_firstconv = model_context[model_key]['use_firstconv'] if 'use_firstconv' in model_context[model_key].keys() else False
    print("")

    """
    Import data
    """
    image_list = [ pj(args.data,c) for c in os.listdir(args.data) if c.endswith(args.ext) ]
    K = min([len(image_list),args.batch_size])
    image_list = image_list[:K]
    print('sampled %d data'%len(image_list))
    print(image_list[0])
    input_img = [cv2.imread(c) for c in image_list]
    # set mlu=False to always trigger normalization(mean,std)
    data = [preprocess_yolov5(c , dst_size=IMG_SIZE , mean=py_mean, std = py_std, mlu= args.mlu if use_firstconv else False) for c in input_img]
    data = [c[0] for c in data]
    print('len of data: %d'%len(data))
    data = torch.stack(data,dim=0)
    print('data shape =',data.shape)
    if args.mlu:
        if args.half_input:
            data = data.type(torch.HalfTensor)
        data = data.to(ct.mlu_device())

    """
    Import pytorch model on cpu first
    """
    print('==pytorch==')
    use_device = 'cpu'
    loading = True if not args.mlu else False
    yolov5face = model_context[model_key]['yolov5face'] if 'yolov5face' in model_context[model_key].keys() else False
    # yolov5_obj = yolov5_net(detect_sz=IMG_SIZE[0], torchLowVersion=True, fuse=False, singleOutput=True)
    if model_context[model_key]['weights'] is not None:
        model = load_yolov5x6_model(detect_sz=IMG_SIZE, stride= model_context[model_key]['stride'],
                                    cfg=model_context[model_key]['cfg'],
                                    weights=model_context[model_key]['weights'].replace('.pt','.pth'),
                                    preTrained=loading,
                                    yolov5face=yolov5face)
    else:
        print('No weights found...Initial mode started')
        model = load_yolov5x6_model(detect_sz=IMG_SIZE, stride= model_context[model_key]['stride'],
                                    cfg=model_context[model_key]['cfg'],
                                    weights=None,
                                    preTrained=False,
                                    yolov5face=yolov5face)

    print('==end==')
    model = model.eval().float()

    if args.quantization:
        print('doing quantization on cpu')
        use_avg = False if data.shape[0] == 1 else True
        if use_firstconv:
            print('using first conv')
            qconfig = { 'per_channel':False, 'firstconv':use_firstconv, 'mean': mlu_mean , 'std':mlu_std}
        else:
            qconfig = {'per_channel': False, 'firstconv': use_firstconv}
        model_quantized = mlu_quantize.quantize_dynamic_mlu(model, qconfig, dtype=dtype, gen_quant = True)
        print('data.shape=',data.shape)
        preds = model_quantized(data)
        torch.save(model_quantized.state_dict(), model_online_fullname )
        print("int8 quantization end!")
        print('preds[0]: ', preds[0].shape)
        if isinstance(preds, list) or isinstance(preds, tuple):
            # for tn, p in enumerate(preds):
            #     print('#{:d}'.format(tn), p.shape)
            _preds = fetch_cpu_data(preds[0], args.half_input)
        else:
            _preds = fetch_cpu_data(preds, args.half_input)

        print('saving', _preds.shape)
        np.save('cpu_quant_pred.npy', _preds)
    else:
        if not args.mlu:
            print('doing cpu inference')
            with torch.no_grad():
                preds = model(data)
                print('preds[0]: ', preds[0].shape)
                if isinstance(preds, list) or isinstance(preds, tuple):
                    # for tn,p in enumerate(preds):
                    #     print('#{:d}'.format(tn), p.shape)
                    _preds = fetch_cpu_data(preds[0], args.half_input)
                else:
                    _preds = fetch_cpu_data(preds, args.half_input)
                print('saving', _preds.shape)
                np.save('cpu_pred.npy', _preds)
            print("cpu inference finished!")
        else:
            print('doing mlu inference')
            model = mlu_quantize.quantize_dynamic_mlu(model)
            checkpoint = torch.load(model_online_fullname, map_location='cpu')
            model.load_state_dict(checkpoint, strict=False)
            # model.eval().float()
            model = model.eval().float().to(ct.mlu_device())
            if args.jit:
                print('using jit inference')
                randinput = torch.rand(1, 3, IMG_SIZE[1], IMG_SIZE[0]) * 255
                randinput = randinput.to(ct.mlu_device())
                traced_model = torch.jit.trace(model, randinput, check_trace=False)
                # print(traced_model.graph)
                print('start inference')
                preds = traced_model(data)
                print('end inference')
                if isinstance(preds, tuple):
                    for tn,p in enumerate(preds):
                        if isinstance(p, list):
                            for tnn,_p in enumerate(p):
                                print('#{:d}'.format(tnn), _p.shape)
                        else:
                            print('#{:d}'.format(tn), p.shape)
                    _preds = fetch_cpu_data(preds[0], args.half_input)
                else:
                    _preds = fetch_cpu_data(preds, args.half_input)
                print('saving', _preds.shape)
                np.save('mlu_jit_pred.npy', _preds)
                print("mlu inference finished!")
            else:
                print('using layer by layer inference')
                data = data.to(ct.mlu_device())
                preds = model(data)
                print('done')
                if isinstance(preds, list):
                    for tn,p in enumerate(preds):
                        print('#{:d}'.format(tn), p.shape)
                    _preds = fetch_cpu_data(preds[0], args.half_input)
                else:
                    _preds = fetch_cpu_data(preds, args.half_input)
                print('saving', _preds.shape)
                np.save('mlu_pred.npy', _preds)
                print("mlu inference finished!")