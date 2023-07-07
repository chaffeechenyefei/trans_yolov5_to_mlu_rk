import argparse
import sys
import time


sys.path.append('./')  # to run '$ python *.py' files in subdirectories
from utils.onnx_infer import ONNXModel
from onnx_detect import decode_yolo_output_rknn
import numpy as np
import os
pj = os.path.join

from config import model_context, MODELROOT

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_key', type=str, help='{}'.format(list(model_context.keys())))
    parser.add_argument('--fast',action='store_true')
    parser.add_argument('--img',type=str,default=None)
    parser.add_argument('--post',action='store_true',
                        help='use decode module for output. only valid for onnx exported by mode 0')
    parser.add_argument('--mode',type=int, default=0)

    opt = parser.parse_args()
    print(opt)

    model_key = opt.model_key
    print('model_key = {}'.format(model_key))
    if not model_key in model_context.keys():
        print('model_key:{} not founded...'.format(model_key))
        exit(-1)

    w = model_context[model_key]['w'] if 'w' in model_context[model_key].keys() else 736
    h = model_context[model_key]['h'] if 'h' in model_context[model_key].keys() else 416

    pt_weight_full_name = pj(MODELROOT, model_context[model_key]['weights'])
    onnx_weight_full_name = pt_weight_full_name.replace('.pt', '_mode{:d}_rknn.onnx'.format(opt.mode))
    rknn_weight_name = '{}_{:d}x{:d}_mode{:d}{}.rknn'.format(model_key, w , h, opt.mode ,'_precompiled' if opt.fast else '')
    rknn_weight_full_name = pj('rknn/',rknn_weight_name)

    print('input onnx from: {}'.format(onnx_weight_full_name))
    print('output rknn to: {}'.format(rknn_weight_full_name))


    if opt.img is not None and opt.img != '':
        import cv2
        imgpath = opt.img
        print('reading image from {}'.format(imgpath))
        cv_img = cv2.imread(imgpath)
        if cv_img is not None:
            cv_img = cv2.cvtColor(cv_img,cv2.COLOR_BGR2RGB)
            cv_img = cv2.resize(cv_img,dsize=(w,h))
        else:
            print('{} not read'.format(imgpath))
            exit(-1)

        onnx_input = cv_img.astype(np.float32)
        onnx_input = np.transpose(onnx_input, axes=(2,0,1)) #hwc->chw
        onnx_input = onnx_input[np.newaxis,...]
        onnx_input /= 255
        rknn_input = cv_img
    else:
        onnx_input = np.random.random((1, 3, h, w)).astype(np.float32)
        rknn_input = onnx_input[0]*255
        rknn_input = np.transpose(rknn_input,axes=(1,2,0))#chw->hwc
        rknn_input = rknn_input.astype(np.uint8)

    print('onnx_input:', onnx_input.shape)
    print('rknn_input:', rknn_input.shape)

    print('--> data for checking')
    print('--> onnx runtime')
    onnx_model = ONNXModel(onnx_path=onnx_weight_full_name)
    output = onnx_model.forward(onnx_input)
    if isinstance(output, list):
        for o in output:
            print(o.shape)
            print(o.reshape(-1)[:10])
        # print(output[-1].reshape(-1))
    else:
        print(output.shape)
    print('<-- data for checking')

    if opt.post:
        bboxes = decode_yolo_output_rknn(output,det_threshold=0.6,nms_threshold=0.4)
        print(bboxes.shape)

    # outputs = np.load('rknn_output.npy')
    # outputs = [ outputs[...,:2], outputs[...,2:4], outputs[...,4:]]
    # bboxes = decode_yolo_output_rknn(outputs, det_threshold=0.6, nms_threshold=0.4)
    # print(bboxes.shape)
    #
    # for bbox in bboxes:
    #     x1, y1, x2, y2 = [int(pt) for pt in bbox[:4]]
    #     cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
    # cv2.imwrite('result.jpg', cv_img)
    #
    # exit(0)

    from rknn.api import RKNN
    QUANTIZE_ON = False
    rknn = RKNN()
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                reorder_channel='0 1 2',
                target_platform='rk3399pro',
                optimization_level=3,
                quantize_input_node=QUANTIZE_ON,
                output_optimize=1,
                # quantized_algorithm = 'mmse',
                # mmse_epoch = 6,
                )
    print('done')

    # Load Pytorch model
    print('--> Loading model')
    dtype = 'int8'
    pre_compile = True if opt.fast else False
    input_size_list = [[3, h, w]]
    # ret = rknn.load_pytorch(model=pt_weights, input_size_list=input_size_list)
    ret = rknn.load_onnx(model=onnx_weight_full_name)
    if ret != 0:
        print('Load Pytorch model failed!')
        exit(ret)
    print('done')
    print('--> Building model')

    ret = rknn.build(do_quantization=True, dataset='./datasets.txt', pre_compile=pre_compile)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # print('--> Accuracy analysis')
    # rknn.accuracy_analysis(inputs='./dataset.txt', target='rk3399pro')
    # print('done')

    # Export RKNN model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(rknn_weight_full_name)
    if ret != 0:
        print('Export {} failed!'.format(rknn_weight_full_name))
        exit(ret)
    print('done')




    # fast模式下， cpu上不能进行模拟推理
    if not opt.fast:
        print('--> Import RKNN model and infering')
        ret = rknn.load_rknn(rknn_weight_full_name)

        # Init runtime environment
        print('--> Init runtime environment')
        ret = rknn.init_runtime()
        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)
        print('done')

        # Inference
        print('--> Running model')
        outputs = rknn.inference(inputs=[rknn_input])
        for o in outputs:
            print(o.shape)
            print(o.reshape(-1)[:10]) # if model already softmax
        print('done')

        output = np.concatenate(outputs,axis=-1)
        np.save('rknn_output.npy',output)
        if opt.post:
            bboxes = decode_yolo_output_rknn(outputs, det_threshold=0.6, nms_threshold=0.4)
            print(bboxes.shape)

    rknn.release()