from __future__ import division
import sys,os
pj = os.path.join


import os
import sys
import logging
import argparse
import torch
import torchvision.models as models
import torch_mlu.core.mlu_model as ct
import torch_mlu.core.mlu_quantize as mlu_quantize

#configure logging path
logging.basicConfig(level=logging.INFO,
                    format='[genoff.py line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger("TestNets")

abs_path = os.path.dirname(os.path.realpath(__file__))

torch.set_grad_enabled(False)

def get_args():
    parser = argparse.ArgumentParser(description='Generate offline model.')
    parser.add_argument("-model", dest='model', help=
                        "The network name of the offline model needs to be generated",
                        default="", type=str)
    parser.add_argument("-core_number", dest="core_number", help=
                        "Core number of offline model with simple compilation. ",
                        default=1, type=int)
    parser.add_argument("-mname", dest='mname', help=
                        "The name for the offline model to be generated",
                        default="offline", type=str)
    parser.add_argument("-mcore", dest='mcore', help=
                        "Specify the offline model run device type.",
                        default="MLU270", type=str)
    parser.add_argument("-modelzoo", dest='modelzoo', type=str,
                        help="Specify the path to the model weight file.",
                        default=None)
    parser.add_argument("-channel_size", dest="channel_size", help=
                        "channel size for one inference.",
                        default=3, type=int)
    parser.add_argument("-batch_size", dest="batch_size", help="batch size for one inference.",
                        default=1, type=int)
    parser.add_argument("-in_height", dest="in_height", help="input height.",
                        default=224, type=int)
    parser.add_argument("-in_width", dest="in_width", help="input width.",
                        default=224, type=int)
    parser.add_argument("-half_input", dest='half_input', help=
                        "the input data type, 0-float32, 1-float16/Half, default 1.",
                        default=1, type=int)
    parser.add_argument("-fake_device", dest='fake_device', help=
                        "genoff offline cambricon without mlu device if \
                        fake device is true. 1-fake_device, 0-mlu_device",
                        default=1, type=int)
    parser.add_argument("-quantized_mode", dest='quantized_mode', help=
                        "the data type, 1-mlu int8, 2-mlu int16, default 1.",
                        default=1, type=int)
    parser.add_argument("-input_format", dest="input_format", help=
                        "describe input image channel order in C direction, \
                        0-rgba, 1-argb, 2-bgra, 3-abgr",
                        default=0, type=int)
    parser.add_argument("-autotune", dest="autotune", help="autotune mode",
                        default=0, type=int)
    parser.add_argument("-autotune_config_path", dest="autotune_config_path", \
                        help="autotune configuration file path", default="config.ini", type=str)
    parser.add_argument("-autotune_time_limit", dest="autotune_time_limit", \
                        help="time limit for autotune", default=20, type=int)

    return parser.parse_args()

def genoff_yolov5(model, mname, batch_size, core_number, in_height, in_width,
           half_input, input_format, fake_device):
    # set offline flag
    ct.set_core_number(core_number)
    ct.set_core_version(mcore)
    if fake_device:
        ct.set_device(-1)
    ct.save_as_cambricon(mname)
    ct.set_input_format(input_format)

    if autotune:
        print('using autotune')
        ct.set_autotune(True)
        ct.set_autotune_config_path(autotune_config_path)
        ct.set_autotune_time_limit(autotune_time_limit)

    from yolov5_api import load_yolov5x6_model, load_yolo_model
    from trans_weights_to_old_pth import model_context
    model_path = './weights'
    model_name = model + '.pth'
    assert model in model_context.keys()
    model_online_fullname = pj(model_path, 'mlu_int8_{}_{:d}x{:d}.pth'.format(model, in_width, in_height))
    IMG_SIZE = [in_width, in_height]  # [w,h]

    """
    Import pytorch model on cpu first
    """
    print('==pytorch==')
    use_device = 'cpu'
    use_firstconv = model_context[model]['use_firstconv'] if 'use_firstconv' in model_context[
        model].keys() else False
    loading = False
    # yolov5_obj = yolov5_net(detect_sz=IMG_SIZE[0], torchLowVersion=True, fuse=False, singleOutput=True)
    # model = load_yolov5x6_model(detect_sz=IMG_SIZE[0], preTrained=loading)
    yolov5face = model_context[model]['yolov5face'] if 'yolov5face' in model_context[model].keys() else False
    net = load_yolov5x6_model(detect_sz=IMG_SIZE, stride= model_context[model]['stride'],
                              cfg=model_context[model]['cfg'], weights=None, preTrained=loading, yolov5face=yolov5face)
    print('==end==')
    net = net.eval().float()

    net = mlu_quantize.quantize_dynamic_mlu(net)
    checkpoint = torch.load(model_online_fullname, map_location='cpu')
    net.load_state_dict(checkpoint, strict=False)

    # prepare input
    example_mlu = torch.randn(batch_size, args.channel_size, IMG_SIZE[1], IMG_SIZE[0], dtype=torch.float)
    out = net(example_mlu)
    print(len(out))
    print(out[0].shape)
    print(out[0][0])
    randn_mlu = torch.randn(1, args.channel_size, IMG_SIZE[1], IMG_SIZE[0], dtype=torch.float)
    if use_firstconv:
        example_mlu = example_mlu*255
        randn_mlu = randn_mlu*255

    if half_input:
        randn_mlu = randn_mlu.type(torch.HalfTensor)
        example_mlu = example_mlu.type(torch.HalfTensor)

    net = net.to(ct.mlu_device())
    net_traced = torch.jit.trace(net.to(ct.mlu_device()),
                                 randn_mlu.to(ct.mlu_device()),
                                 check_trace=False)

    # run inference and save cambricon
    net_traced(example_mlu.to(ct.mlu_device()))



if __name__ == "__main__":
    args = get_args()
    model = args.model
    core_number = args.core_number
    modelzoo = args.modelzoo
    mcore = args.mcore
    batch_size = args.batch_size
    in_height = args.in_height
    in_width = args.in_width
    half_input = args.half_input
    input_format = args.input_format
    fake_device = args.fake_device
    autotune = args.autotune
    autotune_config_path = args.autotune_config_path
    autotune_time_limit = args.autotune_time_limit

    #check param
    assert model != "", "Generating the offline model requires" + \
        "specifying the generated network name."
    assert not fake_device or not autotune, "Fake device is not supported for autotune!"

    # env
    if modelzoo != None:
        os.environ['TORCH_HOME'] = modelzoo
        logger.info("TORCH_HOME: " + modelzoo)
    # else:
    #     TORCH_HOME = os.getenv('TORCH_HOME')
    #     if TORCH_HOME == None:
    #         print("Warning: please set environment variable TORCH_HOME such as $PWD/models/pytorch")
    #         exit(1)

    #genoff
    platform_name = 'mlu220' if mcore=='MLU220' else 'mlu270'
    fpX = "_fp16" if half_input else ""
    if autotune:
        mname = './{}/{}_auto_{}x{}_{}_bs{}c{}{}'.format(platform_name, model, in_width, in_height, platform_name, batch_size, core_number, fpX)
    else:
        mname = './{}/{}_{}x{}_{}_bs{}c{}{}'.format(platform_name, model, in_width, in_height, platform_name, batch_size, core_number, fpX)

    logger.info("Generate offline model: " + model)
    # genoff(model, mname, batch_size, core_number,
    #        in_height, in_width, half_input, input_format, fake_device)
    genoff_yolov5(model, mname, batch_size, core_number,
           in_height, in_width, half_input, input_format, fake_device)