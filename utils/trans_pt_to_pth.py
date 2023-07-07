import torch
from torch import nn
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU
from models.common import Conv
import argparse

model_context = {
    # 人头检测
    'yolov5s-conv-head': {'cfg': '',
                            'weights': '/mnt/projects/ObjectDetection/weights/yolo_weights/Head_v5s_conv/weights/best.pt',
                            'stride':[8., 16., 32.], 'use_firstconv':True}
}

def update_models(model):
    for k, m in model.named_modules():
        # m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='yolov5s-conv-head', help='{}'.format(list(model_context.keys())))
    args = parser.parse_args()

    model_name = args.model_name
    weights = model_context[model_name]['weights']
    yolov5_net = attempt_load(weights, map_location='cpu')
    yolov5_net = update_models(yolov5_net)

    torch.save(yolov5_net.state_dict(), model_context[model_name]['weights'].replace('.pt', '.pth'),
               _use_new_zipfile_serialization=False)
    print('Transferd success!')