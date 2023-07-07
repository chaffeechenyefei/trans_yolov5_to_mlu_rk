from config import model_context, MODELROOT
from api_for_mlu import load_yolo_model, update_models, initial_weights
import torch
import argparse

import os
pj = os.path.join

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_key', default='yolov5s', help='{}'.format(list(model_context.keys())))
    args = parser.parse_args()

    print("Find models in {}".format(MODELROOT))

    model_key = args.model_key
    print('Transfering {}'.format(model_key))
    pretrained = True if model_context[model_key]['weights'] is not None else False
    yolov5_net = load_yolo_model(cfg=model_context[model_key]['cfg'], weights= pj(MODELROOT, model_context[model_key]['weights']), preTrained=pretrained)
    yolov5_net = update_models(yolov5_net)

    if not pretrained:
        print('initial weights...')
        initial_weights(yolov5_net)

    if pretrained:
        torch.save(yolov5_net.state_dict(),  pj(MODELROOT, model_context[model_key]['weights'].replace('.pt','.pth')),_use_new_zipfile_serialization=False)
    else:
        print('no need to store for initial weights')