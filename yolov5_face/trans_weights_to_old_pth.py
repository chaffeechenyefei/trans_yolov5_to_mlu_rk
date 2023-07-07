# -*- coding:utf-8 -*-
from yolov5_api import load_yolo_model, update_models, initial_weights
from utils.activations import *
import argparse

model_context = {
    'yolov5s'   :{'cfg':'./models/yolov5s.yaml','weights':'./weights/yolov5s.pt', 'stride':[8., 16., 32] },
    'yolov5l'   :{'cfg':'./models/yolov5l.yaml','weights':'./weights/yolov5l.pt', 'stride':[8., 16., 32.] },
    'yolov5x'   :{'cfg':'./models/yolov5x.yaml','weights':'./weights/yolov5x.pt', 'stride':[8., 16., 32.]},
    'yolov5s6'  :{'cfg':'./models/hub/yolov5s6.yaml','weights':'./weights/yolov5s6.pt', 'stride':[8., 16., 32., 64.] },
    'yolov5x6'  :{'cfg':'./models/hub/yolov5x6.yaml', 'weights':'./weights/yolov5x6.pt', 'stride':[8., 16., 32., 64]},
    'yolov5l6'  :{'cfg':'./models/hub/yolov5l6.yaml', 'weights':'./weights/yolov5l6.pt', 'stride':[8., 16., 32., 64]},
    'yolov5m6'  :{'cfg':'./models/hub/yolov5m6.yaml', 'weights':'./weights/yolov5m6.pt', 'stride':[8., 16., 32., 64]},

    'yolov3'        :{'cfg':'./models/hub/yolov3.yaml',         'weights':None, 'stride':[8., 16., 32.]},
    'yolov3-tiny'   :{'cfg':'./models/hub/yolov3-tiny.yaml',    'weights':None, 'stride':[16., 32.], 'use_firstconv':True},
    'yolov5s-rand'  :{'cfg':'./models/yolov5s.yaml',            'weights':None, 'stride':[8., 16., 32.] },
    'yolov5s-conv'  :{'cfg':'./models/yolov5s-conv.yaml',       'weights':None, 'stride':[8., 16., 32.], 'use_firstconv':True },
    'yolov5x-rand'  :{'cfg':'./models/yolov5x.yaml',            'weights':None, 'stride':[8., 16., 32.]},
    'yolov5x-conv'  :{'cfg':'./models/yolov5x-conv.yaml',       'weights':None, 'stride':[8., 16., 32.]},
    # people detection mAP=68.4%
    'yolov5s-people'   :{'cfg':'./models/yolov5s-people.yaml',          'weights':'./weights/people_detection/yolov5s-people.pt', 'stride':[8., 16., 32] },
    # mAP = 64.5%
    'yolov5s-conv-people'           :{'cfg':'./models/yolov5s-conv-people.yaml', 'weights':'./weights/people_detection/yolov5s-conv-people.pt', 'stride':[8., 16., 32.], 'use_firstconv':True },
    'yolov5s-conv-people-aug-fall'  :{'cfg':'./models/yolov5s-conv-people.yaml', 'weights':'./weights/people_detection_aug_fall/yolov5s-conv-people-aug-fall.pt', 'stride':[8., 16., 32.], 'use_firstconv':True },
    #mAP = 80.9
    'yolov5x6-people'  :{'cfg':'./models/hub/yolov5x6-people.yaml',     'weights':'./weights/people_detection/yolov5x6-people.pt', 'stride':[8., 16., 32., 64]},
    'yolov5s-conv-9'  :{
        'cfg':'./models/yolov5s-conv-9.yaml',           'weights':'./weights/nine_class_detection/yolov5s-conv-9.pt', 'stride':[8., 16., 32.], 'use_firstconv':True ,
        'class_order': [ 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
                        },
    'yolov5s-conv-9-20210927'  :{
        'cfg':'./models/yolov5s-conv-9.yaml',           'weights':'./weights/nine_class_detection/yolov5s-conv-9-20210927.pt', 'stride':[8., 16., 32.], 'use_firstconv':True ,
        'class_order': [ 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
                        },
    'yolov5s-conv-9-20211104'  :{
        'cfg':'./models/yolov5s-conv-9.yaml',           'weights':'./weights/nine_class_detection/yolov5s-conv-9-20211104.pt', 'stride':[8., 16., 32.], 'use_firstconv':True ,
        'class_order': [ 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
                        },
    #火焰检测
    'yolov5s-conv-fire'             :{'cfg':'./models/yolov5s-conv-people.yaml', 'weights':'./weights/fire_detection/yolov5s-conv-fire.pt', 'stride':[8., 16., 32.], 'use_firstconv':True },
    'yolov5s-conv-fire-21102010'    :{'cfg':'./models/yolov5s-conv-people.yaml', 'weights':'./weights/fire_detection/yolov5s-conv-fire-21102010.pt', 'stride':[8., 16., 32.], 'use_firstconv':True },
    'yolov5s-conv-fire-220407'      :{'cfg':'./models/yolov5s-conv-people.yaml', 'weights':'./weights/fire_detection/yolov5s-conv-fire-220407.pt', 'stride':[8., 16., 32.], 'use_firstconv':True },
    #摔倒检测
    'yolov5s-conv-fall-ped'   :{'cfg':'./models/yolov5s-conv-people.yaml', 'weights':'./weights/people_fall_detection/yolov5s-conv-fall-ped.pt', 'stride':[8., 16., 32.], 'use_firstconv':True },
    #摔倒检测+
    'yolov5s-conv-fall-ped-20220301'   :{'cfg':'./models/yolov5s-conv-people.yaml', 'weights':'./weights/people_fall_detection/yolov5s-conv-fall-ped-20220301.pt', 'stride':[8., 16., 32.], 'use_firstconv':True },
    #摔倒检测+ 效果不佳
    'yolov5s-conv-fall-ped-20220222'   :{'cfg':'./models/yolov5s-conv-people.yaml', 'weights':'./weights/people_fall_detection/yolov5s-conv-fall-ped-20220222.pt', 'stride':[8., 16., 32.], 'use_firstconv':True },
    #安全帽检测
    'yolov5s-conv-safety-hat'   :{'cfg':'./models/yolov5s-conv-safetyhat.yaml', 'weights':'./weights/safety_hat/yolov5s-conv-safety-hat.pt', 'stride':[8., 16., 32.], 'use_firstconv':True },
    #安全帽检测+
    'yolov5s-conv-safety-hat-20220217'   :{'cfg':'./models/yolov5s-conv-safetyhat.yaml', 'weights':'./weights/safety_hat/yolov5s-conv-safety-hat-20220217.pt', 'stride':[8., 16., 32.], 'use_firstconv':True },
    #垃圾袋检测
    'yolov5s-conv-trashbag'   :{'cfg':'./models/yolov5s-conv-trashbag.yaml', 'weights':'./weights/trash_bag/yolov5s-conv-trashbag.pt', 'stride':[8., 16., 32.], 'use_firstconv':True },
    #垃圾袋检测-2021-12-14
    'yolov5s-conv-trashbag-20211214'   :{'cfg':'./models/yolov5s-conv-trashbag.yaml', 'weights':'./weights/trash_bag/yolov5s-conv-trashbag-20211214.pt', 'stride':[8., 16., 32.], 'use_firstconv':True },
    #横幅标语检测
    'yolov5s-conv-banner-20211130'   :{'cfg':'./models/yolov5s-conv-people.yaml', 'weights':'./weights/banner_detection_v5s_conv/yolov5s-conv-banner-20211130.pt', 'stride':[8., 16., 32.], 'use_firstconv':True ,
                                       'class_order':['trashbag','cardboard','bottle','trash_on_sea']},
    #非机动车检测
    'yolov5s-conv-motor-20211209'   :{'cfg':'./models/yolov5s-conv-motor.yaml', 'weights':'./weights/motor_detection_v5s_conv/yolov5s-conv-motor-20211209.pt', 'stride':[8., 16., 32.], 'use_firstconv':True,
                                      'class_order': ['motor','bicycle']},
    'yolov5s-conv-motor-20211217'   :{'cfg':'./models/yolov5s-conv-motor.yaml', 'weights':'./weights/motor_detection_v5s_conv/yolov5s-conv-motor-20211217.pt', 'stride':[8., 16., 32.], 'use_firstconv':True,
                                      'class_order': ['motor','bicycle']},
    #手的检测 224x320(wh)
    'yolov5s-conv-hand-20220117': {'cfg': './models/yolov5s-conv-people.yaml',
                                    'weights': './weights/hand_detection_v5s_conv/yolov5s-conv-hand-20220117.pt',
                                    'stride': [8., 16., 32.], 'use_firstconv': True,
                                    'class_order': ['hand']},
    # 手的检测 736x416(wh)
    'yolov5s-conv-hand-20220118': {'cfg': './models/yolov5s-conv-people.yaml',
                                   'weights': './weights/hand_detection_v5s_conv_736/yolov5s-conv-hand-20220118.pt',
                                   'stride': [8., 16., 32.], 'use_firstconv': True,
                                   'class_order': ['hand']},
    # 香烟检测 256x256(wh)
    'yolov5s-conv-cig-20220222': {'cfg': './models/yolov5s-conv-people.yaml',
                                   'weights': './weights/cig_detection_v5s_conv/yolov5s-conv-cig-20220222.pt',
                                   'stride': [8., 16., 32.], 'use_firstconv': True,
                                   'class_order': ['cig']},
    'yolov5s-conv-cig-20220224': {'cfg': './models/yolov5s-conv-people.yaml',
                                   'weights': './weights/cig_detection_v5s_conv/yolov5s-conv-cig-20220224.pt',
                                   'stride': [8., 16., 32.], 'use_firstconv': True,
                                   'class_order': ['cig']},
    'yolov5s-conv-cig-20220301': {'cfg': './models/yolov5s-conv-people.yaml',
                                   'weights': './weights/cig_detection_v5s_conv/yolov5s-conv-cig-20220301.pt',
                                   'stride': [8., 16., 32.], 'use_firstconv': True,
                                   'class_order': ['cig']},
    'yolov5s-conv-cig-20220311': {'cfg': './models/yolov5s-conv-people.yaml',
                                  'weights': './weights/cig_detection_v5s_conv/yolov5s-conv-cig-20220311.pt',
                                  'stride': [8., 16., 32.], 'use_firstconv': True,
                                  'class_order': ['cig']},
    #人头检测
    'yolov5s-conv-head-20220121'   :{'cfg':'./models/yolov5s-conv-people.yaml', 'weights':'./weights/head_detection/yolov5s-conv-head-20220121.pt', 'stride':[8., 16., 32.], 'use_firstconv':True },
    #车牌检测
    'yolov5s-face-licplate-20220815'  :{'cfg':'./models/yolov5s-face.yaml', 'weights':'./weights/licplate_detection/yolov5s-face-licplate-20220815.pt', 'stride':[8., 16., 32.], 'use_firstconv':True, 'yolov5face':True },
}

"""
yolov5s.pt
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='yolov5s-conv-licplate-20220815', help='{}'.format(list(model_context.keys())))
    args = parser.parse_args()

    model_name = args.model_name
    print('Transfering {}'.format(model_name))
    pretrained = True if model_context[model_name]['weights'] is not None else False
    yolov5_net = load_yolo_model(cfg=model_context[model_name]['cfg'], weights=model_context[model_name]['weights'], preTrained=pretrained)
    yolov5_net = update_models(yolov5_net)

    if not pretrained:
        print('initial weights...')
        initial_weights(yolov5_net)

    if pretrained:
        torch.save(yolov5_net.state_dict(), model_context[model_name]['weights'].replace('.pt','.pth'),_use_new_zipfile_serialization=False)
    else:
        print('no need to store for initial weights')