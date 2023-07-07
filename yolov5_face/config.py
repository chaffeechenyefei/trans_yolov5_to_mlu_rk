MODELROOT = '/project/git/shawn_qian/objectdetection/yolov5_face'
model_context = {
    #车牌检测 **anchors:  tensor([  4.,   5.,   8.,  10.,  13.,  16.,  23.,  29.,  43.,  55.,  73., 105., 146., 217., 231., 300., 335., 433.])
    'yolov5s-face-licplate-20220815'   :{'cfg':'./models/yolov5s-face-licplate.yaml',
                                     'weights':'weights/yolov5s-face-licplate-20220815.pt',
                                     'stride':[8., 16., 32.], 'use_firstconv':True, 'w': 736, 'h': 416 },

}