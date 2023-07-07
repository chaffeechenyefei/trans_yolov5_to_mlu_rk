import os
import time
import torch
import torchvision
import cv2
import onnxruntime
import numpy as np

def decode_yolo_output_rknn(output:list, det_threshold=0.6, nms_threshold=0.2):
    """
    :param output:(1, 18837, 2),(1, 18837, 2),(1, 18837, 10) 
    :param det_threshold: 
    :param nms_threshold: 
    :return: 
    """
    output_filter = np.concatenate((output[0][0],output[1][0],output[2][0][:,0:1]), axis=-1) #[-1,5]
    cxcywhs = output_filter[output_filter[:, -1] >= det_threshold, :]  # [-1,5]
    xyxys = cxcywhs2xyxys(cxcywhs)
    box_ids = nms(xyxys, nms_threshold)
    boxes_picked = []
    for ids, box in enumerate(xyxys):
        if ids in box_ids:
            boxes_picked.append(box)
            # x1, y1, x2, y2 = [int(pt) for pt in box.tolist()]
        else:
            pass
    return np.array(boxes_picked)#[-1,5]




def decode_yolo_output_1nc(output , det_threshold=0.6 ,nms_threshold=0.2):
    """
    :param output:[1,b,5]
    :return: [-1,5]
    """
    output_filter = output[0,:,:] #[-1,6]
    # objness for threshold
    # output_filter[:,4] *= output_filter[:,5]
    output_filter = output_filter[:,:5]#[-1,5]
    cxcywhs = output_filter[output_filter[:,-1]>=det_threshold ,:] #[-1,5]
    xyxys = cxcywhs2xyxys(cxcywhs)
    box_ids = nms(xyxys, nms_threshold)
    boxes_picked = []
    for ids, box in enumerate(xyxys):
        if ids in box_ids:
            boxes_picked.append(box)
            # x1, y1, x2, y2 = [int(pt) for pt in box.tolist()]
        else:
            pass
    return np.array(boxes_picked)#[-1,5]

def cxcywhs2xyxys(bboxes):
    """
    :param bboxes: np.array(-1,6(xywhs))
    :return:
    """
    output = bboxes*1;
    w = output[:,2]
    h = output[:,3]
    output[:,0] = output[:,0] - w/2
    output[:,1] = output[:,1] - h/2
    output[:,2] = output[:,0]+w
    output[:,3] = output[:,1]+h
    return output

def nms(boxes, overlap_threshold=0.5, mode='union'):
    """Non-maximum suppression.

    Arguments:
        boxes: a float numpy array of shape [n, 5],
            where each row is (xmin, ymin, xmax, ymax, score).
        overlap_threshold: a float number.
        mode: 'union' or 'min'.
    Returns:
        list with indices of the selected boxes
    """
    # if there are no boxes, return the empty list
    if len(boxes) == 0:
        return []
    # list of picked indices
    pick = []
    # grab the coordinates of the bounding boxes
    x1, y1, x2, y2, score = [boxes[:, i] for i in range(5)]
    area = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    ids = np.argsort(score)  # in increasing order
    while len(ids) > 0:
        # grab index of the largest value
        last = len(ids) - 1
        i = ids[last]
        pick.append(i)

        # compute intersections
        # of the box with the largest score
        # with the rest of boxes

        # left top corner of intersection boxes
        ix1 = np.maximum(x1[i], x1[ids[:last]])
        iy1 = np.maximum(y1[i], y1[ids[:last]])

        # right bottom corner of intersection boxes
        ix2 = np.minimum(x2[i], x2[ids[:last]])
        iy2 = np.minimum(y2[i], y2[ids[:last]])

        # width and height of intersection boxes
        w = np.maximum(0.0, ix2 - ix1 + 1.0)
        h = np.maximum(0.0, iy2 - iy1 + 1.0)

        # intersections' areas
        inter = w * h
        if mode == 'min':
            overlap = inter / np.minimum(area[i], area[ids[:last]])
        elif mode == 'union':
            # intersection over union (IoU)
            overlap = inter / (area[i] + area[ids[:last]] - inter)

        # delete all boxes where overlap is too big
        ids = np.delete(
            ids,
            np.concatenate([[last], np.where(overlap > overlap_threshold)[0]])
        )

    return pick

def cv2_to_1chw(img_cv2, dst_size , m=None, std=None,to_bgr:bool=True):
    """
    :param img_cv2:
    :param dst_size: [W,H]
    :param to_bgr:
    :return:
    """
    h, w = img_cv2.shape[:2]
    dstW, dstH = dst_size
    aspect_ratio = min([dstW / w, dstH / h])
    _h = min([int(h * aspect_ratio), dstH])
    _w = min([int(w * aspect_ratio), dstW])

    padh = dstH - _h
    padw = dstW - _w
    img_resized = cv2.resize(img_cv2, (_w, _h))

    img_dst = np.zeros([dstH,dstW,3],np.uint8)
    img_dst[0:_h,0:_w] = img_resized*1

    img_bgr = img_dst.copy()

    if not to_bgr:
        img_dst = cv2.cvtColor(img_dst, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    else:
        img_dst = img_dst.transpose(2,0,1)

    img_dst = img_dst.astype(np.float32)

    if m is not None:
        m = np.array(m).reshape(-1,1,1)
        std = np.array(std).reshape(-1,1,1)
        img_dst = (img_dst-m)/std

    img_dst = np.expand_dims(img_dst, axis=0)
    img_dst = img_dst.astype(np.float32)

    return {'data': img_dst,
            'aspect_ratio': aspect_ratio,
            'img': img_bgr}
if __name__ == '__main__':
    onnx_fp = '/data_share/weights/head_detection_yolov5/best.onnx'
    onnx_model = onnxruntime.InferenceSession(onnx_fp)
    # resize to (736, 416)
    img_fp = '/data_share/weights/head_detection_yolov5/head.jpg'
    img0 = cv2.imread(img_fp).astype(np.float32)
    data = cv2_to_1chw(img0, [736, 416], [0, 0, 0], [255, 255, 255], False)
    img = data['data']#[1,c,h,w]
    aspect_ratio = data['aspect_ratio']
    print('img.shape: ', img.shape)

    # inference
    inputs = {onnx_model.get_inputs()[0].name: img}
    output = onnx_model.run(None, {'images': img})
    print('output: ', len(output), [o.shape for o in output])

    # postprocess bbox
    bboxes = decode_yolo_output_1nc(output[0], det_threshold=0.2, nms_threshold=0.5)
    bboxes[:, :4] = bboxes[:, :4] / aspect_ratio
    print('bboxes: ', bboxes)

    # plot img
    for bbox in bboxes:
        x1, y1, x2, y2 = [int(pt) for pt in bbox[:4]]
        cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.imwrite('/data_share/weights/head_detection_yolov5/head_res.jpg', img0)