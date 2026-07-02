"""
后处理工具模块: YOLO 输出解码 + 坐标缩放 (NumPy 实现, 无 PyTorch 依赖).

Postprocessing utilities: YOLO output decoding + coordinate scaling (NumPy, no PyTorch).

Provides:
  - decode_and_nms: 将 ONNX export mode=0 输出解码为 [x1, y1, x2, y2, conf, cls]
  - scale_coords: 将推理坐标映射回原图分辨率
  - cxcywhs2xyxys: cxcywh → xyxy 转换
"""

import numpy as np

from .nms import numpy_nms


def scale_coords(model_shape, coords: np.ndarray, img_shape) -> np.ndarray:
    """将推理坐标从模型输入空间映射回原图分辨率.
    Map inference coords from model input space back to original image resolution.

    等价于 PyTorch 版 utils.general.scale_coords.
    Equivalent to PyTorch utils.general.scale_coords.

    Args:
        model_shape: (H, W) of model input.
        coords: (N, 4) array of [x1, y1, x2, y2] in model input space.
        img_shape: (H, W) of original image.

    Returns:
        (N, 4) array of [x1, y1, x2, y2] in original image space.
    """
    model_h, model_w = model_shape[0], model_shape[1]
    im_h, im_w = img_shape[0], img_shape[1]

    gain = min(model_h / im_h, model_w / im_w)
    pad_h = (model_h - im_h * gain) / 2
    pad_w = (model_w - im_w * gain) / 2

    result = coords.copy()
    result[:, 0] = (result[:, 0] - pad_w) / gain  # x1
    result[:, 1] = (result[:, 1] - pad_h) / gain  # y1
    result[:, 2] = (result[:, 2] - pad_w) / gain  # x2
    result[:, 3] = (result[:, 3] - pad_h) / gain  # y2

    # clip 到原图范围 / Clip to image bounds
    result[:, 0] = np.clip(result[:, 0], 0, im_w - 1)
    result[:, 1] = np.clip(result[:, 1], 0, im_h - 1)
    result[:, 2] = np.clip(result[:, 2], 0, im_w - 1)
    result[:, 3] = np.clip(result[:, 3], 0, im_h - 1)

    return result


def cxcywhs2xyxys(bboxes: np.ndarray) -> np.ndarray:
    """将中心点坐标 (cx, cy, w, h) 转换为左上右下 (x1, y1, x2, y2).
    Convert center-format (cx, cy, w, h) to corner-format (x1, y1, x2, y2).

    Args:
        bboxes: (N, >=4) array where columns 0-3 are [cx, cy, w, h].

    Returns:
        (N, 4) array of [x1, y1, x2, y2].
    """
    out = bboxes.copy()
    w = out[:, 2]
    h = out[:, 3]
    out[:, 0] = out[:, 0] - w / 2
    out[:, 1] = out[:, 1] - h / 2
    out[:, 2] = out[:, 0] + w
    out[:, 3] = out[:, 1] + h
    return out


def decode_and_nms(outputs: list, im_h: int, im_w: int,
                   model_h: int, model_w: int,
                   conf_thres: float = 0.5, iou_thres: float = 0.3,
                   nc: int = 1) -> np.ndarray:
    """将 ONNX export mode=0 输出解码为 [x1, y1, x2, y2, conf, cls].

    严格匹配 PyTorch non_max_suppression 的行为.
    Decode ONNX export mode=0 outputs; strictly matches PyTorch non_max_suppression.

    Args:
        outputs: ONNX raw outputs [xys, whs, confs], each shape (1, N, ...).
        im_h: Original image height (for scale_coords).
        im_w: Original image width.
        model_h: Model input height.
        model_w: Model input width.
        conf_thres: Confidence threshold.
        iou_thres: IoU threshold for NMS.
        nc: Number of classes.

    Returns:
        (M, 6) array of [x1, y1, x2, y2, conf, cls] in original image coords.
    """
    xys = outputs[0][0]     # (N, 2) [cx, cy] in pixel coords
    whs = outputs[1][0]     # (N, 2) [w, h] in pixel coords
    confs = outputs[2][0]   # (N, nc+1) [obj_conf, cls_conf_0, ...]

    obj_conf = confs[:, 0]       # (N,) obj conf
    cls_conf = confs[:, 1:]      # (N, nc) class conf

    # Step 1: 按 obj_conf 过滤候选 / Filter candidates by obj_conf
    candidate_mask = obj_conf > conf_thres
    if not candidate_mask.any():
        return np.zeros((0, 6), dtype=np.float32)

    xys_c = xys[candidate_mask]
    whs_c = whs[candidate_mask]
    obj_c = obj_conf[candidate_mask][:, None]  # (K, 1)
    cls_c = cls_conf[candidate_mask]            # (K, nc)

    # Step 2: conf = obj * cls (matching PT: x[:, 5:] *= x[:, 4:5])
    cls_c = cls_c * obj_c

    # Step 3: cxcywh → xyxy
    boxes_xyxy = np.zeros((len(xys_c), 4), dtype=np.float32)
    boxes_xyxy[:, 0] = xys_c[:, 0] - whs_c[:, 0] / 2  # x1
    boxes_xyxy[:, 1] = xys_c[:, 1] - whs_c[:, 1] / 2  # y1
    boxes_xyxy[:, 2] = xys_c[:, 0] + whs_c[:, 0] / 2  # x2
    boxes_xyxy[:, 3] = xys_c[:, 1] + whs_c[:, 1] / 2  # y2

    # Step 4: 每个框取最佳类别 / Take best class per box
    best_conf = cls_c.max(axis=1)            # (K,)
    best_cls = cls_c.argmax(axis=1).astype(np.float32)  # (K,)

    # Step 5: 按 combined conf 过滤 / Filter by combined conf
    final_mask = best_conf > conf_thres
    if not final_mask.any():
        return np.zeros((0, 6), dtype=np.float32)

    boxes_xyxy = boxes_xyxy[final_mask]
    best_conf = best_conf[final_mask]
    best_cls = best_cls[final_mask]

    # Step 6: NMS
    boxes_5 = np.column_stack([boxes_xyxy, best_conf])  # (K', 5)
    keep = numpy_nms(boxes_5, iou_thres)

    result_boxes = boxes_xyxy[keep]
    result_conf = best_conf[keep][:, None]
    result_cls = best_cls[keep][:, None]

    result = np.column_stack([result_boxes, result_conf, result_cls])

    # Step 7: 坐标缩放到原图分辨率 / Scale coords to original image
    result[:, :4] = scale_coords((model_h, model_w), result[:, :4], (im_h, im_w))

    return result
