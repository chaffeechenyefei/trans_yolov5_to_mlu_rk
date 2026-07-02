"""
NumPy 实现的 NMS (Non-Maximum Suppression), 零 PyTorch 依赖.

NumPy NMS implementation, zero PyTorch dependency.

Provides:
  - numpy_nms: 标准 NMS, 与 torchvision.ops.nms 行为一致
  - nms: 兼容旧版 API (支持 union/min 两种模式)
"""

import numpy as np


def numpy_nms(boxes: np.ndarray, iou_thres: float) -> list:
    """NumPy NMS, 与 PyTorch torchvision.ops.nms 行为一致.

    Args:
        boxes: (N, 5) array of [x1, y1, x2, y2, score].
        iou_thres: IoU threshold.

    Returns:
        List of kept box indices.
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    areas = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    order = scores.argsort()[::-1]  # 分数降序 / Descending by score

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1.0)
        h = np.maximum(0.0, yy2 - yy1 + 1.0)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]

    return keep


def nms(boxes: np.ndarray, overlap_threshold: float = 0.5, mode: str = 'union') -> list:
    """兼容旧版 API 的 NMS (支持 union/min 两种模式).

    Legacy NMS API supporting both 'union' and 'min' modes.

    Args:
        boxes: (N, 5) array of [xmin, ymin, xmax, ymax, score].
        overlap_threshold: IoU threshold.
        mode: 'union' (standard IoU) or 'min' (intersection / min area).

    Returns:
        List of kept box indices.
    """
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2, score = [boxes[:, i] for i in range(5)]
    area = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    ids = np.argsort(score)

    pick = []
    while len(ids) > 0:
        last = len(ids) - 1
        i = ids[last]
        pick.append(i)

        ix1 = np.maximum(x1[i], x1[ids[:last]])
        iy1 = np.maximum(y1[i], y1[ids[:last]])
        ix2 = np.minimum(x2[i], x2[ids[:last]])
        iy2 = np.minimum(y2[i], y2[ids[:last]])

        w = np.maximum(0.0, ix2 - ix1 + 1.0)
        h = np.maximum(0.0, iy2 - iy1 + 1.0)
        inter = w * h

        if mode == 'min':
            overlap = inter / np.minimum(area[i], area[ids[:last]])
        elif mode == 'union':
            overlap = inter / (area[i] + area[ids[:last]] - inter)
        else:
            overlap = inter / (area[i] + area[ids[:last]] - inter)

        ids = np.delete(
            ids,
            np.concatenate([[last], np.where(overlap > overlap_threshold)[0]])
        )

    return pick
