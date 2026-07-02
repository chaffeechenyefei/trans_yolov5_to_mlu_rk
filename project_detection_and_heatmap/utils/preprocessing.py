"""
预处理工具模块 (NumPy 实现, 无 PyTorch 依赖).

Preprocessing utilities (NumPy implementation, zero PyTorch dependency).

Provides:
  - letterbox: 图像 resize + pad 保持宽高比
  - check_img_size: 确保尺寸是 stride 的整数倍
  - parse_img_size: 解析并校准输入尺寸
  - make_even: 保证偶数值
"""

import cv2
import numpy as np


def make_even(v: int) -> int:
    """保证偶数值, 用于视频编解码器兼容 / Ensure even value for video codec compatibility."""
    return max((int(v) // 2) * 2, 2)


def check_img_size(img_size: int, s: int = 32) -> int:
    """确保 img_size 是 stride s 的整数倍 / Ensure img_size is divisible by stride s."""
    new_size = max(make_even(img_size), s)
    if new_size % s != 0:
        new_size = (new_size // s) * s
    return new_size


def parse_img_size(img_size, stride: int = 32):
    """解析并校准输入尺寸使之为 stride 的整数倍 / Parse and calibrate input size to stride multiples.

    Args:
        img_size: int, [int], or [h, w]
        stride: model stride (e.g. 32 for YOLOv5s)

    Returns:
        [h, w] list of stride-aligned dimensions
    """
    if isinstance(img_size, int):
        s = check_img_size(img_size, stride)
        return [s, s]
    if len(img_size) == 1:
        s = check_img_size(img_size[0], stride)
        return [s, s]
    h = check_img_size(img_size[0], stride)
    w = check_img_size(img_size[1], stride)
    return [h, w]


def letterbox(img: np.ndarray, new_shape=(416, 736), color=(114, 114, 114)) -> np.ndarray:
    """Resize 并 pad 图像到目标尺寸 (保持宽高比), NumPy 实现.

    Resize and pad image to target shape (keep aspect ratio), NumPy implementation.

    Args:
        img: Input BGR image (H, W, 3).
        new_shape: Target (H, W) or int.
        color: Padding color (BGR).

    Returns:
        Padded image of shape (new_shape[0], new_shape[1], 3).
    """
    shape = img.shape[:2]  # [h, w]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 缩放比例 / Scale ratio
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # 计算新尺寸和 padding / Compute new size and padding
    new_unpad_w = int(round(shape[1] * r))
    new_unpad_h = int(round(shape[0] * r))
    dw = new_shape[1] - new_unpad_w
    dh = new_shape[0] - new_unpad_h

    if shape[::-1] != (new_unpad_w, new_unpad_h):
        img = cv2.resize(img, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)

    top = dh // 2
    bottom = dh - top
    left = dw // 2
    right = dw - left
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img


def preprocess_frame(im0: np.ndarray, model_h: int, model_w: int) -> np.ndarray:
    """完整预处理管线: letterbox → BGR→RGB → HWC→CHW → normalize → add batch dim.

    Complete preprocessing pipeline for ONNX inference.

    Args:
        im0: Input BGR frame (H, W, 3).
        model_h: Model input height.
        model_w: Model input width.

    Returns:
        np.ndarray shape (1, 3, H, W) float32, normalized to [0, 1].
    """
    img = letterbox(im0, new_shape=(model_h, model_w))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
    img = np.ascontiguousarray(img)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img
