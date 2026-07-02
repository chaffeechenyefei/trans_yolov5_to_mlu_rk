"""
ONNX Runtime 推理引擎封装 / ONNX Runtime inference engine wrapper.

封装 ONNXDetector 类, 提供统一的预处理 → 推理 → 后处理管线.
Wraps the ONNXDetector class with unified preprocess → infer → postprocess pipeline.

Usage:
    detector = ONNXDetector('model.onnx', model_h=416, model_w=736)
    detections = detector.infer(frame)  # returns (M, 6) array
"""

import onnxruntime as ort
import numpy as np

from utils.preprocessing import preprocess_frame
from utils.postprocessing import decode_and_nms


def _get_onnx_providers(device: str) -> list:
    """将 CLI --device 映射到 ONNX Runtime execution providers.
    Map CLI --device to ONNX Runtime execution providers."""
    device_lower = device.lower()
    if device_lower in ('cuda', 'gpu', '0'):
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']
    elif device_lower in ('mps', 'coreml'):
        return ['CoreMLExecutionProvider', 'CPUExecutionProvider']
    else:
        return ['CPUExecutionProvider']


class ONNXDetector:
    """ONNX Runtime YOLOv5 检测器封装 / ONNX Runtime YOLOv5 detector wrapper."""

    def __init__(self, onnx_path: str, device: str = 'cpu',
                 model_h: int = 416, model_w: int = 736,
                 conf_thres: float = 0.5, iou_thres: float = 0.3,
                 names: list = None):
        providers = _get_onnx_providers(device)
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.names = names or ['person']
        self.nc = len(self.names)

        # 从 ONNX 模型获取实际输入尺寸 / Get actual input size from ONNX model
        onnx_shape = self.session.get_inputs()[0].shape
        self.model_h = model_h
        self.model_w = model_w
        if onnx_shape[2] is not None and onnx_shape[3] is not None:
            self.model_h = onnx_shape[2]
            self.model_w = onnx_shape[3]

        print(f'ONNXDetector initialized:')
        print(f'  Model: {onnx_path}')
        print(f'  Input: H={self.model_h}, W={self.model_w}')
        print(f'  Providers: {self.session.get_providers()}')
        print(f'  Classes ({self.nc}): {self.names}')

    def infer(self, im0: np.ndarray) -> np.ndarray:
        """端到端推理: 预处理 → ONNX 推理 → 解码 + NMS → bbox 列表.

        End-to-end: preprocess → ONNX inference → decode + NMS → bbox list.

        Returns:
            (M, 6) array of [x1, y1, x2, y2, conf, cls] in original image coords.
        """
        im_h, im_w = im0.shape[:2]
        img = preprocess_frame(im0, self.model_h, self.model_w)

        # ONNX 推理 / ONNX inference
        outputs = self.session.run(None, {self.input_name: img})

        # 解码 + NMS / Decode + NMS
        return decode_and_nms(
            outputs, im_h, im_w,
            self.model_h, self.model_w,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            nc=self.nc,
        )
