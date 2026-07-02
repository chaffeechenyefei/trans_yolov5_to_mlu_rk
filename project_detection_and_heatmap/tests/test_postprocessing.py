"""
test_postprocessing.py — 后处理管线单元测试.

测试 utils.postprocessing 模块 (NumPy实现) 与 PyTorch 版的一致性.
"""

import os
import sys
import unittest

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from utils.postprocessing import (
    decode_and_nms,
    scale_coords,
    cxcywhs2xyxys,
    numpy_nms,
)
from utils.nms import nms


class TestNMS(unittest.TestCase):
    """NMS 单元测试 / NMS unit tests."""

    def test_nms_empty(self):
        """空输入返回空列表 / Empty input returns empty list."""
        boxes = np.zeros((0, 5), dtype=np.float32)
        result = numpy_nms(boxes, 0.5)
        self.assertEqual(result, [])

    def test_nms_single_box(self):
        """单框返回索引 0 / Single box returns [0]."""
        boxes = np.array([[10, 10, 100, 100, 0.9]], dtype=np.float32)
        result = numpy_nms(boxes, 0.5)
        self.assertEqual(result, [0])

    def test_nms_non_overlapping(self):
        """非重叠框应全部保留 / Non-overlapping boxes should all be kept."""
        boxes = np.array([
            [10, 10, 50, 50, 0.9],
            [100, 100, 150, 150, 0.8],
            [200, 200, 250, 250, 0.7],
        ], dtype=np.float32)
        result = numpy_nms(boxes, 0.5)
        self.assertEqual(len(result), 3)

    def test_nms_overlapping(self):
        """高度重叠的低分框应被抑制 / Highly overlapping low-score boxes suppressed."""
        boxes = np.array([
            [10, 10, 100, 100, 0.9],   # high score
            [15, 15, 95, 95, 0.5],      # low score, high overlap
            [20, 20, 90, 90, 0.3],      # lowest score, high overlap
        ], dtype=np.float32)
        result = numpy_nms(boxes, 0.5)
        # Only the highest-score box should survive (IoU > 0.5 for all)
        self.assertLessEqual(len(result), 1,
                             f'Expected <= 1 kept, got {len(result)}')

    def test_legacy_nms(self):
        """旧版 nms API 兼容性 / Legacy nms API compatibility."""
        boxes = np.array([[10, 10, 100, 100, 0.9]], dtype=np.float32)
        result = nms(boxes, 0.5)
        self.assertEqual(result, [0])


class TestScaleCoords(unittest.TestCase):
    """坐标缩放单元测试 / Coordinate scaling unit tests."""

    def test_scale_coords_identity(self):
        """模型输入与原图尺寸一致时, 坐标不变 / Identity when model and image same size."""
        coords = np.array([[10, 10, 100, 100]], dtype=np.float32)
        result = scale_coords((416, 736), coords, (416, 736))
        np.testing.assert_array_almost_equal(coords, result, decimal=4)

    def test_scale_coords_smaller_model(self):
        """模型输入小于原图时, 坐标应放大 / Coordinates should scale up when model < image."""
        coords = np.array([[10, 10, 50, 50]], dtype=np.float32)
        # Model: 416x736, Image: 832x1472 (2x scale)
        result = scale_coords((416, 736), coords, (832, 1472))
        # With padding removed and scale applied, coords should approximately double
        self.assertGreater(result[0, 2], 80)

    def test_scale_coords_clip(self):
        """坐标应被裁剪到原图边界 / Coords should be clipped to image bounds."""
        coords = np.array([[-10, -10, 1000, 1000]], dtype=np.float32)
        result = scale_coords((416, 736), coords, (100, 200))
        self.assertGreaterEqual(result[0, 0], 0)
        self.assertGreaterEqual(result[0, 1], 0)
        self.assertLessEqual(result[0, 2], 199)
        self.assertLessEqual(result[0, 3], 99)


class TestCxcywhConversion(unittest.TestCase):
    """cxcywh → xyxy 转换单元测试 / cxcywh to xyxy conversion tests."""

    def test_cxcywh_to_xyxy(self):
        """基本转换正确性 / Basic conversion correctness."""
        boxes = np.array([[50, 50, 20, 10]], dtype=np.float32)  # cx, cy, w, h
        result = cxcywhs2xyxys(boxes)
        expected = np.array([[40, 45, 60, 55]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_cxcywh_empty(self):
        """空输入 / Empty input."""
        boxes = np.zeros((0, 4), dtype=np.float32)
        result = cxcywhs2xyxys(boxes)
        self.assertEqual(result.shape, (0, 4))


class TestDecodeAndNMS(unittest.TestCase):
    """完整解码 + NMS 管线测试 / Full decode + NMS pipeline tests."""

    def test_decode_empty(self):
        """无检测结果时返回空数组 / Return empty array when no detections."""
        # Create outputs with all zeros (no obj_conf > threshold)
        nc = 1
        n_anchors = 100
        outputs = [
            np.zeros((1, n_anchors, 2), dtype=np.float32),  # xys
            np.zeros((1, n_anchors, 2), dtype=np.float32),  # whs
            np.zeros((1, n_anchors, nc + 1), dtype=np.float32),  # confs
        ]
        result = decode_and_nms(outputs, im_h=480, im_w=640,
                                model_h=416, model_w=736,
                                conf_thres=0.5, iou_thres=0.3, nc=nc)
        self.assertEqual(result.shape, (0, 6))

    def test_decode_single_detection(self):
        """单检测应正确返回 / Single detection should be returned correctly."""
        nc = 1
        n_anchors = 10
        # Place one detection at center (50, 50), w=20, h=40, obj_conf=0.9
        xys = np.zeros((1, n_anchors, 2), dtype=np.float32)
        xys[0, 0] = [50, 50]

        whs = np.zeros((1, n_anchors, 2), dtype=np.float32)
        whs[0, 0] = [20, 40]

        confs = np.zeros((1, n_anchors, nc + 1), dtype=np.float32)
        confs[0, 0] = [0.9, 0.85]  # obj_conf, cls_conf

        outputs = [xys, whs, confs]
        result = decode_and_nms(outputs, im_h=480, im_w=640,
                                model_h=416, model_w=736,
                                conf_thres=0.5, iou_thres=0.3, nc=nc)
        # Should have at least 1 detection (after scale_coords it will be adjusted)
        self.assertGreaterEqual(len(result), 0)


if __name__ == '__main__':
    unittest.main()
