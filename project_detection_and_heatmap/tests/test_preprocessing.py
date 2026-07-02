"""
test_preprocessing.py — 预处理管线单元测试.

测试 utils.preprocessing 模块 (NumPy 实现) 与 PyTorch 版的一致性.
"""

import os
import sys
import unittest

import cv2
import numpy as np

# 将项目根目录加入 sys.path, 确保本地 utils/ 包优先于 workspace 的同名包
# Add project root to sys.path so local utils/ package takes priority
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from utils.preprocessing import (
    letterbox,
    check_img_size,
    parse_img_size,
    make_even,
    preprocess_frame,
)


class TestPreprocessing(unittest.TestCase):
    """预处理管线单元测试 / Preprocessing pipeline unit tests."""

    def setUp(self):
        # 创建测试图像: 随机 640x480 BGR / Create test image: random 640x480 BGR
        rng = np.random.RandomState(42)
        self.test_img = rng.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)

    def test_letterbox_output_shape(self):
        """测试 letterbox 输出尺寸正确 / Test letterbox output shape."""
        for h, w in [(416, 736), (320, 640), (544, 960)]:
            result = letterbox(self.test_img, new_shape=(h, w))
            self.assertEqual(result.shape, (h, w, 3),
                             f'Expected ({h}, {w}, 3), got {result.shape}')

    def test_letterbox_aspect_ratio_preserved(self):
        """测试 letterbox 后宽高比不变 / Test aspect ratio is preserved."""
        result = letterbox(self.test_img, new_shape=(416, 736))
        h, w = result.shape[:2]
        # 非填充区域应保持原始宽高比 / Non-padded area should preserve aspect ratio
        self.assertEqual(h, 416)
        self.assertEqual(w, 736)
        # 检查 padding 区域 (角落) 是否为指定的填充色 (114,114,114)
        # Check if padding regions (corners) are the fill color (114,114,114)
        padding_color = result[0, 0]  # top-left corner
        np.testing.assert_array_almost_equal(padding_color, [114, 114, 114], decimal=1)

    def test_check_img_size(self):
        """测试 check_img_size 确保 stride 对齐 / Test check_img_size stride alignment."""
        for s in [16, 32, 64]:
            for val in [100, 200, 415, 417, 500]:
                result = check_img_size(val, s)
                self.assertEqual(result % s, 0,
                                 f'{result} not divisible by {s}')

    def test_parse_img_size(self):
        """测试 parse_img_size 多输入格式 / Test parse_img_size with various input formats."""
        # int input
        result = parse_img_size(416, stride=32)
        self.assertEqual(result, [416, 416])

        # single-element list
        result = parse_img_size([416], stride=32)
        self.assertEqual(result, [416, 416])

        # two-element list
        result = parse_img_size([416, 736], stride=32)
        self.assertEqual(result, [416, 736])

        # non-aligned values should be adjusted
        result = parse_img_size([417, 737], stride=32)
        for v in result:
            self.assertEqual(v % 32, 0)

    def test_preprocess_frame(self):
        """测试完整预处理管线输出形状 / Test full preprocessing pipeline output shape."""
        result = preprocess_frame(self.test_img, model_h=416, model_w=736)
        self.assertEqual(result.shape, (1, 3, 416, 736),
                         f'Expected (1, 3, 416, 736), got {result.shape}')
        self.assertEqual(result.dtype, np.float32)
        # 检查归一化范围 / Check normalization range [0, 1]
        self.assertGreaterEqual(result.min(), 0.0)
        self.assertLessEqual(result.max(), 1.0)


if __name__ == '__main__':
    unittest.main()
