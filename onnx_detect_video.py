"""
onnx_detect_video.py

ONNX Runtime 版 YOLOv5 视频检测脚本.
功能与 detect_video.py 一致, 但不依赖 PyTorch, 仅需 onnxruntime + opencv-python + numpy.
ONNX Runtime-based YOLOv5 video detection. Same features as detect_video.py,
but zero PyTorch dependency — only onnxruntime + opencv-python + numpy.

Usage:
    python onnx_detect_video.py \
      --weights weights/yolov5s-people_mode0.onnx \
      --source data/videos/rtmart-001.mp4 \
      --save_dir data/result \
      --img_size 736 416 \
      --conf_thres 0.5 --iou_thres 0.3 \
      --sample_fps 25
"""

import argparse
import math
import os
import random
import sys
import time
from collections import deque

import cv2
import numpy as np
import onnxruntime as ort


# ============================================================================
# Utility Functions / 工具函数
# ============================================================================


def _make_even(v):
    """保证偶数值, 用于视频编解码器兼容 / Ensure even value for video codec compatibility."""
    return max((int(v) // 2) * 2, 2)


def _create_video_writer(save_path, fps, width, height, codec):
    """编码器自动回退 / Auto-fallback codec selection."""
    if str(codec).lower() == 'auto':
        codec_candidates = ['avc1', 'H264', 'mp4v', 'XVID']
    else:
        codec_candidates = [codec]

    for c in codec_candidates:
        writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*c), fps, (width, height))
        if writer.isOpened():
            return writer, c, codec_candidates
        writer.release()

    return None, None, codec_candidates


def _parse_img_size(img_size, stride=32):
    """解析并校准输入尺寸使之为 stride 的整数倍 / Parse and calibrate input size to stride multiples."""
    if isinstance(img_size, int):
        s = _check_img_size(img_size, stride)
        return [s, s]
    if len(img_size) == 1:
        s = _check_img_size(img_size[0], stride)
        return [s, s]
    h = _check_img_size(img_size[0], stride)
    w = _check_img_size(img_size[1], stride)
    return [h, w]


def _check_img_size(img_size, s=32):
    """确保 img_size 是 stride s 的整数倍 / Ensure img_size is divisible by stride s."""
    new_size = max(_make_even(img_size), s)
    if new_size % s != 0:
        new_size = (new_size // s) * s
    return new_size


def _letterbox(img, new_shape=(416, 736), color=(114, 114, 114)):
    """Resize 并 pad 图像到目标尺寸 (保持宽高比), NumPy 实现.
    Resize and pad image to target shape (keep aspect ratio), NumPy implementation."""
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


def _plot_one_box(xyxy, img, color=(0, 255, 0), label=None, line_thickness=2):
    """在图像上绘制一个 bbox (纯 cv2 实现, 不依赖 torch).
    Draw one bounding box on image (pure cv2, no torch dependency)."""
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness)
    if label:
        tf = max(line_thickness - 1, 1)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, tf)
        cv2.rectangle(img, (x1, y1 - th - 3), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), tf, cv2.LINE_AA)


# ============================================================================
# ONNX Inference Engine / ONNX 推理引擎
# ============================================================================


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
        self.nc = len(self.names)  # 类别数 / Number of classes

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
        print(f'  Classes: {self.names}')

    def preprocess(self, im0: np.ndarray) -> np.ndarray:
        """预处理: letterbox → BGR→RGB → HWC→CHW → normalize → add batch dim.
        Preprocess: letterbox → BGR→RGB → HWC→CHW → normalize → add batch dim."""
        img = _letterbox(im0, new_shape=(self.model_h, self.model_w))
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32) / 255.0  # normalize to [0, 1]
        img = np.expand_dims(img, axis=0)  # (1, 3, H, W)
        return img

    def infer(self, im0: np.ndarray):
        """端到端推理: 预处理 → ONNX 推理 → 解码 + NMS → bbox 列表.
        End-to-end: preprocess → ONNX inference → decode + NMS → bbox list.

        Returns: np.ndarray shape (M, 6) [x1, y1, x2, y2, conf, cls] in original image coords.
        """
        im_h, im_w = im0.shape[:2]
        img = self.preprocess(im0)

        # ONNX 推理 / ONNX inference
        outputs = self.session.run(None, {self.input_name: img})

        # 解码 + NMS / Decode + NMS
        return self._decode_and_nms(outputs, im_h, im_w)

    def _decode_and_nms(self, outputs: list, im_h: int, im_w: int) -> np.ndarray:
        """将 ONNX export mode=0 输出解码为 [x1, y1, x2, y2, conf, cls].
        严格匹配 PyTorch non_max_suppression 的行为.
        Decode ONNX export mode=0 outputs; strictly matches PyTorch non_max_suppression."""
        xys = outputs[0][0]    # (18837, 2) [cx, cy] in pixel coords
        whs = outputs[1][0]    # (18837, 2) [w, h] in pixel coords
        confs = outputs[2][0]  # (18837, nc+1) [obj_conf, cls_conf_0, ...]

        obj_conf = confs[:, 0]     # (18837,) obj conf
        cls_conf = confs[:, 1:]    # (18837, nc) class conf

        # Step 1: 按 obj_conf 过滤候选 / Filter candidates by obj_conf
        candidate_mask = obj_conf > self.conf_thres
        if not candidate_mask.any():
            return np.zeros((0, 6), dtype=np.float32)

        xys_c = xys[candidate_mask]
        whs_c = whs[candidate_mask]
        obj_c = obj_conf[candidate_mask][:, None]   # (K, 1)
        cls_c = cls_conf[candidate_mask]             # (K, nc)

        # Step 2: conf = obj * cls (matching PT: x[:, 5:] *= x[:, 4:5])
        cls_c = cls_c * obj_c

        # Step 3: cxcywh → xyxy
        boxes_xyxy = np.zeros((len(xys_c), 4), dtype=np.float32)
        boxes_xyxy[:, 0] = xys_c[:, 0] - whs_c[:, 0] / 2  # x1
        boxes_xyxy[:, 1] = xys_c[:, 1] - whs_c[:, 1] / 2  # y1
        boxes_xyxy[:, 2] = xys_c[:, 0] + whs_c[:, 0] / 2  # x2
        boxes_xyxy[:, 3] = xys_c[:, 1] + whs_c[:, 1] / 2  # y2

        # Step 4: 每个框取最佳类别 / Take best class per box
        best_conf = cls_c.max(axis=1)           # (K,)
        best_cls = cls_c.argmax(axis=1).astype(np.float32)  # (K,)

        # Step 5: 按 combined conf 过滤 / Filter by combined conf
        final_mask = best_conf > self.conf_thres
        if not final_mask.any():
            return np.zeros((0, 6), dtype=np.float32)

        boxes_xyxy = boxes_xyxy[final_mask]
        best_conf = best_conf[final_mask]
        best_cls = best_cls[final_mask]

        # Step 6: NMS
        boxes_5 = np.column_stack([boxes_xyxy, best_conf])  # (K', 5)
        keep = self._numpy_nms(boxes_5, self.iou_thres)

        result_boxes = boxes_xyxy[keep]
        result_conf = best_conf[keep][:, None]
        result_cls = best_cls[keep][:, None]

        result = np.column_stack([result_boxes, result_conf, result_cls])

        # Step 7: 坐标缩放到原图分辨率 / Scale coords to original image
        # 使用与 PyTorch scale_coords 等价的逻辑 / Equivalent to PyTorch scale_coords
        gain = min(self.model_h / im_h, self.model_w / im_w)
        pad_h = (self.model_h - im_h * gain) / 2
        pad_w = (self.model_w - im_w * gain) / 2

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

    @staticmethod
    def _numpy_nms(boxes: np.ndarray, iou_thres: float) -> list:
        """NumPy NMS, 与 PyTorch torchvision.ops.nms 行为一致.
        NumPy NMS matching PyTorch torchvision.ops.nms behavior."""
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


# ============================================================================
# Heatmap / 热力图
# ============================================================================


def _build_frame_heatmap(shape_hw, det):
    """将当前帧 bbox 转换成热度图贡献 / Convert current-frame bboxes to heat contribution map."""
    h, w = shape_hw
    frame_heat = np.zeros((h, w), dtype=np.float32)
    if det is None or not len(det):
        return frame_heat

    for *xyxy, conf, _cls in det:
        x1 = max(int(xyxy[0]), 0)
        y1 = max(int(xyxy[1]), 0)
        x2 = min(int(xyxy[2]), w - 1)
        y2 = min(int(xyxy[3]), h - 1)
        if x2 <= x1 or y2 <= y1:
            continue
        frame_heat[y1:y2 + 1, x1:x2 + 1] += float(conf)

    return frame_heat


def _update_temporal_heatmap(frame_heat, current_sec, second_bins,
                             heat_sum, heat_time_sum, decay_seconds):
    """以秒为粒度维护滑动窗口 / Maintain per-second sliding window for temporal decay."""
    current_bin_sec = int(current_sec)
    if second_bins and second_bins[-1][0] == current_bin_sec:
        second_bins[-1][1][:] += frame_heat
    else:
        second_bins.append((current_bin_sec, frame_heat.copy()))

    heat_sum += frame_heat
    heat_time_sum += frame_heat * current_bin_sec

    while second_bins and (current_sec - second_bins[0][0]) > decay_seconds:
        sec_old, heat_old = second_bins.popleft()
        heat_sum -= heat_old
        heat_time_sum -= heat_old * sec_old

    weighted_heat = (heat_sum * (1.0 - current_sec / decay_seconds) +
                     heat_time_sum / decay_seconds)
    np.maximum(weighted_heat, 0.0, out=weighted_heat)
    return weighted_heat


def _overlay_heatmap(im0, weighted_heat, alpha):
    """将热力图映射成伪彩色并与原图 alpha 叠加 / Map heatmap to color + alpha-blend with frame."""
    heat_max = float(weighted_heat.max())
    if heat_max <= 1e-6:
        return im0

    heat_norm = np.clip(weighted_heat / heat_max, 0.0, 1.0)
    heat_u8 = (heat_norm * 255.0).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    return cv2.addWeighted(im0, 1.0 - alpha, heat_color, alpha, 0.0)


# ============================================================================
# Main Pipeline / 主流程
# ============================================================================


def _parse_names(names_str: str) -> list:
    """解析 --names 参数: "person" 或 "0:person,1:head".
    Parse --names argument into a list of class names."""
    parts = [p.strip() for p in names_str.split(',')]
    has_colon = any(':' in p for p in parts)
    if has_colon:
        name_dict = {}
        for p in parts:
            if ':' not in p:
                continue
            k, v = p.split(':', 1)
            name_dict[int(k.strip())] = v.strip()
        if name_dict:
            max_idx = max(name_dict.keys())
            return [name_dict.get(i, f'class_{i}') for i in range(max_idx + 1)]
    return parts


_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def _is_image_file(path: str) -> bool:
    """根据扩展名判断是否为图像文件 / Detect if path is an image file by extension."""
    ext = os.path.splitext(path)[1].lower()
    return ext in _IMAGE_EXTENSIONS


def detect_image_onnx():
    """ONNX Runtime 单图检测 / ONNX Runtime single image detection."""
    source = opt.source
    save_dir = opt.save_dir
    os.makedirs(save_dir, exist_ok=True)

    imgsz = _parse_img_size(opt.img_size, stride=32)
    names = _parse_names(opt.names)

    detector = ONNXDetector(
        onnx_path=opt.weights,
        device=opt.device,
        model_h=imgsz[0],
        model_w=imgsz[1],
        conf_thres=opt.conf_thres,
        iou_thres=opt.iou_thres,
        names=names,
    )

    im0 = cv2.imread(source)
    if im0 is None:
        raise RuntimeError(f'Cannot read image: {source}')

    im_h, im_w = im0.shape[:2]
    print(f'Image: {source} | Size: {im_w}x{im_h}')

    t0 = time.time()
    det = detector.infer(im0)
    infer_time = time.time() - t0
    print(f'Detection: {len(det)} objects in {infer_time * 1000:.1f} ms')

    # ---- bbox-only output ----
    if bool(opt.bbox_output):
        bbox_normalized = bool(opt.bbox_normalized)
        base_name = os.path.splitext(os.path.basename(source))[0]
        bbox_path = os.path.join(save_dir, f'{base_name}_bbox_onnx.txt')
        with open(bbox_path, 'w', encoding='utf-8') as f:
            f.write('# frame_id, cls_id, x, y, w, h, conf\n')
            f.write(f'# normalized: {"true" if bbox_normalized else "false"} '
                    f'(x, w normalized by im_w; y, h normalized by im_h)\n')
            for i in range(len(det)):
                x1, y1, x2, y2, conf, cls_id = det[i]
                w_box = x2 - x1
                h_box = y2 - y1
                if bbox_normalized:
                    x_out = x1 / im_w if im_w > 0 else 0.0
                    y_out = y1 / im_h if im_h > 0 else 0.0
                    w_out = w_box / im_w if im_w > 0 else 0.0
                    h_out = h_box / im_h if im_h > 0 else 0.0
                else:
                    x_out, y_out, w_out, h_out = x1, y1, w_box, h_box
                f.write(f'0, {int(cls_id)}, {x_out:.6f}, {y_out:.6f}, '
                        f'{w_out:.6f}, {h_out:.6f}, {float(conf):.6f}\n')
        print(f'BBox saved to: {bbox_path} | normalized={bbox_normalized}')
        return

    # ---- Draw bbox ----
    colors = [[random.randint(0, 255) for _ in range(3)]
              for _ in range(len(detector.names))]
    for i in range(len(det)):
        x1, y1, x2, y2, conf, cls_id = det[i]
        cls_id_int = int(cls_id)
        color = colors[cls_id_int % len(colors)]
        label_name = (detector.names[cls_id_int]
                      if cls_id_int < len(detector.names)
                      else f'cls_{cls_id_int}')
        _plot_one_box([x1, y1, x2, y2], im0, color=color,
                      label=f'{label_name} {conf:.2f}', line_thickness=2)

    base_name = os.path.splitext(os.path.basename(source))[0]
    save_path = os.path.join(save_dir, f'{base_name}_detect_onnx.jpg')
    cv2.imwrite(save_path, im0)
    print(f'Result saved to: {save_path}')


def detect_video_onnx():
    """ONNX Runtime 视频检测主流程 / ONNX Runtime video detection main pipeline."""

    source = opt.source
    save_dir = opt.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # ---- 解析模型输入尺寸 / Parse model input size ----
    imgsz = _parse_img_size(opt.img_size, stride=32)

    # ---- 解析类别名 / Parse class names ----
    names = _parse_names(opt.names)

    # ---- 创建 ONNX 检测器 / Create ONNX detector ----
    detector = ONNXDetector(
        onnx_path=opt.weights,
        device=opt.device,
        model_h=imgsz[0],
        model_w=imgsz[1],
        conf_thres=opt.conf_thres,
        iou_thres=opt.iou_thres,
        names=names,
    )

    # ---- 打开视频源 / Open video source ----
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video source: {source}')

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        src_fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sample_fps = max(opt.sample_fps, 0.01)
    frame_interval = max(int(round(src_fps / sample_fps)), 1)
    out_fps = src_fps / frame_interval
    sampled_total_frames = (int(math.ceil(total_frames / frame_interval))
                            if total_frames > 0 else 0)

    # 输出缩放 / Output scaling
    output_scale = max(float(opt.output_scale), 0.05)
    out_width = _make_even(width * output_scale)
    out_height = _make_even(height * output_scale)

    video_name = os.path.splitext(os.path.basename(source))[0]
    save_path = os.path.join(save_dir, f'{video_name}_detect_onnx.mp4')

    # ---- bbox-only 模式 / bbox-only mode ----
    bbox_only = bool(opt.bbox_output)
    bbox_normalized = bool(opt.bbox_normalized)
    bbox_path = (os.path.join(save_dir, f'{video_name}_bbox_onnx.txt')
                 if bbox_only else None)
    bbox_fp = open(bbox_path, 'w', encoding='utf-8') if bbox_only else None
    if bbox_only:
        bbox_fp.write('# frame_id, cls_id, x, y, w, h, conf\n')
        bbox_fp.write(f'# normalized: {"true" if bbox_normalized else "false"} '
                      f'(x, w normalized by im_w; y, h normalized by im_h)\n')
        print(f'BBox-only mode. Output: {bbox_path} | normalized={bbox_normalized}')

    # ---- 创建视频写入器 / Create video writer ----
    if bbox_only:
        writer = None
        chosen_codec = None
        codec_candidates = []
    else:
        writer, chosen_codec, codec_candidates = _create_video_writer(
            save_path, out_fps, out_width, out_height, opt.codec,
        )
        if writer is None:
            cap.release()
            raise RuntimeError(f'Cannot open video writer: {save_path}')

        fallback_tag = ''
        if (str(opt.codec).lower() == 'auto' and
                chosen_codec != codec_candidates[0]):
            fallback_tag = ' (fallback applied)'
        print(f'Codec: {opt.codec} → {chosen_codec}{fallback_tag} '
              f'(tried: {" → ".join(codec_candidates)})')
        print(f'Output: {out_width}x{out_height} @ {out_fps:.2f} fps')

    # ---- 热力图参数 / Heatmap params ----
    heatmap_alpha = min(max(float(opt.heatmap_alpha), 0.0), 1.0)
    if opt.enable_heatmap:
        print(f'HeatMap: decay={max(float(opt.heat_decay_seconds), 1.0):.1f}s, '
              f'alpha={heatmap_alpha:.2f}')

    # ---- 主循环 / Main loop ----
    frame_idx = 0
    processed = 0
    infer_fps_ema = None
    start_time = time.time()

    second_bins = deque()
    heat_sum = np.zeros((height, width), dtype=np.float32)
    heat_time_sum = np.zeros((height, width), dtype=np.float32)

    try:
        while True:
            ok, im0 = cap.read()
            if not ok:
                break

            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue

            t0 = time.time()

            # ONNX 推理 / ONNX inference
            det = detector.infer(im0)  # (M, 6) [x1, y1, x2, y2, conf, cls]

            # ---- bbox-only 输出 / bbox-only output ----
            if bbox_only:
                im_h, im_w = im0.shape[0], im0.shape[1]
                for i in range(len(det)):
                    x1, y1, x2, y2, conf, cls_id = det[i]
                    w_box = x2 - x1
                    h_box = y2 - y1
                    if bbox_normalized:
                        x_out = x1 / im_w if im_w > 0 else 0.0
                        y_out = y1 / im_h if im_h > 0 else 0.0
                        w_out = w_box / im_w if im_w > 0 else 0.0
                        h_out = h_box / im_h if im_h > 0 else 0.0
                    else:
                        x_out, y_out, w_out, h_out = x1, y1, w_box, h_box
                    bbox_fp.write(
                        f'{frame_idx}, {int(cls_id)}, {x_out:.6f}, {y_out:.6f}, '
                        f'{w_out:.6f}, {h_out:.6f}, {float(conf):.6f}\n')

                # 进度 / Progress
                infer_time = max(time.time() - t0, 1e-6)
                infer_fps = 1.0 / infer_time
                infer_fps_ema = (infer_fps if infer_fps_ema is None
                                 else infer_fps_ema * 0.9 + infer_fps * 0.1)
                processed += 1
                elapsed = max(time.time() - start_time, 1e-6)
                avg_proc_fps = processed / elapsed
                progress_info = ''
                if sampled_total_frames > 0:
                    pct = min(processed / sampled_total_frames * 100.0, 100.0)
                    progress_info = (f' | Progress: {pct:.2f}% '
                                     f'| Remaining: {max(100.0 - pct, 0.0):.2f}%')
                print(f'\rFrames: {processed} | Avg FPS: {avg_proc_fps:.2f} '
                      f'| Cur FPS: {infer_fps_ema:.2f}{progress_info}',
                      end='', flush=True)
                frame_idx += 1
                continue

            # ---- 绘制 bbox / Draw bbox ----
            colors = [[random.randint(0, 255) for _ in range(3)]
                      for _ in range(len(detector.names))]
            for i in range(len(det)):
                x1, y1, x2, y2, conf, cls_id = det[i]
                cls_id_int = int(cls_id)
                color = colors[cls_id_int % len(colors)]
                label_name = (detector.names[cls_id_int]
                              if cls_id_int < len(detector.names)
                              else f'cls_{cls_id_int}')
                _plot_one_box([x1, y1, x2, y2], im0, color=color,
                              label=f'{label_name} {conf:.2f}', line_thickness=2)

            # ---- 热力图叠加 / Heatmap overlay ----
            if opt.enable_heatmap:
                current_sec = frame_idx / src_fps
                frame_heat = _build_frame_heatmap((height, width), det)
                weighted_heat = _update_temporal_heatmap(
                    frame_heat=frame_heat, current_sec=current_sec,
                    second_bins=second_bins, heat_sum=heat_sum,
                    heat_time_sum=heat_time_sum,
                    decay_seconds=max(float(opt.heat_decay_seconds), 1.0),
                )
                im0 = _overlay_heatmap(im0, weighted_heat, alpha=heatmap_alpha)

            # ---- FPS 显示 / FPS display ----
            infer_time = max(time.time() - t0, 1e-6)
            infer_fps = 1.0 / infer_time
            infer_fps_ema = (infer_fps if infer_fps_ema is None
                             else infer_fps_ema * 0.9 + infer_fps * 0.1)

            cv2.putText(im0,
                        f'ONNX Infer FPS: {infer_fps_ema:.2f} | Sample: {sample_fps:.2f}',
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2, cv2.LINE_AA)

            # ---- 输出缩放 + 写入 / Output scaling + write ----
            out_frame = (cv2.resize(im0, (out_width, out_height),
                                    interpolation=cv2.INTER_AREA)
                         if (out_width != width or out_height != height)
                         else im0)

            writer.write(out_frame)
            processed += 1

            # ---- 进度打印 / Progress print ----
            elapsed = max(time.time() - start_time, 1e-6)
            avg_proc_fps = processed / elapsed
            progress_info = ''
            if sampled_total_frames > 0:
                pct = min(processed / sampled_total_frames * 100.0, 100.0)
                progress_info = (f' | Progress: {pct:.2f}% '
                                 f'| Remaining: {max(100.0 - pct, 0.0):.2f}%')
            print(f'\rFrames: {processed} | Avg FPS: {avg_proc_fps:.2f} '
                  f'| Cur FPS: {infer_fps_ema:.2f}{progress_info}',
                  end='', flush=True)

            frame_idx += 1

    except KeyboardInterrupt:
        print('\nInterrupted by Ctrl+C. Saving current output...')
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if bbox_fp is not None:
            bbox_fp.flush()
            bbox_fp.close()
        cv2.destroyAllWindows()

    # ---- 完成总结 / Summary ----
    total_elapsed = max(time.time() - start_time, 1e-6)
    final_progress_info = ''
    if sampled_total_frames > 0:
        pct = min(processed / sampled_total_frames * 100.0, 100.0)
        final_progress_info = (f' Progress: {pct:.2f}% '
                               f'| Remaining: {max(100.0 - pct, 0.0):.2f}%.')

    if bbox_only:
        print(f'\nDone. BBox saved to: {bbox_path}. '
              f'{processed} frames in {total_elapsed:.2f}s.{final_progress_info}')
    else:
        print(f'\nDone. Output saved to: {save_path}. '
              f'{processed} frames in {total_elapsed:.2f}s.{final_progress_info}')
        if chosen_codec:
            print(f'Final codec: {chosen_codec}')


# ============================================================================
# CLI / 命令行接口
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ONNX Runtime YOLOv5 图像/视频检测 (Image/Video Detection)')
    parser.add_argument('--weights', type=str, default='weights/yolov5s-people_mode0.onnx',
                        help='Path to .onnx model file')
    parser.add_argument('--source', type=str, default='data/calibration/test_frame_001.jpg',
                        help='Input image or video path, default: data/calibration/test_frame_001.jpg')
    parser.add_argument('--save_dir', type=str, default='data/result',
                        help='Directory to save output')
    parser.add_argument('--img_size', type=int, nargs='+', default=[736, 416],
                        help='Model input size (w h), default: 736 416')
    parser.add_argument('--conf_thres', type=float, default=0.5,
                        help='Confidence threshold, default: 0.5')
    parser.add_argument('--iou_thres', type=float, default=0.3,
                        help='IoU threshold for NMS, default: 0.3')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device: cpu / cuda / mps')
    parser.add_argument('--sample_fps', type=float, default=1.0,
                        help='Sample frame rate for detection, default: 1')
    parser.add_argument('--codec', type=str, default='auto',
                        help='Video codec: auto/avc1/H264/mp4v/XVID')
    parser.add_argument('--output_scale', type=float, default=1.0,
                        help='Output resolution scale (0-1), default: 1.0')
    parser.add_argument('--enable_heatmap', action='store_true',
                        help='Overlay weighted heatmap on output')
    parser.add_argument('--heat_decay_seconds', type=float, default=60.0,
                        help='Heatmap decay window in seconds, default: 60')
    parser.add_argument('--heatmap_alpha', type=float, default=0.35,
                        help='Heatmap overlay alpha [0, 1], default: 0.35')
    parser.add_argument('--bbox_output', action='store_true',
                        help='Only output bbox text file, skip video writing')
    parser.add_argument('--bbox_normalized', action='store_true',
                        help='Normalize bbox to [0,1] (only with --bbox_output)')
    parser.add_argument('--names', type=str, default='person',
                        help='Class names: "person" or "0:person,1:head"')

    opt = parser.parse_args()

    mode = 'Image' if _is_image_file(opt.source) else 'Video'
    print(f'{"=" * 55}')
    print(f'ONNX {mode} Detection')
    print(f'  Weights: {opt.weights}')
    print(f'  Source:  {opt.source}')
    print(f'  Save:    {opt.save_dir}')
    print(f'  Conf: {opt.conf_thres}, IoU: {opt.iou_thres}')
    print(f'  Device: {opt.device}')
    if mode == 'Video':
        print(f'  Sample FPS: {opt.sample_fps}')
    print(f'{"=" * 55}')

    if mode == 'Image':
        detect_image_onnx()
    else:
        detect_video_onnx()
