"""
通用工具模块 (纯 cv2 + numpy, 无 PyTorch 依赖).

Common utility functions (pure cv2 + numpy, no PyTorch dependency).

Provides:
  - create_video_writer: 编码器自动回退的视频写入器创建
  - plot_one_box: 在图像上绘制单个 bbox
  - build_frame_heatmap: 将当前帧 bbox 转换成热度图贡献
  - update_temporal_heatmap: 以秒为粒度维护滑动窗口时间衰减热力图
  - overlay_heatmap: 将热力图映射成伪彩色并与原图 alpha 叠加
"""

from collections import deque

import cv2
import numpy as np


def create_video_writer(save_path, fps, width, height, codec):
    """编码器自动回退，优先高压缩编码以减小文件体积 / Auto-fallback codecs.

    Args:
        save_path: Output video path.
        fps: Frames per second.
        width: Frame width.
        height: Frame height.
        codec: Preferred codec string or 'auto'.

    Returns:
        (writer, chosen_codec, codec_candidates) tuple.
    """
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


def make_even(v):
    """保证偶数值 / Ensure even value."""
    return max((int(v) // 2) * 2, 2)


def plot_one_box(xyxy, img, color=(0, 255, 0), label=None, line_thickness=2):
    """在图像上绘制一个 bbox (纯 cv2 实现, 不依赖 torch).

    Draw one bounding box on image (pure cv2, no torch dependency).
    """
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness)
    if label:
        tf = max(line_thickness - 1, 1)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, tf)
        cv2.rectangle(img, (x1, y1 - th - 3), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), tf, cv2.LINE_AA)


def build_frame_heatmap(shape_hw, det):
    """将当前帧 bbox 转换成热度图贡献 / Convert current-frame bboxes to heat contribution map.

    Args:
        shape_hw: (H, W) of the frame.
        det: (M, 6) array of [x1, y1, x2, y2, conf, cls] or None.

    Returns:
        (H, W) float32 heat contribution map.
    """
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


def update_temporal_heatmap(frame_heat, current_sec, second_bins,
                            heat_sum, heat_time_sum, decay_seconds):
    """以秒为粒度维护滑动窗口, 线性时间衰减.

    Maintain per-second sliding window with linear temporal decay.

    Args:
        frame_heat: Current frame's heat contribution (H, W).
        current_sec: Current video timestamp in seconds.
        second_bins: Mutable deque of (second, heat_ndarray) tuples.
        heat_sum: Mutable cumulative heat sum (H, W).
        heat_time_sum: Mutable cumulative time-weighted heat sum (H, W).
        decay_seconds: Decay window in seconds.

    Returns:
        (H, W) weighted heatmap.
    """
    current_bin_sec = int(current_sec)
    if second_bins and second_bins[-1][0] == current_bin_sec:
        second_bins[-1][1][:] += frame_heat
    else:
        second_bins.append((current_bin_sec, frame_heat.copy()))

    heat_sum += frame_heat
    heat_time_sum += frame_heat * current_bin_sec

    # 清理超过衰减窗口的历史数据 / Drop stale history outside decay window
    while second_bins and (current_sec - second_bins[0][0]) > decay_seconds:
        sec_old, heat_old = second_bins.popleft()
        heat_sum -= heat_old
        heat_time_sum -= heat_old * sec_old

    # 线性时间衰减 / Linear temporal decay: weight = max(0, 1 - age/decay_seconds)
    weighted_heat = (heat_sum * (1.0 - current_sec / decay_seconds) +
                     heat_time_sum / decay_seconds)
    np.maximum(weighted_heat, 0.0, out=weighted_heat)
    return weighted_heat


def overlay_heatmap(im0, weighted_heat, alpha):
    """将热力图映射成伪彩色并与原图 alpha 叠加.

    Map heatmap to color and alpha-blend with original frame.

    Args:
        im0: Original BGR frame.
        weighted_heat: (H, W) float32 heatmap.
        alpha: Overlay alpha in [0, 1].

    Returns:
        Blended BGR frame.
    """
    heat_max = float(weighted_heat.max())
    if heat_max <= 1e-6:
        return im0

    heat_norm = np.clip(weighted_heat / heat_max, 0.0, 1.0)
    heat_u8 = (heat_norm * 255.0).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    return cv2.addWeighted(im0, 1.0 - alpha, heat_color, alpha, 0.0)
