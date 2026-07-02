"""
detect_video.py — ONNX Runtime 版 YOLOv5 视频/图像检测脚本 (重构版).

功能与 workspace 根目录的 detect_video.py 一致, 但不依赖 PyTorch,
仅需 onnxruntime + opencv-python + numpy.

---
ONNX Runtime-based YOLOv5 image/video detection (refactored).
Same features as the original detect_video.py, but zero PyTorch dependency.

Usage:
    # 视频检测 / Video detection
    python detect_video.py \
      --weights ../weights/yolov5s-people_mode0.onnx \
      --source ../data/videos/rtmart-001.mp4 \
      --sample_fps 25

    # 图像检测 / Image detection
    python detect_video.py \
      --weights ../weights/yolov5s-people_mode0.onnx \
      --source ../data/calibration/test_frame_001.jpg

    # BBox-only 模式 / BBox-only mode
    python detect_video.py \
      --weights ../weights/yolov5s-people_mode0.onnx \
      --source ../data/videos/rtmart-001.mp4 \
      --bbox_output --bbox_normalized --sample_fps 25
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

from onnx_inference import ONNXDetector
from utils.common import (create_video_writer, make_even, plot_one_box,
                           build_frame_heatmap, update_temporal_heatmap,
                           overlay_heatmap)
from utils.preprocessing import parse_img_size


# ============================================================================
# Helper / 辅助函数
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


# ============================================================================
# Image Detection / 图像检测
# ============================================================================


def detect_image():
    """ONNX Runtime 单图检测 / ONNX Runtime single image detection."""
    source = opt.source
    save_dir = opt.save_dir
    os.makedirs(save_dir, exist_ok=True)

    imgsz = parse_img_size(opt.img_size, stride=32)
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

    colors = [[random.randint(0, 255) for _ in range(3)]
              for _ in range(len(detector.names))]
    for i in range(len(det)):
        x1, y1, x2, y2, conf, cls_id = det[i]
        cls_id_int = int(cls_id)
        color = colors[cls_id_int % len(colors)]
        label_name = (detector.names[cls_id_int]
                      if cls_id_int < len(detector.names)
                      else f'cls_{cls_id_int}')
        plot_one_box([x1, y1, x2, y2], im0, color=color,
                     label=f'{label_name} {conf:.2f}', line_thickness=2)

    base_name = os.path.splitext(os.path.basename(source))[0]
    save_path = os.path.join(save_dir, f'{base_name}_detect_onnx.jpg')
    cv2.imwrite(save_path, im0)
    print(f'Result saved to: {save_path}')


# ============================================================================
# Video Detection / 视频检测
# ============================================================================


def detect_video():
    """ONNX Runtime 视频检测主流程 / ONNX Runtime video detection main pipeline."""
    source = opt.source
    save_dir = opt.save_dir
    os.makedirs(save_dir, exist_ok=True)

    imgsz = parse_img_size(opt.img_size, stride=32)
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

    output_scale = max(float(opt.output_scale), 0.05)
    out_width = make_even(int(width * output_scale))
    out_height = make_even(int(height * output_scale))

    video_name = os.path.splitext(os.path.basename(source))[0]
    save_path = os.path.join(save_dir, f'{video_name}_detect_onnx.mp4')

    # ---- bbox-only mode ----
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

    # ---- Video writer ----
    if bbox_only:
        writer = None
        chosen_codec = None
        codec_candidates = []
    else:
        writer, chosen_codec, codec_candidates = create_video_writer(
            save_path, out_fps, out_width, out_height, opt.codec,
        )
        if writer is None:
            cap.release()
            raise RuntimeError(f'Cannot open video writer: {save_path}')

        fallback_tag = ''
        if (str(opt.codec).lower() == 'auto' and
                chosen_codec != codec_candidates[0]):
            fallback_tag = ' (fallback applied)'
        print(f'Codec: {opt.codec} -> {chosen_codec}{fallback_tag} '
              f'(tried: {" -> ".join(codec_candidates)})')
        print(f'Output: {out_width}x{out_height} @ {out_fps:.2f} fps')

    # ---- Heatmap params ----
    heatmap_alpha = min(max(float(opt.heatmap_alpha), 0.0), 1.0)
    if opt.enable_heatmap:
        print(f'HeatMap: decay={max(float(opt.heat_decay_seconds), 1.0):.1f}s, '
              f'alpha={heatmap_alpha:.2f}')

    # ---- Main loop ----
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

            # ONNX inference
            det = detector.infer(im0)  # (M, 6) [x1, y1, x2, y2, conf, cls]

            # ---- bbox-only output ----
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

                # Progress
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
                plot_one_box([x1, y1, x2, y2], im0, color=color,
                             label=f'{label_name} {conf:.2f}', line_thickness=2)

            # ---- Heatmap overlay ----
            if opt.enable_heatmap:
                current_sec = frame_idx / src_fps
                frame_heat = build_frame_heatmap((height, width), det)
                weighted_heat = update_temporal_heatmap(
                    frame_heat=frame_heat, current_sec=current_sec,
                    second_bins=second_bins, heat_sum=heat_sum,
                    heat_time_sum=heat_time_sum,
                    decay_seconds=max(float(opt.heat_decay_seconds), 1.0),
                )
                im0 = overlay_heatmap(im0, weighted_heat, alpha=heatmap_alpha)

            # ---- FPS display ----
            infer_time = max(time.time() - t0, 1e-6)
            infer_fps = 1.0 / infer_time
            infer_fps_ema = (infer_fps if infer_fps_ema is None
                             else infer_fps_ema * 0.9 + infer_fps * 0.1)

            cv2.putText(im0,
                        f'ONNX Infer FPS: {infer_fps_ema:.2f} | Sample: {sample_fps:.2f}',
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2, cv2.LINE_AA)

            # ---- Output scaling + write ----
            out_frame = (cv2.resize(im0, (out_width, out_height),
                                    interpolation=cv2.INTER_AREA)
                         if (out_width != width or out_height != height)
                         else im0)

            writer.write(out_frame)
            processed += 1

            # ---- Progress print ----
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

    # ---- Summary ----
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
# CLI
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ONNX Runtime YOLOv5 图像/视频检测 (Image/Video Detection)')
    parser.add_argument('--weights', type=str,
                        default='../weights/yolov5s-people_mode0.onnx',
                        help='Path to .onnx model file')
    parser.add_argument('--source', type=str, required=True,
                        help='Input image or video path')
    parser.add_argument('--save_dir', type=str, default='data/result',
                        help='Directory to save output')
    parser.add_argument('--img_size', type=int, nargs='+', default=[736, 416],
                        help='Model input size (w h), default: 736 416')
    parser.add_argument('--conf_thres', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.3,
                        help='IoU threshold for NMS')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device: cpu / cuda / mps')
    parser.add_argument('--sample_fps', type=float, default=1.0,
                        help='Sample frame rate')
    parser.add_argument('--codec', type=str, default='auto',
                        help='Video codec: auto/avc1/H264/mp4v/XVID')
    parser.add_argument('--output_scale', type=float, default=1.0,
                        help='Output resolution scale (0-1)')
    parser.add_argument('--enable_heatmap', action='store_true',
                        help='Overlay weighted heatmap on output')
    parser.add_argument('--heat_decay_seconds', type=float, default=60.0,
                        help='Heatmap decay window in seconds')
    parser.add_argument('--heatmap_alpha', type=float, default=0.35,
                        help='Heatmap overlay alpha [0, 1]')
    parser.add_argument('--bbox_output', action='store_true',
                        help='Only output bbox text file, skip video')
    parser.add_argument('--bbox_normalized', action='store_true',
                        help='Normalize bbox to [0,1]')
    parser.add_argument('--names', type=str, default='person',
                        help='Class names: "person" or "0:person,1:head"')

    opt = parser.parse_args()

    mode = 'Image' if _is_image_file(opt.source) else 'Video'
    print(f'{"=" * 55}')
    print(f'ONNX {mode} Detection')
    print(f'  Weights: {opt.weights}')
    print(f'  Source: {opt.source}')
    print(f'  ImgSize: {opt.img_size}')
    print(f'  Conf/IoU: {opt.conf_thres}/{opt.iou_thres}')
    print(f'  Device: {opt.device}')
    print(f'  Names: {opt.names}')
    print(f'{"=" * 55}')

    if _is_image_file(opt.source):
        detect_image()
    else:
        detect_video()
