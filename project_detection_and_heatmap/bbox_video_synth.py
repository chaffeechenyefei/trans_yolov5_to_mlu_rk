"""
bbox_video_synth.py

将 detect_video.py 通过 --bbox_output 产出的 bbox 文本文件
与原始视频合成带 bbox 的结果视频, 便于直接查看结果。

依赖: opencv-python + numpy (零 PyTorch/ONNX 依赖).

---
Synthesize a bbox-annotated video by overlaying the bbox text file
(produced by detect_video.py --bbox_output) onto the original source video.
Dependencies: opencv-python + numpy (zero PyTorch/ONNX dependency).
"""

import argparse
import math
import os
import random
import time
from collections import defaultdict

import cv2

from utils.common import create_video_writer, make_even


# 默认绘制颜色与线宽, 与 detect.py 风格保持一致 / Default draw style aligned with detect.py
_DEFAULT_LINE_THICKNESS = 2
_DEFAULT_LABEL_BG_ALPHA = 0.55  # label 背景填充透明度 / label background alpha


def _stable_color(cls_id):
    # 同一类别固定一种颜色, 便于视觉对比 / Stable per-class color for visual consistency
    rng = random.Random(int(cls_id) + 17)
    return [rng.randint(0, 255) for _ in range(3)]


def _parse_bbox_file(bbox_path, force_normalized=None):
    # 解析 bbox 文本文件, 返回 (frame_id -> list of (cls_id, x1, y1, x2, y2, conf), normalized_flag)
    bboxes_per_frame = defaultdict(list)
    max_frame_id = -1
    header_normalized = None
    with open(bbox_path, 'r', encoding='utf-8') as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            if s.startswith('#'):
                low = s.lower()
                if 'normalized:' in low:
                    tail = low.split('normalized:', 1)[1].strip()
                    token = tail.split()[0] if tail.split() else ''
                    header_normalized = (token in ('1', 'true', 'yes', 'y'))
                continue
            parts = s.split(',')
            if len(parts) != 7:
                continue
            try:
                frame_id = int(float(parts[0]))
                cls_id = int(float(parts[1]))
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                conf = float(parts[6])
            except ValueError:
                continue
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h
            bboxes_per_frame[frame_id].append((cls_id, x1, y1, x2, y2, conf))
            if frame_id > max_frame_id:
                max_frame_id = frame_id

    if force_normalized is not None:
        normalized_flag = bool(force_normalized)
    elif header_normalized is not None:
        normalized_flag = header_normalized
    else:
        normalized_flag = False

    return bboxes_per_frame, max_frame_id, normalized_flag


def _draw_boxes(im0, boxes, names, color_map, conf_threshold, label_fmt):
    # boxes: list of tuples (cls_id, x1, y1, x2, y2, conf) -- 坐标已在原画面分辨率
    for cls_id, x1, y1, x2, y2, conf in boxes:
        if conf_threshold > 0 and conf < conf_threshold:
            continue
        c1 = (int(round(x1)), int(round(y1)))
        c2 = (int(round(x2)), int(round(y2)))
        color = color_map.get(cls_id)
        if color is None:
            color = _stable_color(cls_id)
            color_map[cls_id] = color
        cv2.rectangle(im0, c1, c2, color, thickness=_DEFAULT_LINE_THICKNESS, lineType=cv2.LINE_AA)

        if not label_fmt:
            continue

        if callable(names):
            cls_text = names(cls_id)
        else:
            cls_text = names.get(cls_id, str(cls_id))

        if label_fmt == 'conf':
            label = f'{conf:.2f}'
        elif label_fmt == 'cls_conf':
            label = f'{cls_text} {conf:.2f}'
        elif label_fmt == 'cls':
            label = f'{cls_text}'
        else:
            label = f'{conf:.2f}'

        tl = _DEFAULT_LINE_THICKNESS
        tf = max(tl - 1, 1)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, tl / 3, tf)
        bg_c2 = (c1[0] + tw, c1[1] - th - 3)
        overlay = im0.copy()
        cv2.rectangle(overlay, c1, bg_c2, color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, _DEFAULT_LABEL_BG_ALPHA, im0, 1 - _DEFAULT_LABEL_BG_ALPHA, 0, dst=im0)
        cv2.putText(im0, label, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, tl / 3,
                    (225, 255, 255), thickness=tf, lineType=cv2.LINE_AA)


def _resolve_bbox_path(args):
    if args.bbox:
        return args.bbox

    video_name = os.path.splitext(os.path.basename(args.source))[0]
    candidate_dir = args.video_dir or os.path.dirname(os.path.abspath(args.source))
    candidate = os.path.join(candidate_dir, f'{video_name}_bbox.txt')
    if os.path.exists(candidate):
        return candidate

    candidate = os.path.join(args.save_dir, f'{video_name}_bbox.txt')
    if os.path.exists(candidate):
        return candidate

    return os.path.join(args.save_dir, f'{video_name}_bbox.txt')


def _resolve_class_names(args):
    names = {}

    if not args.names:
        return names

    if ',' in args.names or ':' in args.names:
        for chunk in args.names.split(','):
            chunk = chunk.strip()
            if not chunk:
                continue
            if ':' not in chunk:
                continue
            cls_str, label = chunk.split(':', 1)
            try:
                names[int(float(cls_str))] = label.strip()
            except ValueError:
                continue
    else:
        names[0] = args.names.strip()

    return names


def synthesize():
    source = opt.source
    save_dir = opt.save_dir
    os.makedirs(save_dir, exist_ok=True)

    bbox_path = _resolve_bbox_path(opt)
    if not os.path.exists(bbox_path):
        raise FileNotFoundError(
            f'BBox file not found: {bbox_path}. '
            f'Pass --bbox <path> or run detect_video.py with --bbox_output first.'
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

    out_fps = opt.output_fps if opt.output_fps and opt.output_fps > 0 else src_fps

    output_scale = max(float(opt.output_scale), 0.05)
    out_width = make_even(int(width * output_scale))
    out_height = make_even(int(height * output_scale))

    video_name = os.path.splitext(os.path.basename(source))[0]
    save_path = os.path.join(save_dir, f'{video_name}_synthesized.mp4')

    writer, chosen_codec, codec_candidates = create_video_writer(save_path, out_fps, out_width, out_height, opt.codec)
    if writer is None:
        cap.release()
        raise RuntimeError(f'Cannot open video writer: {save_path}')

    fallback_tag = ' (fallback applied)' if str(opt.codec).lower() == 'auto' and chosen_codec != codec_candidates[0] else ''
    print(
        f'Requested codec: {opt.codec} | Selected codec: {chosen_codec}{fallback_tag} '
        f'| Tried: {" -> ".join(codec_candidates)}'
    )
    print(f'Synthetic video output: {out_width}x{out_height} @ {out_fps:.2f} fps')

    sample_fps = max(opt.sample_fps, 0.01)
    frame_interval = max(int(round(src_fps / sample_fps)), 1)

    force_normalized = opt.bbox_normalized if opt.bbox_normalized is not None else None
    bboxes_per_frame, max_bbox_frame_id, bbox_normalized = _parse_bbox_file(bbox_path, force_normalized=force_normalized)
    bbox_count = sum(len(v) for v in bboxes_per_frame.values())
    print(
        f'BBox file: {bbox_path} | Loaded {bbox_count} boxes across '
        f'{len(bboxes_per_frame)} sampled frames '
        f'(frame_interval={frame_interval}, source_fps={src_fps:.2f}, '
        f'normalized={bbox_normalized})'
    )
    if max_bbox_frame_id >= 0:
        print(f'  bbox max frame_id: {max_bbox_frame_id}')

    if total_frames > 0 and max_bbox_frame_id > total_frames + frame_interval:
        print(
            f'  WARNING: bbox max frame_id ({max_bbox_frame_id}) exceeds video frames ({total_frames}). '
            f'Check whether --sample_fps matches the value used in detect_video.py.'
        )

    names_map = _resolve_class_names(opt)
    if names_map:
        print(f'Class names: {names_map}')

    color_map = {}
    frame_idx = 0
    written = 0
    drawn_boxes = 0
    held_boxes = []
    last_sampled_frame_id = None
    start_time = time.time()

    try:
        while True:
            ok, im0 = cap.read()
            if not ok:
                break

            is_sampled_frame = (frame_idx % frame_interval == 0)
            if is_sampled_frame:
                raw_boxes = bboxes_per_frame.get(frame_idx, [])
                if bbox_normalized:
                    boxes = [
                        (cls_id, x1 * width, y1 * height, x2 * width, y2 * height, conf)
                        for (cls_id, x1, y1, x2, y2, conf) in raw_boxes
                    ]
                else:
                    boxes = raw_boxes
                held_boxes = boxes
                last_sampled_frame_id = frame_idx
            else:
                if opt.non_sampled_strategy == 'passthrough':
                    boxes = []
                elif opt.non_sampled_strategy == 'hold_last':
                    boxes = held_boxes
                else:
                    boxes = []

            if boxes:
                _draw_boxes(im0, boxes, names_map, color_map, opt.conf_thres, opt.label)
                if is_sampled_frame:
                    drawn_boxes += len(boxes)

            if opt.show_frame_id:
                cv2.putText(
                    im0,
                    f'frame_id={frame_idx}',
                    (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                if last_sampled_frame_id is not None and not is_sampled_frame:
                    cv2.putText(
                        im0,
                        f'(bbox from frame_id={last_sampled_frame_id})',
                        (12, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 200, 255),
                        2,
                        cv2.LINE_AA,
                    )

            if out_width != width or out_height != height:
                out_frame = cv2.resize(im0, (out_width, out_height), interpolation=cv2.INTER_AREA)
            else:
                out_frame = im0

            writer.write(out_frame)
            written += 1

            if written % 10 == 0 or (total_frames > 0 and written == total_frames):
                elapsed = max(time.time() - start_time, 1e-6)
                if total_frames > 0:
                    processed_pct = min(written / total_frames * 100.0, 100.0)
                    remaining_pct = max(100.0 - processed_pct, 0.0)
                    progress_info = f' | Progress: {processed_pct:.2f}% | Remaining: {remaining_pct:.2f}%'
                else:
                    progress_info = ''
                print(
                    f'\rWritten frames: {written} | Avg FPS: {written / elapsed:.2f}{progress_info}',
                    end='',
                    flush=True,
                )

            frame_idx += 1

    except KeyboardInterrupt:
        print('\nInterrupted by Ctrl+C. Saving current output...')
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    total_elapsed = max(time.time() - start_time, 1e-6)
    if total_frames > 0:
        final_pct = min(written / total_frames * 100.0, 100.0)
        final_remaining = max(100.0 - final_pct, 0.0)
        progress_info = f' Progress: {final_pct:.2f}% | Remaining: {final_remaining:.2f}%.'
    else:
        progress_info = ''
    print(
        f'\nDone. Synthetic video saved to: {save_path}. '
        f'Written {written} frames in {total_elapsed:.2f}s '
        f'(drew boxes on {drawn_boxes} sampled detections).{progress_info}'
    )
    print(f'Final selected codec: {chosen_codec}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='原始视频路径或流地址 / Original video path or stream URL')
    parser.add_argument('--bbox', type=str, default=None,
                        help='bbox 文本路径, 缺省按 <video_name>_bbox.txt 解析 / Optional bbox txt path; auto-resolved if omitted')
    parser.add_argument('--video_dir', type=str, default=None,
                        help='寻找 bbox 时优先使用的目录 / Optional directory used to auto-resolve bbox file')
    parser.add_argument('--save_dir', type=str, default='data/result', help='结果视频保存目录 / Output directory')
    parser.add_argument('--sample_fps', type=float, default=25.0,
                        help='必须与 detect_video.py 中 --sample_fps 取值一致 / Must match the value used in detect_video.py')
    parser.add_argument('--codec', type=str, default='auto', help='Video codec: auto/avc1/H264/mp4v/XVID')
    parser.add_argument('--output_scale', type=float, default=1.0,
                        help='输出分辨率缩放(0-1以减小文件体积) / Output resolution scale in (0, 1]')
    parser.add_argument('--output_fps', type=float, default=0.0,
                        help='输出帧率, 0 表示沿用源视频 fps / Output fps; 0 means follow source fps')
    parser.add_argument('--conf_thres', type=float, default=0.0,
                        help='二次过滤阈值, 默认 0 即复用 bbox 文件全部 conf / Secondary filter on confidence; 0 keeps all')
    parser.add_argument('--label', type=str, default='cls_conf',
                        choices=['none', 'conf', 'cls', 'cls_conf'],
                        help='bbox 标签内容 / Label content rendered above each box')
    parser.add_argument('--names', type=str, default=None,
                        help='类别名映射, 例如 "0:head,1:person" 或单值 "head" / Class name mapping, e.g. "0:head,1:person" or "head"')
    parser.add_argument('--non_sampled_strategy', type=str, default='hold_last',
                        choices=['hold_last', 'passthrough'],
                        help='非采样帧处理策略: hold_last=沿用上一采样帧 bbox, passthrough=原画面透传 / How to handle frames between two sampled frames')
    parser.add_argument('--show_frame_id', action='store_true', help='在画面上叠加 frame_id 水印 / Overlay frame_id watermark')
    parser.add_argument('--bbox_normalized', type=str, default=None,
                        choices=['true', 'false'],
                        help='强制指定 bbox 文件是否归一化 (true/false); 不传则按 header `# normalized:` 自动识别 / Force bbox normalized flag; auto-detect from header if omitted')

    opt = parser.parse_args()

    if opt.bbox_normalized is not None:
        opt.bbox_normalized = (opt.bbox_normalized == 'true')
    synthesize()
