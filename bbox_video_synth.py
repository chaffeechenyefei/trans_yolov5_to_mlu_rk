"""
bbox_video_synth.py

将 detect_video.py 通过 --bbox_output 产出的 bbox 文本文件
与原始视频合成带 bbox 的结果视频, 便于直接查看结果。
不依赖 torch / 模型权重, 纯 OpenCV 写入, 可在轻量环境下运行。

---
Synthesize a bbox-annotated video by overlaying the bbox text file
(produced by detect_video.py --bbox_output) onto the original source video.
No torch / model weights required: pure OpenCV based output, runs in a lean env.
"""

import argparse
import math
import os
import random
import time
from collections import defaultdict

import cv2

try:
    # 复用 detect_video.py 中的工具函数, 保持画面编码行为一致 / Reuse helpers from detect_video.py for consistent video encoding
    from detect_video import _create_video_writer, _make_even
except Exception:  # pragma: no cover - 兜底实现, 避免 detect_video.py 缺失时直接断链 / fallback so this script still runs without detect_video.py
    def _make_even(v):
        return max((int(v) // 2) * 2, 2)

    def _create_video_writer(save_path, fps, width, height, codec):
        if str(codec).lower() == 'auto':
            candidates = ['avc1', 'H264', 'mp4v', 'XVID']
        else:
            candidates = [codec]
        for c in candidates:
            w = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*c), fps, (width, height))
            if w.isOpened():
                return w, c, candidates
            w.release()
        return None, None, candidates


# 默认绘制颜色与线宽, 与 detect.py 风格保持一致 / Default draw style aligned with detect.py
_DEFAULT_LINE_THICKNESS = 2
_DEFAULT_LABEL_BG_ALPHA = 0.55  # label 背景填充透明度 / label background alpha


def _stable_color(cls_id):
    # 同一类别固定一种颜色, 便于视觉对比 / Stable per-class color for visual consistency
    rng = random.Random(int(cls_id) + 17)
    return [rng.randint(0, 255) for _ in range(3)]


def _parse_bbox_file(bbox_path, force_normalized=None):
    # 解析 bbox 文本文件, 返回 (frame_id -> list of (cls_id, x1, y1, x2, y2, conf), normalized_flag)
    # 第 7 个字段为 conf, 第 3~6 个字段是 (x, y, w, h), 表示 top-left 坐标 + 宽高
    #
    # 坐标尺度识别优先级:
    #   1) force_normalized: 调用方显式指定 (True/False), 覆盖 header
    #   2) header `# normalized: true|false` 自动识别 (detect_video.py 写入)
    #   3) 默认 False (像素坐标, 向后兼容旧文件)
    #
    # Parse bbox text file into (frame_id -> list of (cls_id, x1, y1, x2, y2, conf), normalized_flag)
    # The 7th column is conf; columns 3..6 are (x, y, w, h) which is top-left + width/height.
    # Coordinate scale detection priority:
    #   1) force_normalized overrides header
    #   2) header `# normalized: true|false` auto-detect
    #   3) defaults to False (pixel coords, backward compat)
    bboxes_per_frame = defaultdict(list)
    max_frame_id = -1
    header_normalized = None
    with open(bbox_path, 'r', encoding='utf-8') as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            if s.startswith('#'):
                # 解析 header 中的归一化标识, 不区分大小写 / Parse normalization flag from header, case-insensitive
                low = s.lower()
                if 'normalized:' in low:
                    tail = low.split('normalized:', 1)[1].strip()
                    # 取首个 token, 例如 "true (...)" -> "true" / Take first token, e.g. "true (...)" -> "true"
                    token = tail.split()[0] if tail.split() else ''
                    header_normalized = (token in ('1', 'true', 'yes', 'y'))
                continue
            parts = s.split(',')
            if len(parts) != 7:
                # 字段数不足直接丢, 不影响其它行 / Skip malformed line without aborting the whole file
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
    # names: function or dict to map cls_id -> label text
    # boxes are already in original video resolution
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

        # 仅当用户显式选择带 label 时才绘制文字背景块 / Only render label block when explicitly requested
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
        # 与 plot_one_box 视觉风格保持一致: 顶部填充背景框 / Match plot_one_box style: filled rect above the box
        bg_c2 = (c1[0] + tw, c1[1] - th - 3)
        overlay = im0.copy()
        cv2.rectangle(overlay, c1, bg_c2, color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, _DEFAULT_LABEL_BG_ALPHA, im0, 1 - _DEFAULT_LABEL_BG_ALPHA, 0, dst=im0)
        cv2.putText(im0, label, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, tl / 3,
                    (225, 255, 255), thickness=tf, lineType=cv2.LINE_AA)


def _resolve_bbox_path(args):
    # 优先级: 命令行显式 -> 同源目录 -> save_dir -> 与 video 同名约定 / Resolution priority order
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

    # 找不到则返回第一个候选, 让后面错误信息更具语义 / Return a path so downstream error messages are meaningful
    return os.path.join(args.save_dir, f'{video_name}_bbox.txt')


def _resolve_class_names(args):
    # 解析类名映射: 来自 --names "0:head,1:person" 或者直接 --names head / Build cls_id -> name mapping
    names = {}

    if not args.names:
        return names

    if ',' in args.names or ':' in args.names:
        # 完整映射形式 / Full mapping form
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
        # 单类名 / Single class name
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

    # 输出帧率与原始视频一致, 合成时保持时间轴不变 / Match source FPS so the synthetic video keeps original timeline
    out_fps = opt.output_fps if opt.output_fps and opt.output_fps > 0 else src_fps

    output_scale = max(float(opt.output_scale), 0.05)
    out_width = _make_even(width * output_scale)
    out_height = _make_even(height * output_scale)

    video_name = os.path.splitext(os.path.basename(source))[0]
    save_path = os.path.join(save_dir, f'{video_name}_synthesized.mp4')

    writer, chosen_codec, codec_candidates = _create_video_writer(save_path, out_fps, out_width, out_height, opt.codec)
    if writer is None:
        cap.release()
        raise RuntimeError(f'Cannot open video writer: {save_path}')

    fallback_tag = ' (fallback applied)' if str(opt.codec).lower() == 'auto' and chosen_codec != codec_candidates[0] else ''
    print(
        f'Requested codec: {opt.codec} | Selected codec: {chosen_codec}{fallback_tag} '
        f'| Tried: {" -> ".join(codec_candidates)}'
    )
    print(f'Synthetic video output: {out_width}x{out_height} @ {out_fps:.2f} fps')

    # 必须与 detect_video.py 中 --sample_fps 一致, 否则 bbox 投影到错误帧 / Must match detect_video.py --sample_fps or boxes will land on wrong frames
    sample_fps = max(opt.sample_fps, 0.01)
    frame_interval = max(int(round(src_fps / sample_fps)), 1)

    # 调用方通过 CLI --bbox_normalized 显式覆盖; 为 None 时依据 header 自动识别 / CLI --bbox_normalized overrides header; None means auto-detect from header
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

    # 校验: bbox 收录的最大 frame_id + 帧间隔 不应超过原视频总帧数过多 / Sanity check: bbox coverage vs video length
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
    held_boxes = []  # hold_last 策略: 缓存上一个采样帧的 bbox / Cache for hold_last strategy
    last_sampled_frame_id = None
    start_time = time.time()

    try:
        while True:
            ok, im0 = cap.read()
            if not ok:
                break

            is_sampled_frame = (frame_idx % frame_interval == 0)
            if is_sampled_frame:
                # 当前帧就是 detect_video.py 真正推理过的帧, 直接使用其 bbox / This frame was actually inferred
                raw_boxes = bboxes_per_frame.get(frame_idx, [])
                if bbox_normalized:
                    # 反归一化: x1,x2 乘 im_w; y1,y2 乘 im_h, 还原为像素坐标供画框 / Denormalize back to pixels: x1,x2 * im_w; y1,y2 * im_h
                    boxes = [
                        (cls_id, x1 * width, y1 * height, x2 * width, y2 * height, conf)
                        for (cls_id, x1, y1, x2, y2, conf) in raw_boxes
                    ]
                else:
                    boxes = raw_boxes
                held_boxes = boxes
                last_sampled_frame_id = frame_idx
            else:
                # 非采样帧没有对应 bbox, 按策略处理 / Non-sampled frame: choose a strategy
                if opt.non_sampled_strategy == 'passthrough':
                    boxes = []
                elif opt.non_sampled_strategy == 'hold_last':
                    boxes = held_boxes
                elif opt.non_sampled_strategy == 'blank':
                    # 仅写入原始帧, 不绘制 bbox / Pass-through alias, kept for explicit semantic
                    boxes = []
                else:
                    boxes = []

            if boxes:
                _draw_boxes(im0, boxes, names_map, color_map, opt.conf_thres, opt.label)
                if is_sampled_frame:
                    drawn_boxes += len(boxes)

            # 帧号水印, 便于与 bbox 文件对照 / Frame id watermark, helpful when matching against bbox txt
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
                    # 标记 bbox 来源帧, 避免用户误以为该帧被推理过 / Indicate that the displayed boxes originate from another frame
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

    # 把字符串选项转换为三态: True / False / None (None=自动识别 header) / Convert string to tri-state
    if opt.bbox_normalized is not None:
        opt.bbox_normalized = (opt.bbox_normalized == 'true')
    synthesize()
