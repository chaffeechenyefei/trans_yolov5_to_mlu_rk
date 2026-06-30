import argparse
import os
import time
import random
import math
from collections import deque

import cv2
import numpy as np
import torch

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device


def _make_even(v):
    return max((int(v) // 2) * 2, 2)


def _create_video_writer(save_path, fps, width, height, codec):
    # 编码器自动回退，优先高压缩编码以减小文件体积 / Auto-fallback codecs, prioritize high-compression encoders for smaller files
    if str(codec).lower() == 'auto':
        codec_candidates = ['avc1', 'H264', 'mp4v', 'XVID']
    else:
        codec_candidates = [codec]

    for c in codec_candidates:
        writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*c), fps, (width, height))
        if writer.isOpened():
            # 返回最终命中的编码器和候选列表 / Return final matched codec and candidate list
            return writer, c, codec_candidates
        writer.release()

    return None, None, codec_candidates


def _parse_img_size(img_size, stride):
    if isinstance(img_size, int):
        s = check_img_size(img_size, s=stride)
        return [s, s]

    if len(img_size) == 1:
        s = check_img_size(img_size[0], s=stride)
        return [s, s]

    h = check_img_size(img_size[0], s=stride)
    w = check_img_size(img_size[1], s=stride)
    return [h, w]


def _preprocess_frame(im0, imgsz, stride, device, half):
    img = letterbox(im0, new_shape=imgsz, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img


def _build_frame_heatmap(shape_hw, det):
    # 将当前帧bbox转换成热度图贡献，数量与置信度越高热度越高 / Convert current-frame bboxes to heat contribution map; more boxes and higher confidence produce stronger heat
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


def _update_temporal_heatmap(frame_heat, current_sec, second_bins, heat_sum, heat_time_sum, decay_seconds):
    # 以“秒”为粒度累计热度并维护窗口，降低内存占用 / Aggregate heat per second and keep a sliding window to reduce memory usage
    current_bin_sec = int(current_sec)
    if second_bins and second_bins[-1][0] == current_bin_sec:
        # second_bins元素是tuple，不能重绑元素，但可原地更新其中ndarray / tuple items are immutable, but inner ndarray can be updated in-place
        second_bins[-1][1][:] += frame_heat
    else:
        second_bins.append((current_bin_sec, frame_heat.copy()))

    heat_sum += frame_heat
    heat_time_sum += frame_heat * current_bin_sec

    # 清理超过衰减窗口的历史数据，60s前权重为0 / Drop stale history outside decay window where weight should be zero
    while second_bins and (current_sec - second_bins[0][0]) > decay_seconds:
        sec_old, heat_old = second_bins.popleft()
        heat_sum -= heat_old
        heat_time_sum -= heat_old * sec_old

    # 线性时间衰减: weight=max(0,1-age/decay_seconds) / Linear temporal decay: weight=max(0,1-age/decay_seconds)
    weighted_heat = heat_sum * (1.0 - current_sec / decay_seconds) + heat_time_sum / decay_seconds
    np.maximum(weighted_heat, 0.0, out=weighted_heat)
    return weighted_heat


def _overlay_heatmap(im0, weighted_heat, alpha):
    # 将加权热力图映射成伪彩色并与原图alpha叠加 / Convert weighted heatmap to color map and alpha-blend with original frame
    heat_max = float(weighted_heat.max())
    if heat_max <= 1e-6:
        return im0

    heat_norm = np.clip(weighted_heat / heat_max, 0.0, 1.0)
    heat_u8 = (heat_norm * 255.0).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    return cv2.addWeighted(im0, 1.0 - alpha, heat_color, alpha, 0.0)


def detect_video():
    source = opt.source
    save_dir = opt.save_dir
    os.makedirs(save_dir, exist_ok=True)

    device = select_device(opt.device)
    half = device.type != 'cpu'

    model = attempt_load(opt.weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = _parse_img_size(opt.img_size, stride)

    if half:
        model.half()

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz[0], imgsz[1]).to(device).type_as(next(model.parameters())))

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
    sampled_total_frames = int(math.ceil(total_frames / frame_interval)) if total_frames > 0 else 0

    # 输出缩放可显著降低码率和文件体积 / Output scaling can significantly reduce bitrate and file size
    output_scale = max(float(opt.output_scale), 0.05)
    out_width = _make_even(width * output_scale)
    out_height = _make_even(height * output_scale)

    video_name = os.path.splitext(os.path.basename(source))[0]
    save_path = os.path.join(save_dir, f'{video_name}_detect.mp4')

    # bbox-only模式: 仅写bbox文本文件, 跳过视频写入以提升吞吐 / bbox-only mode: write bbox text file only, skip video writing for higher throughput
    bbox_only = bool(opt.bbox_output)
    bbox_path = os.path.join(save_dir, f'{video_name}_bbox.txt') if bbox_only else None
    # 文本写入句柄在finally中统一关闭 / File handle is closed in finally block
    bbox_fp = open(bbox_path, 'w', encoding='utf-8') if bbox_only else None
    if bbox_only:
        # 写入表头, 便于下游解析消费 / Write header for downstream parsing
        bbox_fp.write('# frame_id, cls_id, x, y, w, h, conf\n')
        print(f'BBox-only output enabled. Video writer is disabled. BBox file: {bbox_path}')

    if bbox_only:
        # bbox-only不需要视频编码器 / No video codec needed in bbox-only mode
        writer = None
        chosen_codec = None
        codec_candidates = []
    else:
        writer, chosen_codec, codec_candidates = _create_video_writer(save_path, out_fps, out_width, out_height, opt.codec)

        if writer is None:
            cap.release()
            raise RuntimeError(f'Cannot open video writer: {save_path}')

        # 明确输出用户请求编码器与实际编码器，便于定位是否发生回退 / Print requested and actual codec to confirm fallback behavior
        fallback_tag = ' (fallback applied)' if str(opt.codec).lower() == 'auto' and chosen_codec != codec_candidates[0] else ''
        print(
            f'Requested codec: {opt.codec} | Selected codec: {chosen_codec}{fallback_tag} '
            f'| Tried: {" -> ".join(codec_candidates)}'
        )
        print(f'Video writer output: {out_width}x{out_height} @ {out_fps:.2f} fps')
    heatmap_alpha = min(max(float(opt.heatmap_alpha), 0.0), 1.0)
    if opt.enable_heatmap:
        # 输出热力图参数，便于回溯实验配置 / Print heatmap settings for reproducible experiments
        print(
            f'HeatMap enabled: decay={max(float(opt.heat_decay_seconds), 1.0):.1f}s, '
            f'alpha={heatmap_alpha:.2f}'
        )

    frame_idx = 0
    processed = 0
    infer_fps_ema = None
    start_time = time.time()

    second_bins = deque()
    # 分别维护Σheat与Σ(heat*time)，用于O(1)计算衰减热力图 / Keep Σheat and Σ(heat*time) for O(1) temporal decay computation
    heat_sum = np.zeros((height, width), dtype=np.float32)
    heat_time_sum = np.zeros((height, width), dtype=np.float32)

    try:
        with torch.no_grad():
            while True:
                ok, im0 = cap.read()
                if not ok:
                    break

                if frame_idx % frame_interval != 0:
                    frame_idx += 1
                    continue

                t0 = time.time()

                img = _preprocess_frame(im0, imgsz, stride, device, half)
                pred = model(img)[0]
                pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres)[0]

                if len(pred):
                    pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], im0.shape).round()
                    if bbox_only:
                        # bbox-only模式: 输出xywh到文本文件, 不画bbox不写视频 / bbox-only mode: write xywh to text file, skip drawing and video writing
                        for *xyxy, conf, cls in pred:
                            # xyxy顺序: x1,y1,x2,y2; 转成tight top-left + w,h / xyxy order: x1,y1,x2,y2; convert to top-left + w,h
                            x1 = float(xyxy[0])
                            y1 = float(xyxy[1])
                            x2 = float(xyxy[2])
                            y2 = float(xyxy[3])
                            w = x2 - x1
                            h = y2 - y1
                            # frame_id 对应当前采样帧在原视频中的索引 / frame_id is the index of the current sampled frame in the source video
                            bbox_fp.write(
                                f'{frame_idx}, {int(cls)}, {x1:.2f}, {y1:.2f}, {w:.2f}, {h:.2f}, {float(conf):.6f}\n'
                            )
                    else:
                        for *xyxy, conf, cls in reversed(pred):
                            c = int(cls)
                            label = f'{names[c]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=2)
                elif bbox_only:
                    pass  # 该帧无检测结果, 不写任何bbox行 / No detection for this frame, write nothing

                if bbox_only:
                    # bbox-only模式: 仅记录推理fps, 不做任何画面合成与视频写入 / bbox-only mode: only track inference fps, skip overlay/resize/write
                    infer_time = max(time.time() - t0, 1e-6)
                    infer_fps = 1.0 / infer_time
                    if infer_fps_ema is None:
                        infer_fps_ema = infer_fps
                    else:
                        infer_fps_ema = infer_fps_ema * 0.9 + infer_fps * 0.1

                    processed += 1

                    elapsed = max(time.time() - start_time, 1e-6)
                    avg_proc_fps = processed / elapsed
                    if sampled_total_frames > 0:
                        processed_pct = min(processed / sampled_total_frames * 100.0, 100.0)
                        remaining_pct = max(100.0 - processed_pct, 0.0)
                        progress_info = f' | Progress: {processed_pct:.2f}% | Remaining: {remaining_pct:.2f}%'
                    else:
                        progress_info = ''
                    print(
                        f'\rProcessed frames: {processed} | Avg FPS: {avg_proc_fps:.2f} | Current FPS: {infer_fps_ema:.2f}{progress_info}',
                        end='',
                        flush=True,
                    )

                    frame_idx += 1
                    continue

                if opt.enable_heatmap:
                    # 使用视频时间而不是墙钟时间，确保离线处理与实时播放权重一致 / Use video timeline instead of wall-clock time for stable offline/online behavior
                    current_sec = frame_idx / src_fps
                    frame_heat = _build_frame_heatmap((height, width), pred)
                    weighted_heat = _update_temporal_heatmap(
                        frame_heat=frame_heat,
                        current_sec=current_sec,
                        second_bins=second_bins,
                        heat_sum=heat_sum,
                        heat_time_sum=heat_time_sum,
                        decay_seconds=max(float(opt.heat_decay_seconds), 1.0),
                    )
                    im0 = _overlay_heatmap(im0, weighted_heat, alpha=heatmap_alpha)

                infer_time = max(time.time() - t0, 1e-6)
                infer_fps = 1.0 / infer_time
                if infer_fps_ema is None:
                    infer_fps_ema = infer_fps
                else:
                    infer_fps_ema = infer_fps_ema * 0.9 + infer_fps * 0.1

                cv2.putText(
                    im0,
                    f'Infer FPS: {infer_fps_ema:.2f} | Sample FPS: {sample_fps:.2f}',
                    (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                if out_width != width or out_height != height:
                    out_frame = cv2.resize(im0, (out_width, out_height), interpolation=cv2.INTER_AREA)
                else:
                    out_frame = im0

                writer.write(out_frame)
                processed += 1

                elapsed = max(time.time() - start_time, 1e-6)
                avg_proc_fps = processed / elapsed
                if sampled_total_frames > 0:
                    processed_pct = min(processed / sampled_total_frames * 100.0, 100.0)
                    remaining_pct = max(100.0 - processed_pct, 0.0)
                    progress_info = f' | Progress: {processed_pct:.2f}% | Remaining: {remaining_pct:.2f}%'
                else:
                    progress_info = ''
                print(
                    f'\rProcessed frames: {processed} | Avg FPS: {avg_proc_fps:.2f} | Current FPS: {infer_fps_ema:.2f}{progress_info}',
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
        if bbox_fp is not None:
            # 显式flush+close, 保证Ctrl+C中断后bbox文件落盘 / Explicit flush+close to ensure bbox file is persisted after Ctrl+C
            bbox_fp.flush()
            bbox_fp.close()
        cv2.destroyAllWindows()

    total_elapsed = max(time.time() - start_time, 1e-6)
    if sampled_total_frames > 0:
        final_processed_pct = min(processed / sampled_total_frames * 100.0, 100.0)
        final_remaining_pct = max(100.0 - final_processed_pct, 0.0)
        final_progress_info = f' Progress: {final_processed_pct:.2f}% | Remaining: {final_remaining_pct:.2f}%.'
    else:
        final_progress_info = ''

    if bbox_only:
        print(
            f'\nDone. BBox file saved to: {bbox_path}. '
            f'Processed {processed} frames in {total_elapsed:.2f}s.{final_progress_info}'
        )
    else:
        print(
            f'\nDone. Output saved to: {save_path}. '
            f'Processed {processed} frames in {total_elapsed:.2f}s.{final_progress_info}'
        )
        # 结束再次输出最终编码器，避免中途日志被覆盖 / Print selected codec again at end in case progress logs overwrite earlier output
        print(f'Final selected codec: {chosen_codec}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov5s.pt')
    parser.add_argument('--source', type=str, required=True, help='Input video path or stream URL')
    parser.add_argument('--save_dir', type=str, default='data/result', help='Directory to save output video')
    parser.add_argument('--img_size', type=int, nargs='+', default=[416, 736])
    parser.add_argument('--conf_thres', type=float, default=0.2)
    parser.add_argument('--iou_thres', type=float, default=0.3)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--sample_fps', type=float, default=1.0, help='Sample frame rate for detection')
    parser.add_argument('--codec', type=str, default='auto', help='Video codec: auto/avc1/H264/mp4v/XVID')
    parser.add_argument('--output_scale', type=float, default=1.0, help='Output resolution scale (0-1 to reduce file size)')
    parser.add_argument('--enable_heatmap', action='store_true', help='Overlay weighted heatmap on output video')
    parser.add_argument('--heat_decay_seconds', type=float, default=60.0, help='Heatmap decay window in seconds')
    parser.add_argument('--heatmap_alpha', type=float, default=0.35, help='Heatmap overlay alpha in [0, 1]')
    parser.add_argument('--bbox_output', action='store_true', help='Only output bbox text file (frame_id, cls_id, x, y, w, h, conf); skip video writing')

    opt = parser.parse_args()
    detect_video()
