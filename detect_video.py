import argparse
import os
import time
import random
import math

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

    frame_idx = 0
    processed = 0
    infer_fps_ema = None
    start_time = time.time()

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
                    for *xyxy, conf, cls in reversed(pred):
                        c = int(cls)
                        label = f'{names[c]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=2)

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
        writer.release()
        cv2.destroyAllWindows()

    total_elapsed = max(time.time() - start_time, 1e-6)
    if sampled_total_frames > 0:
        final_processed_pct = min(processed / sampled_total_frames * 100.0, 100.0)
        final_remaining_pct = max(100.0 - final_processed_pct, 0.0)
        final_progress_info = f' Progress: {final_processed_pct:.2f}% | Remaining: {final_remaining_pct:.2f}%.'
    else:
        final_progress_info = ''
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

    opt = parser.parse_args()
    detect_video()
