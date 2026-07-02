"""
test_pt_vs_onnx.py

对比 PyTorch (.pt) 与 ONNX Runtime (.onnx) 推理输出的一致性.
Compare inference output consistency between PyTorch (.pt) and ONNX Runtime (.onnx).

Usage:
    python test_pt_vs_onnx.py \
      --pt_weights weights/yolov5s-people.pt \
      --onnx_weights weights/yolov5s-people_mode0.onnx \
      --img_size 736 416
"""

import argparse
import os
import sys
import time

# 确保能找到 models/ 和 utils/ 目录 / Ensure models/ and utils/ are importable
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE = os.path.dirname(_SCRIPT_DIR) if os.path.basename(_SCRIPT_DIR) == 'project_detection_and_heatmap' else _SCRIPT_DIR
if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)

import numpy as np
import cv2
import torch
import onnxruntime as ort

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device


def _parse_img_size(img_size, stride):
    """解析并校准输入尺寸 / Parse and calibrate input size to be stride-aligned."""
    if isinstance(img_size, int):
        s = check_img_size(img_size, s=stride)
        return [s, s]
    if len(img_size) == 1:
        s = check_img_size(img_size[0], s=stride)
        return [s, s]
    h = check_img_size(img_size[0], s=stride)
    w = check_img_size(img_size[1], s=stride)
    return [h, w]


def pt_inference(model, img_numpy: np.ndarray, device, half: bool):
    """PyTorch 推理, 返回 3 个原始输出张量 (与 export mode=0 一致).
    PyTorch inference, returns 3 raw output tensors (matching export mode=0)."""
    # 预处理: letterbox → HWC→CHW → normalize → add batch dim / Preprocess
    img = letterbox(img_numpy, new_shape=(416, 736), stride=32)[0]  # stride=32 for YOLOv5s
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)  # (1, 3, H, W)

    with torch.no_grad():
        # PT model 正常推理输出 / Normal PT model inference output
        pred = model(img)[0]  # (1, N, 5+nc)  after NMS-style raw output
    return pred, img.shape[2:]


def onnx_inference(session, img_numpy: np.ndarray, input_name: str):
    """ONNX Runtime 推理, 返回 model 原始输出列表.
    ONNX Runtime inference, returns raw model output list."""
    # 预处理: letterbox → HWC→CHW → normalize → add batch dim / Preprocess
    img = letterbox(img_numpy, new_shape=(416, 736), stride=32)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
    img = np.ascontiguousarray(img)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # (1, 3, H, W)

    # ONNX Runtime 推理 / ONNX Runtime inference
    outputs = session.run(None, {input_name: img})
    return outputs


def compare_raw_outputs(pt_output, onnx_outputs, rtol=1e-4, atol=1e-6):
    """对比 PT 与 ONNX 原始输出, 打印逐输出差异统计.
    Compare PT and ONNX raw outputs, print per-output diff stats.

    pt_output: list of 3 PyTorch tensors from export_mode=0.
    onnx_outputs: list of 3 numpy arrays from ONNX Runtime.
    """
    results = {}

    for i, (pt_tensor, onnx_arr) in enumerate(zip(pt_output, onnx_outputs)):
        pt_np = pt_tensor.cpu().numpy().astype(np.float64)
        onnx_arr = onnx_arr.astype(np.float64)
        name = ['xys', 'whs', 'confs'][i]

        abs_diff = np.abs(pt_np - onnx_arr)
        max_abs = float(abs_diff.max())
        mean_abs = float(abs_diff.mean())
        # 相对差异: 对有效值 (>1e-6 量级) 计算, 避免接近零的值放大相对差异
        # Relative diff: compute only on significant values to avoid near-zero inflation
        denom = np.maximum(np.abs(pt_np), np.abs(onnx_arr))
        significant = denom > 1e-4  # 仅考虑量级 > 1e-4 的值 / Only consider values with magnitude > 1e-4
        if significant.any():
            rel_diff = np.divide(abs_diff[significant], denom[significant])
            max_rel = float(rel_diff.max()) if len(rel_diff) > 0 else 0.0
            mean_rel = float(rel_diff.mean()) if len(rel_diff) > 0 else 0.0
            sig_frac = significant.mean()  # 有效值占比 / Fraction of significant values
        else:
            max_rel = 0.0
            mean_rel = 0.0
            sig_frac = 0.0

        results[name] = {
            'max_abs_diff': max_abs,
            'mean_abs_diff': mean_abs,
            'max_rel_diff': max_rel,
            'mean_rel_diff': mean_rel,
            'significant_fraction': sig_frac,
        }

    return results


def compare_export_mode(pt_model, onnx_session, img_numpy, device, half, input_name, onnx_input_shape):
    """以 export_mode=0 运行 PT 模型, 与 ONNX 输出对比.
    Run PT model in export_mode=0 and compare with ONNX outputs.

    onnx_input_shape: (H, W) from ONNX model input, e.g. (416, 736).
    """
    onnx_h, onnx_w = onnx_input_shape

    # 设置 PT 模型为 export mode 0 / Set PT model to export mode 0
    pt_model.model[-1].export = True
    pt_model.model[-1].export_mode = 0

    # PT 推理: 使用与 ONNX 完全相同的输入尺寸 / PT inference: use exact same input size as ONNX
    img = letterbox(img_numpy, new_shape=(onnx_h, onnx_w), stride=32, auto=False)[0]
    # 确保尺寸精确匹配 / Ensure exact size match
    if img.shape[0] != onnx_h or img.shape[1] != onnx_w:
        img = cv2.resize(img, (onnx_w, onnx_h), interpolation=cv2.INTER_LINEAR)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
    img = np.ascontiguousarray(img)
    img_tensor = torch.from_numpy(img).to(device)
    img_tensor = img_tensor.half() if half else img_tensor.float()
    img_tensor /= 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        pt_outputs = pt_model(img_tensor)  # list of 3 tensors: [xys, whs, confs]

    # ONNX 推理 / ONNX inference
    img_onnx = img.astype(np.float32) / 255.0
    img_onnx = np.expand_dims(img_onnx, axis=0)
    onnx_outputs = onnx_session.run(None, {input_name: img_onnx})  # list of 3 arrays

    # 对比 / Compare
    results = compare_raw_outputs(pt_outputs, onnx_outputs)

    # 恢复 PT 模型为非 export 模式, 并重置 training 状态
    # Restore PT model to non-export mode, and reset training state
    # (Detect.forward 中有 self.training |= self.export, 需同步重置)
    pt_model.model[-1].export = False
    pt_model.model[-1].training = False
    # 同时重置 training 属性 (nn.Module 基类属性) / Also reset base nn.Module training attr
    pt_model.model[-1].train(False)

    return results, (onnx_h, onnx_w)


def get_onnx_session(onnx_path: str):
    """创建 ONNX Runtime 会话 / Create ONNX Runtime session."""
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    print(f'ONNX session created:')
    print(f'  Input: {input_name}, shape={session.get_inputs()[0].shape}')
    for o in session.get_outputs():
        print(f'  Output: {o.name}, shape={o.shape}')
    return session, input_name, output_names


def main():
    parser = argparse.ArgumentParser(description='Compare PT vs ONNX inference outputs')
    parser.add_argument('--pt_weights', type=str, required=True,
                        help='Path to .pt weights file')
    parser.add_argument('--onnx_weights', type=str, required=True,
                        help='Path to .onnx weights file')
    parser.add_argument('--img_size', type=int, nargs='+', default=[736, 416],
                        help='Model input size (w h)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for PT inference')
    parser.add_argument('--test_image', type=str, default=None,
                        help='Optional: path to test image, random noise used if not provided')
    opt = parser.parse_args()

    # ---- 加载 PT 模型 / Load PT model ----
    print('=' * 60)
    print('Loading PyTorch model...')
    device = select_device(opt.device)
    half = device.type != 'cpu'
    pt_model = attempt_load(opt.pt_weights, map_location=device)
    stride = int(pt_model.stride.max())
    imgsz = _parse_img_size(opt.img_size, stride)
    if half:
        pt_model.half()
    # warm-up / Warm up
    if device.type != 'cpu':
        pt_model(torch.zeros(1, 3, imgsz[0], imgsz[1]).to(device).type_as(next(pt_model.parameters())))
    print(f'  PT model loaded. Stride={stride}, imgsz={imgsz}, names={pt_model.names}')

    # ---- 加载 ONNX 模型 / Load ONNX model ----
    print('-' * 60)
    print('Loading ONNX model...')
    onnx_session, input_name, output_names = get_onnx_session(opt.onnx_weights)

    # ---- 准备测试输入 / Prepare test input ----
    if opt.test_image and os.path.exists(opt.test_image):
        print('-' * 60)
        print(f'Using test image: {opt.test_image}')
        img_numpy = cv2.imread(opt.test_image)
        if img_numpy is None:
            raise FileNotFoundError(f'Cannot read image: {opt.test_image}')
        print(f'  Image shape: {img_numpy.shape}')
    else:
        # 使用随机图像 (模拟真实场景的像素分布) / Use random image (simulate real pixel distribution)
        print('-' * 60)
        print('No test image provided, using random noise (uint8, uniform 0-255)')
        rng = np.random.RandomState(42)  # 固定种子保证可复现 / Fixed seed for reproducibility
        img_numpy = rng.randint(0, 256, size=(544, 960, 3), dtype=np.uint8)
        print(f'  Random image shape: {img_numpy.shape}')

    # ---- 对比: export mode 0 / Compare in export mode 0 ----
    print('=' * 60)
    print('Comparing PT (export_mode=0) vs ONNX raw outputs...')

    # 从 ONNX session 获取精确输入尺寸 / Get exact input shape from ONNX session
    onnx_input_shape = onnx_session.get_inputs()[0].shape  # [1, 3, H, W]
    onnx_h, onnx_w = onnx_input_shape[2], onnx_input_shape[3]
    print(f'  ONNX input shape: H={onnx_h}, W={onnx_w}')

    t0 = time.time()
    results, (model_h, model_w) = compare_export_mode(
        pt_model, onnx_session, img_numpy, device, half, input_name,
        onnx_input_shape=(onnx_h, onnx_w),
    )
    elapsed = time.time() - t0
    print(f'  Comparison completed in {elapsed:.2f}s')

    # ---- 打印对比结果 / Print comparison results ----
    print('=' * 60)
    print('Per-output difference summary (PT export_mode=0 vs ONNX):')
    print(f'{"Output":<10} {"Max Abs Diff":<16} {"Mean Abs Diff":<16} {"Max Rel Diff":<16} {"Mean Rel Diff":<16} {"Sig Frac":<10}')
    print('-' * 90)
    all_pass = True
    for name in ['xys', 'whs', 'confs']:
        r = results[name]
        # 判定阈值: 绝对差 < 5e-3 视为通过 (FP32 精度, 考虑累积误差)
        # Threshold: absolute diff < 5e-3 → pass (FP32 precision with accumulated errors)
        pass_abs = r['max_abs_diff'] < 5e-3
        ok = pass_abs
        status = '✓ PASS' if ok else '✗ FAIL'
        if not ok:
            all_pass = False
        print(f'{name:<10} {r["max_abs_diff"]:<16.6e} {r["mean_abs_diff"]:<16.6e} '
              f'{r["max_rel_diff"]:<16.6e} {r["mean_rel_diff"]:<16.6e} '
              f'{r["significant_fraction"]:<10.1%}  {status}')

    # ---- 额外对比: NMS 后 bbox 一致性 / Extra: bbox consistency after NMS ----
    print('=' * 60)
    print('Comparing post-NMS bbox outputs (end-to-end)...')

    # PT: 正常推理 + NMS (使用与 ONNX 相同的输入尺寸) / PT: normal inference + NMS (same input size as ONNX)
    img = letterbox(img_numpy, new_shape=(model_h, model_w), stride=32, auto=False)[0]
    if img.shape[0] != model_h or img.shape[1] != model_w:
        img = cv2.resize(img, (model_w, model_h), interpolation=cv2.INTER_LINEAR)
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img_t = torch.from_numpy(img).to(device)
    img_t = img_t.half() if half else img_t.float()
    img_t /= 255.0
    if img_t.ndimension() == 3:
        img_t = img_t.unsqueeze(0)

    with torch.no_grad():
        pt_pred = pt_model(img_t)[0]
    pt_pred_nms = non_max_suppression(pt_pred, 0.5, 0.3)[0]  # conf_thres=0.5, iou_thres=0.3
    if pt_pred_nms is not None and len(pt_pred_nms):
        pt_pred_nms[:, :4] = scale_coords(img_t.shape[2:], pt_pred_nms[:, :4], img_numpy.shape).round()

    # ONNX: 推理 + decode + NumPy NMS / ONNX: inference + decode + NumPy NMS
    img_onnx = img.astype(np.float32) / 255.0
    img_onnx = np.expand_dims(img_onnx, axis=0)
    onnx_raw = onnx_session.run(None, {input_name: img_onnx})  # [xys, whs, confs]

    # 使用与 onnx_detect.py 相同的解码逻辑 / Use same decode logic as onnx_detect.py
    onnx_boxes = decode_and_nms_onnx(onnx_raw, img_numpy.shape, model_h, model_w,
                                     conf_thres=0.5, iou_thres=0.3)

    # 比较 bbox 数量和 IoU / Compare bbox count and IoU
    pt_count = len(pt_pred_nms) if pt_pred_nms is not None else 0
    onnx_count = len(onnx_boxes)
    print(f'  PT bbox count (after NMS): {pt_count}')
    print(f'  ONNX bbox count (after NMS): {onnx_count}')

    if pt_count > 0 and onnx_count > 0:
        # 计算 bbox 匹配率 (IoU >= 0.95 视为匹配) / Compute bbox match rate
        pt_boxes = pt_pred_nms[:, :4].cpu().numpy()
        matched_pt = set()
        matched_onnx = set()
        for pi, pb in enumerate(pt_boxes):
            best_iou = 0
            best_oi = -1
            for oi, ob in enumerate(onnx_boxes[:, :4]):
                iou = box_iou(pb, ob)
                if iou > best_iou:
                    best_iou = iou
                    best_oi = oi
            if best_iou >= 0.95:
                matched_pt.add(pi)
                matched_onnx.add(best_oi)

        pt_recall = len(matched_pt) / pt_count * 100 if pt_count > 0 else 0
        onnx_precision = len(matched_onnx) / onnx_count * 100 if onnx_count > 0 else 0
        print(f'  PT bbox matched (IoU≥0.95): {len(matched_pt)}/{pt_count} ({pt_recall:.1f}%)')
        print(f'  ONNX bbox matched (IoU≥0.95): {len(matched_onnx)}/{onnx_count} ({onnx_precision:.1f}%)')
    elif pt_count == 0 and onnx_count == 0:
        print(f'  Both outputs have zero bboxes → consistent ✓')

    # ---- 总结 / Summary ----
    print('=' * 60)
    if all_pass:
        print('✓ 整体结论: PT 与 ONNX 原始输出完全一致 (差异在浮点误差范围内).')
        print('  Overall: PT and ONNX raw outputs are consistent (within floating-point tolerance).')
    else:
        print('✗ 整体结论: PT 与 ONNX 原始输出存在超出浮点误差的差异, 需进一步排查.')
        print('  Overall: PT and ONNX raw outputs have differences beyond floating-point tolerance.')

    return 0 if all_pass else 1


# ---- NumPy 版 YOLO 解码 + NMS (与 onnx_detect.py 一致) ----
# NumPy version of YOLO decode + NMS (matching onnx_detect.py)


def decode_and_nms_onnx(outputs: list, im0_shape: tuple,
                        model_h: int, model_w: int,
                        conf_thres: float = 0.5, iou_thres: float = 0.3) -> np.ndarray:
    """将 ONNX export mode=0 的 3 个输出解码为 [x1, y1, x2, y2, conf] 格式并做 NMS.
    严格匹配 PyTorch non_max_suppression 的行为.
    Decode ONNX export mode=0 outputs matching PyTorch non_max_suppression behavior exactly.

    outputs: [xys, whs, confs]
      - xys: (1, 18837, 2) [cx, cy] in pixel coords (already decoded)
      - whs: (1, 18837, 2) [w, h] in pixel coords (already decoded)
      - confs: (1, 18837, nc+1) [obj_conf, cls_conf_0, ...] post-sigmoid
    im0_shape: (H, W, C) of original image
    model_h, model_w: model input size (H, W)
    Returns: (M, 5) array of [x1, y1, x2, y2, conf]
    """
    im_h, im_w = im0_shape[:2]

    xys = outputs[0][0]    # (18837, 2) [cx, cy] pixel
    whs = outputs[1][0]    # (18837, 2) [w, h] pixel
    confs = outputs[2][0]  # (18837, nc+1) [obj, cls_0, ...]

    obj_conf = confs[:, 0]     # (18837,) obj conf
    cls_conf = confs[:, 1:]    # (18837, nc) class conf

    # Step 1: 按 obj_conf 过滤候选 / Filter candidates by obj_conf (matching PT: xc = pred[..., 4] > conf_thres)
    candidate_mask = obj_conf > conf_thres
    if not candidate_mask.any():
        return np.zeros((0, 5), dtype=np.float32)

    xys_c = xys[candidate_mask]
    whs_c = whs[candidate_mask]
    obj_c = obj_conf[candidate_mask][:, None]   # (K, 1)
    cls_c = cls_conf[candidate_mask]             # (K, nc)

    # Step 2: conf = obj * cls (matching PT: x[:, 5:] *= x[:, 4:5])
    cls_c = cls_c * obj_c  # (K, nc)

    # Step 3: cxcywh → xyxy (matching PT: box = xywh2xyxy(x[:, :4]))
    boxes_xyxy = np.zeros((len(xys_c), 4), dtype=np.float32)
    boxes_xyxy[:, 0] = xys_c[:, 0] - whs_c[:, 0] / 2  # x1 = cx - w/2
    boxes_xyxy[:, 1] = xys_c[:, 1] - whs_c[:, 1] / 2  # y1 = cy - h/2
    boxes_xyxy[:, 2] = xys_c[:, 0] + whs_c[:, 0] / 2  # x2 = cx + w/2
    boxes_xyxy[:, 3] = xys_c[:, 1] + whs_c[:, 1] / 2  # y2 = cy + h/2

    # Step 4: 取每个候选框的最佳类别 / Take best class per box (matching PT: conf, j = x[:, 5:].max(1))
    best_conf = cls_c.max(axis=1)  # (K,)

    # Step 5: 按 combined conf 过滤 / Filter by combined conf (matching PT: [conf.view(-1) > conf_thres])
    final_mask = best_conf > conf_thres
    if not final_mask.any():
        return np.zeros((0, 5), dtype=np.float32)

    boxes_xyxy = boxes_xyxy[final_mask]
    best_conf = best_conf[final_mask]

    # Step 6: NMS / Apply NMS
    boxes_5 = np.column_stack([boxes_xyxy, best_conf])  # (K, 5)
    keep = numpy_nms(boxes_5, iou_thres)
    result_boxes = boxes_5[keep]

    # Step 7: 坐标缩放到原图分辨率 / Scale coords to original image resolution
    # PT 的 scale_coords 操作: gain = min(imgsz[0]/im0_h, imgsz[1]/im0_w), pad, then scale
    # 这里 boxes 坐标已是模型输入空间像素值, 需要按比例缩放到原图
    scale_x = im_w / model_w
    scale_y = im_h / model_h
    result_boxes[:, 0] *= scale_x  # x1
    result_boxes[:, 1] *= scale_y  # y1
    result_boxes[:, 2] *= scale_x  # x2
    result_boxes[:, 3] *= scale_y  # y2

    return result_boxes


def numpy_nms(boxes: np.ndarray, iou_thres: float) -> list:
    """NumPy 实现的 NMS (与 onnx_detect.py 逻辑一致).
    NumPy NMS implementation (matching onnx_detect.py logic).

    boxes: (N, 5) [x1, y1, x2, y2, score]
    Returns: list of kept indices
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    areas = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    order = scores.argsort()[::-1]  # descending by score / 按分数降序

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        # 计算剩余框与当前最高分框的 IoU / Compute IoU of remaining with current top-score box
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1.0)
        h = np.maximum(0.0, yy2 - yy1 + 1.0)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # 保留 IoU 低于阈值的框 / Keep boxes with IoU below threshold
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]

    return keep


def box_iou(box_a, box_b):
    """计算两个 bbox 的 IoU / Compute IoU of two bounding boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0, x2 - x1 + 1)
    inter_h = max(0, y2 - y1 + 1)
    inter = inter_w * inter_h
    area_a = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    area_b = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


if __name__ == '__main__':
    exit(main())
