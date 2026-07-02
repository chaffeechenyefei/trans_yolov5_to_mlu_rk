"""
test_pt_vs_onnx.py — PT vs ONNX 输出差异度对比测试.

对比 PyTorch (.pt) 与 ONNX Runtime (.onnx) 推理输出的一致性.
Compare inference output consistency between PyTorch (.pt) and ONNX Runtime (.onnx).

Usage:
    python tests/test_pt_vs_onnx.py \
      --pt_weights ../weights/yolov5s-people.pt \
      --onnx_weights ../weights/yolov5s-people_mode0.onnx \
      --img_size 736 416

    # 使用真实图像 / With real image
    python tests/test_pt_vs_onnx.py \
      --pt_weights ../weights/yolov5s-people.pt \
      --onnx_weights ../weights/yolov5s-people_mode0.onnx \
      --img_size 736 416 \
      --test_image ../data/calibration/test_frame_001.jpg
"""

import argparse
import os
import sys
import time

# 确保能找到 workspace 根目录的 models/ 和 utils/ (PT 推理需要 PyTorch 依赖)
# Ensure workspace root's models/ and utils/ are importable (PT inference needs PyTorch)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)  # project_detection_and_heatmap/
_WORKSPACE = os.path.dirname(_PROJECT_DIR)   # trans_yolov5_to_mlu_rk/
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

# 从重构后的 onnx_inference 导入 ONNX 解码函数
# Import ONNX decode function from refactored onnx_inference
sys.path.insert(0, _PROJECT_DIR)
from onnx_inference import ONNXDetector
from utils.postprocessing import decode_and_nms, scale_coords as np_scale_coords
from utils.preprocessing import parse_img_size, check_img_size as np_check_img_size


def pt_inference(model, img_numpy: np.ndarray, device, half: bool):
    """PyTorch 推理, 返回原始输出张量.
    PyTorch inference, returns raw output tensors."""
    img = letterbox(img_numpy, new_shape=(416, 736), stride=32)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img)[0]
    return pred, img.shape[2:]


def get_onnx_session(onnx_path: str):
    """创建 ONNX Runtime 会话 / Create ONNX Runtime session."""
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    print(f'ONNX session:')
    print(f'  Input: {input_name}, shape={session.get_inputs()[0].shape}')
    for o in session.get_outputs():
        print(f'  Output: {o.name}, shape={o.shape}')
    return session, input_name, output_names


def compare_end_to_end(pt_weights, onnx_weights, img_size, device, half, test_image=None):
    """端到端对比 PT 与 ONNX 的输出差异.

    Returns:
        dict with pass/fail and detailed stats.
    """
    print('=' * 60)
    print('End-to-end PT vs ONNX comparison')

    # --- 加载 PT 模型 / Load PT model ---
    print('Loading PyTorch model...')
    pt_model = attempt_load(pt_weights, map_location=device)
    stride = int(pt_model.stride.max())
    imgsz = parse_img_size(img_size, stride)
    if half:
        pt_model.half()
    print(f'  PT model loaded. Stride={stride}, imgsz={imgsz}, names={pt_model.names}')

    # --- 加载 ONNX 模型 / Load ONNX model ---
    print('Loading ONNX model...')
    onnx_session, input_name, _ = get_onnx_session(onnx_weights)
    onnx_input_shape = onnx_session.get_inputs()[0].shape
    onnx_h, onnx_w = onnx_input_shape[2], onnx_input_shape[3]
    model_h, model_w = onnx_h, onnx_w

    # --- 准备测试输入 / Prepare test input ---
    if test_image and os.path.exists(test_image):
        print(f'Using test image: {test_image}')
        img_numpy = cv2.imread(test_image)
        if img_numpy is None:
            raise FileNotFoundError(f'Cannot read image: {test_image}')
    else:
        print('No test image, using random noise (uint8, uniform 0-255, seed=42)')
        rng = np.random.RandomState(42)
        img_numpy = rng.randint(0, 256, size=(544, 960, 3), dtype=np.uint8)

    print(f'  Image shape: {img_numpy.shape}')

    # === PT 推理 ===
    print('-' * 40)
    print('PT inference...')

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

    t0 = time.time()
    with torch.no_grad():
        pt_pred = pt_model(img_t)[0]
    pt_pred_nms = non_max_suppression(pt_pred, 0.5, 0.3)[0]
    pt_time = time.time() - t0

    if pt_pred_nms is not None and len(pt_pred_nms):
        pt_pred_nms[:, :4] = scale_coords(img_t.shape[2:], pt_pred_nms[:, :4], img_numpy.shape).round()

    print(f'  PT: {len(pt_pred_nms) if pt_pred_nms is not None else 0} detections in {pt_time * 1000:.1f}ms')

    # === ONNX 推理 ===
    print('-' * 40)
    print('ONNX inference...')

    detector = ONNXDetector(
        onnx_path=onnx_weights,
        device='cpu',
        model_h=model_h,
        model_w=model_w,
        conf_thres=0.5,
        iou_thres=0.3,
    )

    t0 = time.time()
    onnx_det = detector.infer(img_numpy)
    onnx_time = time.time() - t0
    print(f'  ONNX: {len(onnx_det)} detections in {onnx_time * 1000:.1f}ms')

    # === 对比 / Compare ===
    print('=' * 60)
    print('Comparison results:')

    results = {
        'pt_count': len(pt_pred_nms) if pt_pred_nms is not None else 0,
        'onnx_count': len(onnx_det),
        'pt_time_ms': round(pt_time * 1000, 2),
        'onnx_time_ms': round(onnx_time * 1000, 2),
    }

    # bbox 数量差异 / Bbox count difference
    count_diff = abs(results['pt_count'] - results['onnx_count'])
    results['count_diff'] = count_diff

    # 配对 bbox 的 IoU 和坐标差 / Pair bbox by IoU and coords diff
    if pt_pred_nms is not None and len(pt_pred_nms) and len(onnx_det):
        pt_boxes = pt_pred_nms[:, :4].cpu().numpy()
        pt_confs = pt_pred_nms[:, 4].cpu().numpy()
        pt_cls = pt_pred_nms[:, 5].cpu().numpy()

        onnx_boxes = onnx_det[:, :4]
        onnx_confs = onnx_det[:, 4]
        onnx_cls = onnx_det[:, 5]

        # 计算 IoU 矩阵 / Compute IoU matrix
        iou_matrix = np.zeros((len(pt_boxes), len(onnx_boxes)))
        for i, pb in enumerate(pt_boxes):
            # PT box area
            p_x1, p_y1, p_x2, p_y2 = pb
            for j, ob in enumerate(onnx_boxes):
                o_x1, o_y1, o_x2, o_y2 = ob
                ix1 = max(p_x1, o_x1)
                iy1 = max(p_y1, o_y1)
                ix2 = min(p_x2, o_x2)
                iy2 = min(p_y2, o_y2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                union = (p_x2 - p_x1) * (p_y2 - p_y1) + (o_x2 - o_x1) * (o_y2 - o_y1) - inter
                iou_matrix[i, j] = inter / union if union > 0 else 0.0

        # 贪心配对 / Greedy matching
        matched_pt = set()
        matched_onnx = set()
        coord_diffs = []
        conf_diffs = []
        cls_match_count = 0

        while True:
            if len(matched_pt) >= iou_matrix.shape[0] or len(matched_onnx) >= iou_matrix.shape[1]:
                break
            remaining_rows = [r for r in range(iou_matrix.shape[0]) if r not in matched_pt]
            remaining_cols = [c for c in range(iou_matrix.shape[1]) if c not in matched_onnx]
            if not remaining_rows or not remaining_cols:
                break
            sub = iou_matrix[np.ix_(remaining_rows, remaining_cols)]
            max_idx = sub.argmax()
            max_iou = sub.flat[max_idx]
            if max_iou < 0.5:
                break
            r = remaining_rows[max_idx // sub.shape[1]]
            c = remaining_cols[max_idx % sub.shape[1]]
            matched_pt.add(r)
            matched_onnx.add(c)

            coord_diff = np.linalg.norm(pt_boxes[r] - onnx_boxes[c])
            coord_diffs.append(coord_diff)
            conf_diff = abs(float(pt_confs[r]) - float(onnx_confs[c]))
            conf_diffs.append(conf_diff)
            if int(pt_cls[r]) == int(onnx_cls[c]):
                cls_match_count += 1

        results['matched_pairs'] = len(matched_pt)
        results['mean_coord_diff'] = float(np.mean(coord_diffs)) if coord_diffs else 0.0
        results['max_coord_diff'] = float(np.max(coord_diffs)) if coord_diffs else 0.0
        results['mean_conf_diff'] = float(np.mean(conf_diffs)) if conf_diffs else 0.0
        results['max_conf_diff'] = float(np.max(conf_diffs)) if conf_diffs else 0.0
        results['cls_accuracy'] = cls_match_count / max(len(matched_pt), 1)
    else:
        results['matched_pairs'] = 0
        results['mean_coord_diff'] = 0.0
        results['max_coord_diff'] = 0.0
        results['mean_conf_diff'] = 0.0
        results['max_conf_diff'] = 0.0
        results['cls_accuracy'] = 1.0

    # --- 打印结果 / Print results ---
    print(f'  PT 检测数 / PT detections:     {results["pt_count"]}')
    print(f'  ONNX 检测数 / ONNX detections: {results["onnx_count"]}')
    print(f'  数量差异 / Count diff:          {results["count_diff"]}')
    print(f'  配对 bbox / Matched pairs:      {results["matched_pairs"]}')
    print(f'  坐标差均值 / Mean coord diff:   {results["mean_coord_diff"]:.2f} px')
    print(f'  坐标差最大 / Max coord diff:    {results["max_coord_diff"]:.2f} px')
    print(f'  置信度差均值 / Mean conf diff:  {results["mean_conf_diff"]:.6f}')
    print(f'  置信度差最大 / Max conf diff:   {results["max_conf_diff"]:.6f}')
    print(f'  类别一致性 / Cls accuracy:      {results["cls_accuracy"]:.1%}')
    print(f'  PT 推理耗时 / PT time:          {results["pt_time_ms"]:.1f}ms')
    print(f'  ONNX 推理耗时 / ONNX time:      {results["onnx_time_ms"]:.1f}ms')

    # --- 判定 / Pass/Fail ---
    all_pass = True
    checks = []

    # 数量一致率 / Count consistency
    if results['pt_count'] > 0 and results['onnx_count'] > 0:
        count_ratio = min(results['pt_count'], results['onnx_count']) / max(results['pt_count'], results['onnx_count'])
        if count_ratio >= 0.95:
            checks.append(('✓ 数量一致率 / Count ratio >= 95%', True))
        else:
            checks.append((f'✗ 数量一致率 / Count ratio = {count_ratio:.1%} < 95%', False))
            all_pass = False
    else:
        if results['pt_count'] == 0 and results['onnx_count'] == 0:
            checks.append(('✓ 双方均无检测 / Both zero detections', True))
        else:
            checks.append((f'⚠ 检测数不一致 / Count mismatch: PT={results["pt_count"]}, ONNX={results["onnx_count"]}', False))
            all_pass = False

    # 配对 bbox 的坐标差 / Coord diff of matched bboxes
    if results['matched_pairs'] > 0:
        if results['max_coord_diff'] <= 2.0:
            checks.append((f'✓ 配对坐标差 / Max coord diff {results["max_coord_diff"]:.2f}px <= 2px', True))
        else:
            checks.append((f'⚠ 配对坐标差 / Max coord diff {results["max_coord_diff"]:.2f}px > 2px', False))
            all_pass = False

    # 置信度差 / Conf diff
    if results['matched_pairs'] > 0:
        if results['max_conf_diff'] <= 0.01:
            checks.append((f'✓ 置信度差 / Max conf diff {results["max_conf_diff"]:.6f} <= 0.01', True))
        else:
            checks.append((f'⚠ 置信度差 / Max conf diff {results["max_conf_diff"]:.6f} > 0.01', False))

    print('-' * 40)
    print('Checks:')
    for msg, passed in checks:
        print(f'  {msg}')

    print('=' * 60)
    if all_pass:
        print('✓ ALL CHECKS PASSED / 所有检查通过')
    else:
        print('⚠ Some checks failed / 部分检查未通过')

    return results, all_pass


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
                        help='Optional: path to test image')
    opt = parser.parse_args()

    device = select_device(opt.device)
    half = device.type != 'cpu'

    results, passed = compare_end_to_end(
        pt_weights=opt.pt_weights,
        onnx_weights=opt.onnx_weights,
        img_size=opt.img_size,
        device=device,
        half=half,
        test_image=opt.test_image,
    )

    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
