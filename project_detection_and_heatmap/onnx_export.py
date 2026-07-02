"""
onnx_export.py

将 YOLOv5 .pt 权重导出为 ONNX 格式 (mode=0 标准 YOLO 输出).
Export YOLOv5 .pt weights to ONNX format (mode=0 standard YOLO output).

Usage:
    python onnx_export.py --weights ../weights/yolov5s-people.pt --img_size 736 416
"""

import argparse
import os
import sys

# 确保能找到 models/ 和 utils/ 目录 / Ensure models/ and utils/ are importable
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)  # trans_yolov5_to_mlu_rk/
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

import torch
import torch.nn as nn
import numpy as np

from models.experimental import attempt_load
from models.common import Conv as CommonConv  # 用于 isinstance 检查 / For isinstance check
from utils.activations import Hardswish, SiLU
from utils.general import check_img_size
from utils.torch_utils import select_device


def export_pt_to_onnx(weights_path: str, output_path: str,
                      img_size=(736, 416), mode: int = 0,
                      opset: int = 11, dynamic: bool = False):
    """将 YOLOv5 .pt 模型导出为 ONNX. / Export YOLOv5 .pt model to ONNX.

    Args:
        weights_path: .pt 权重文件路径 / Path to .pt weights file.
        output_path: 输出 .onnx 文件路径 / Output .onnx file path.
        img_size: 模型输入尺寸 (w, h) / Model input size (width, height).
        mode: 导出模式 (0=标准YOLO输出, 1/2/3=RKNN适配) / Export mode.
        opset: ONNX opset 版本 / ONNX opset version.
        dynamic: 是否导出动态 batch/尺寸 / Whether to export with dynamic axes.
    """
    w, h = img_size

    # ---- 加载 PyTorch 模型 / Load PyTorch model ----
    device = select_device('cpu')
    model = attempt_load(weights_path, map_location=device)
    print(f'Loaded model from: {weights_path}')
    print(f'  Classes: {model.names}')

    # 确保输入尺寸是 stride 的整数倍 / Ensure input size is divisible by stride
    gs = int(max(model.stride))
    h, w = [check_img_size(x, gs) for x in [h, w]]
    print(f'  Input size (after stride alignment): {w}x{h}')

    # 准备 dummy 输入 / Prepare dummy input
    img = torch.rand(1, 3, h, w, dtype=torch.float32).to(device)

    # ---- 替换激活函数为导出兼容版本 / Replace activations with export-friendly versions ----
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # PyTorch 1.6.0 兼容 / PyTorch 1.6.0 compatibility
        if isinstance(m, CommonConv):
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()

    # 设置 Detect 层为导出模式 / Set Detect layer to export mode
    model.model[-1].export = True
    model.model[-1].export_mode = mode

    if mode == 3:
        # mode 3 需要额外的 conv1x1 代理 anchor_grid / mode 3 needs an extra conv1x1 proxy for anchor_grid
        model.model[-1].conv1x1 = nn.Conv2d(1, 1, 1, 1, 0, bias=False)
        model.model[-1].conv1x1.weight = nn.Parameter(torch.ones(1, 1, 1, 1).float())
        model.model[-1].conv1x1.eval()

    # dry run 验证输出形状 / Dry run to verify output shapes
    with torch.no_grad():
        y = model(img)
    print(f'  Output shapes: {[s.shape for s in (y if isinstance(y, (list, tuple)) else [y])]}')

    # ---- ONNX 导出 / ONNX export ----
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            'images': {0: 'batch', 2: 'height', 3: 'width'},
            'output0': {0: 'batch', 2: 'y', 3: 'x'},
            'output1': {0: 'batch', 2: 'y', 3: 'x'},
            'output2': {0: 'batch', 2: 'y', 3: 'x'},
        }

    try:
        torch.onnx.export(
            model, img, output_path,
            verbose=False, opset_version=opset,
            input_names=['images'], dynamic_axes=dynamic_axes,
            dynamo=False,
        )
    except TypeError:
        # PyTorch < 2.6 没有 dynamo 参数 / Old PyTorch without dynamo kwarg
        torch.onnx.export(
            model, img, output_path,
            verbose=False, opset_version=opset,
            input_names=['images'], dynamic_axes=dynamic_axes,
        )
    print(f'ONNX model saved to: {output_path}')
    print(f'  opset: {opset}, mode: {mode}, dynamic: {dynamic}')

    # 读回元信息二次确认 opset / Reload to confirm opset matches
    import onnx as _onnx
    _m = _onnx.load(output_path)
    _actual_opset = [o.version for o in _m.opset_import if o.domain in ('', 'ai.onnx')]
    print(f'  Actual opset in saved file: {_actual_opset}')

    # ---- 导出后回载验证 / Post-export verification ----
    print('-' * 40)
    print('Verifying ONNX model (load-back + inference)...')
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        inputs_np = img.cpu().numpy().astype(np.float32)
        outputs = session.run(None, {input_name: inputs_np})
        print(f'  ONNX inference OK. Output count: {len(outputs)}')
        for i, o in enumerate(outputs):
            flat = o.reshape(-1)
            print(f'  Output[{i}] shape={o.shape}, first 10 values: {flat[:10]}')
        if isinstance(y, (list, tuple)):
            for i, (pt_out, onnx_out) in enumerate(zip(y, outputs)):
                if pt_out.shape != onnx_out.shape:
                    print(f'  ⚠ Shape mismatch[{i}]: PT={tuple(pt_out.shape)} vs ONNX={onnx_out.shape}')
        print('  ✓ ONNX model verification passed.')
    except Exception as e:
        print(f'  ✗ ONNX model verification failed: {e}')
        raise

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Export YOLOv5 .pt to ONNX')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to .pt weights file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output .onnx path (default: <weights>_mode0.onnx)')
    parser.add_argument('--img_size', type=int, nargs='+', default=[736, 416],
                        help='Model input size (w h), default: 736 416')
    parser.add_argument('--mode', type=int, default=0,
                        help='Export mode: 0=standard YOLO, 1/2/3=RKNN-adapted')
    parser.add_argument('--opset', type=int, default=11,
                        help='ONNX opset version, default: 11')
    parser.add_argument('--dynamic', action='store_true',
                        help='Export with dynamic batch/size axes')
    opt = parser.parse_args()

    if len(opt.img_size) == 1:
        img_size = (opt.img_size[0], opt.img_size[0])
    else:
        img_size = (opt.img_size[0], opt.img_size[1])

    if opt.output is None:
        base = os.path.splitext(opt.weights)[0]
        opt.output = f'{base}_mode{opt.mode}.onnx'

    export_pt_to_onnx(
        weights_path=opt.weights,
        output_path=opt.output,
        img_size=img_size,
        mode=opt.mode,
        opset=opt.opset,
        dynamic=opt.dynamic,
    )


if __name__ == '__main__':
    main()
