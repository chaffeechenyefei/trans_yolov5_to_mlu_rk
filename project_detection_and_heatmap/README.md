# Project Detection & Heatmap

独立项目: YOLOv5 目标检测 + 多相机 2D 平面热力图生成.
Standalone project: YOLOv5 object detection + multi-camera 2D floor plan heatmap generation.

## 依赖 / Dependencies

```bash
pip install -r requirements.txt
```

核心依赖 (零 PyTorch):
- `opencv-python` — 图像/视频处理
- `numpy` — 数值计算
- `onnxruntime` — ONNX 模型推理

> 注意: `onnx_export.py` (PT→ONNX 导出) 需要 PyTorch, 但这是一次性操作.
> 导出完成后, 日常检测和热力图生成不需要 PyTorch.

## 快速开始 / Quick Start

### 1. 目录结构 / Directory Structure

```
project_detection_and_heatmap/
├── detect_video.py        # ONNX Runtime 视频/图像检测 (核心脚本)
├── calibrate_camera.py    # 交互式相机标定工具
├── plane_heatmap.py       # 2D平面热力图生成
├── bbox_video_synth.py    # bbox+video 合成工具
├── onnx_export.py         # PT→ONNX 模型导出 (需要 PyTorch)
├── onnx_inference.py      # ONNX Runtime 推理引擎封装
├── utils/                 # 工具模块 (NumPy 实现, 零 PyTorch)
│   ├── preprocessing.py   #   预处理 (letterbox, 图像归一化)
│   ├── postprocessing.py  #   后处理 (YOLO解码, NMS, 坐标缩放)
│   ├── nms.py             #   NumPy 实现的 NMS
│   └── common.py          #   通用工具 (视频写入, 热力图, 绘图)
├── tests/                 # 测试
│   ├── test_pt_vs_onnx.py #   PT vs ONNX 端到端差异度对比
│   ├── test_preprocessing.py  # 预处理单元测试
│   └── test_postprocessing.py # 后处理单元测试
├── data/
│   ├── calibration/       # 标定素材 & homography JSON
│   └── result/            # 检测结果 & dashboard.html
├── requirements.txt
└── README.md
```

### 2. 导出 ONNX 模型 / Export ONNX Model

```bash
python onnx_export.py --weights ../weights/yolov5s-people.pt --img_size 736 416
```

### 3. 图像检测 / Image Detection

```bash
python detect_video.py \
  --weights ../weights/yolov5s-people_mode0.onnx \
  --source ../data/calibration/test_frame_001.jpg
```

### 4. 视频检测 + BBox 输出 / Video Detection + BBox Output

```bash
python detect_video.py \
  --weights ../weights/yolov5s-people_mode0.onnx \
  --source ../data/videos/rtmart-001.mp4 \
  --sample_fps 25 \
  --bbox_output --bbox_normalized
```

### 5. 更多用法 / More Usage

参见 `docs/feature-multi-camera-bbox-to-2d-plane-heatmap.md` 了解完整工作流:
- 交互式相机标定 (`calibrate_camera.py`)
- 2D 平面热力图生成 (`plane_heatmap.py`)
- 结果 Dashboard (`data/result/dashboard.html`)

## 测试 / Testing

```bash
# PT vs ONNX 对比测试 (需要 PyTorch)
python tests/test_pt_vs_onnx.py \
  --pt_weights ../weights/yolov5s-people.pt \
  --onnx_weights ../weights/yolov5s-people_mode0.onnx

# 使用真实图像对比
python tests/test_pt_vs_onnx.py \
  --pt_weights ../weights/yolov5s-people.pt \
  --onnx_weights ../weights/yolov5s-people_mode0.onnx \
  --test_image ../data/calibration/test_frame_001.jpg

# 单元测试 (零 PyTorch)
python -m pytest tests/
```

## 架构 / Architecture

```
输入 (图像/视频)
    │
    ▼
┌───────────────────────────────────────────────┐
│  utils/preprocessing.py                       │
│  - letterbox (resize + pad, 保持宽高比)       │
│  - BGR→RGB, HWC→CHW, normalize to [0,1]      │
└───────────────────┬───────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────┐
│  onnx_inference.py :: ONNXDetector            │
│  - onnxruntime.InferenceSession               │
│  - session.run(None, {input_name: img})       │
└───────────────────┬───────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────┐
│  utils/postprocessing.py                      │
│  - decode_and_nms (YOLO 输出解码)              │
│  - numpy_nms (NumPy NMS)                      │
│  - scale_coords (坐标映射回原图)               │
└───────────────────┬───────────────────────────┘
                    │
                    ▼
            检测结果 [x1,y1,x2,y2,conf,cls]
                    │
         ┌─────────┴─────────┐
         ▼                   ▼
   bbox 文本输出      视频/图像输出
   (--bbox_output)    (带 bbox 绘制)
         │
         ▼
  ┌──────────────────────────────┐
  │  plane_heatmap.py            │
  │  + calibrate_camera.py       │
  │  → 2D 平面热力图 + Dashboard │
  └──────────────────────────────┘
```
