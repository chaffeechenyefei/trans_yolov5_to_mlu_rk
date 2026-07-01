# Prequisite

遵守how-to-read-md-files技能指导

## PY环境
位置
```
which python
/opt/anaconda3/envs/akila/bin/python
```

---

# Background
[yolo检测迭代记录](../detect_video.md), 可以获取 video 的 bbox
[2d plane heatmap solution](supermarket-heatmap-solution.md)

# Task
## [modified] 1. 生成2d heatmap plane的计划
### 1.1 现状与期望
- camera_A `data/videos/rtmart-001.mp4` 与 `data/videos/rtmart-002.mp4` 是同一个摄像机不同时刻的video
- camera_B `data/videos/rtmart-003.mp4` 与 `data/videos/rtmart-004.mp4` 是另一个摄像机不同时刻的video
- `data/videos/rtmart-001.mp4` 与 `data/videos/rtmart-003.mp4` 都是 20 点开始的 5min数据
- `data/videos/rtmart-002.mp4` 与 `data/videos/rtmart-004.mp4` 都是 19 点开始的 5min数据
- 他们对应的bbox在 [result目录下](../data/result)
- 具体bbox的使用请参考 [yolo检测迭代记录](../detect_video.md), 和合成代码 - [bbox_video_synth](../bbox_video_synth.py)
- 需要分别将camera A 和 B, 与2D平面进行点的映射标定

- [modified] 目前两个camera总共有2个时间段, 基本都是5min, 生成的静态PNG用5min为一个窗口进行聚合, 聚合成两个PNG图像
- [modified] 已规划dashboard展示方案: 两个时间窗口的同环比分析(差值热力图+变化率热力图+区域统计对比), 见 1.2.6 节

### 1.2 实施计划 / Implementation Plan

#### 1.2.1 整体数据流 / Overall Data Flow

```
┌──────────────────────────────────────────────────────────────────┐
│  Step 1: Calibration (一次性标定, 每个camera做一次)               │
│  camera image ←→ 2D floor plan 对应点 → 计算 Homography Matrix   │
└────────────────────────────┬─────────────────────────────────────┘
                             │ homography_A.json, homography_B.json
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  Step 2: 读取已有 bbox 文件 (normalized [0,1] 格式)              │
│  按时间窗口分组:                                                  │
│    Window 19:00-19:05 → rtmart-002_bbox.txt + rtmart-004_bbox.txt│
│    Window 20:00-20:05 → rtmart-001_bbox.txt + rtmart-003_bbox.txt│
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  Step 3: 每个窗口内, 逐帧 bbox → 投影到 2D 平面                   │
│  - 取 bbox 底边中点 (foot point) 作为人员在地面的位置              │
│  - 反归一化: x_pixel = x_norm * im_w, y_pixel = y_norm * im_h    │
│  - 通过 Homography 矩阵变换到 2D 平面坐标                          │
│  - camera_A 的 bbox 用 H_A 变换; camera_B 的 bbox 用 H_B 变换     │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  Step 4: 5min 窗口内直接累加 (无时间衰减)                          │
│  - 窗口足够短 (5min), 不需要衰减, 直接累加所有 bbox 的置信度       │
│  - 两路 camera 的贡献在同一个平面网格上叠加                        │
│  - 每个窗口产出独立的累积热力矩阵                                  │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  Step 5: 渲染输出 2D 平面热力图 PNG                                │
│  - 每个窗口 → cv2.applyColorMap + 叠加到平面底图 → 独立 PNG       │
│  - 产出: plane_heatmap_1900.png + plane_heatmap_2000.png          │
└──────────────────────────────────────────────────────────────────┘
```

#### 1.2.2 新增脚本规划 / New Scripts

| 脚本 | 用途 | 依赖 |
|---|---|---|
| `calibrate_camera.py` | 交互式标定工具: 用户在 camera 画面和平面图上点击对应点, 计算并保存 Homography 矩阵 | OpenCV, 已有的 bbox_video_synth |
| `plane_heatmap.py` | 读取多路 bbox 文件 + 标定参数, 生成 2D 平面热力图 | OpenCV, NumPy, 复用 detect_video 热力逻辑 |

#### 1.2.3 calibrate_camera.py 详细设计

**输入:**
- 一张 camera 的截图 (从 video 中抽取一帧, 包含清晰的地面特征)
- 一张 2D 平面图 (超市 floor plan 图片或空白参考网格)
- 用户在两张图上分别点击 N≥4 组对应点 (如地砖角、货架角等明显地面标记)

**处理流程:**
1. 用 OpenCV 展示 camera 截图, 用户点击 N 个点 (记录 image points)
2. 用 OpenCV 展示平面图, 用户在对应位置点击 N 个点 (记录 plane points)
3. `cv2.findHomography(src_pts=image_points, dst_pts=plane_points)` 计算变换矩阵
4. 保存为 JSON: `{camera_name, image_width, image_height, homography_matrix (3×3 list), plane_width, plane_height, correspondences: [{img_xy, plane_xy}], timestamp}`

**CLI 示例:**
```shell
python calibrate_camera.py \
  --camera_name camera_A \
  --camera_image data/calibration/rtmart-001_frame0.jpg \
  --plane_image data/calibration/floor_plan.png \
  --output data/calibration/homography_A.json
```

#### 1.2.4 plane_heatmap.py 详细设计

**输入:**
- 多个 bbox 文件路径 + 对应的 camera 标定 JSON
- 2D 平面底图 (可选, 用于叠加渲染)
- 聚合模式: `window` (按时间窗口聚合成静态PNG) 或 `streaming` (逐帧衰减输出视频)
- `--window_duration_seconds`: 窗口聚合时长, 默认 300s (5min)

**处理流程:**

*模式 A: window 窗口聚合模式 (当前需求, 生成静态PNG)*
1. 解析 CLI 中的时间窗口分组: `--group` 参数将 bbox 文件分组到不同窗口, 如 `--group "19:00:camera_A=rtmart-002_bbox.txt,camera_B=rtmart-004_bbox.txt" --group "20:00:camera_A=rtmart-001_bbox.txt,camera_B=rtmart-003_bbox.txt"`
2. 对每个时间窗口独立处理:
   a. 加载该窗口所有 camera 的 Homography 矩阵和 bbox 文件
   b. 遍历所有帧, 计算 foot point → 反归一化 → Homography 投影到平面
   c. 在平面网格上**直接累加** (窗口内不做时间衰减, 因为窗口已经足够短)
   d. 网格值 = Σ(所有 bbox 的置信度), 反映该窗口内的总停留/经过人次
3. 渲染: `cv2.applyColorMap` + 叠加到平面底图 → 输出 PNG
4. 支持同时生成两个窗口的 PNG

*模式 B: streaming 流式模式 (可选扩展)*
1. 按帧时间线遍历, 每帧投影后应用滑动窗口时间衰减
2. 逐帧输出热力图, 可合成视频

**CLI 示例 (当前需求 - 两窗口静态PNG):**
```shell
python plane_heatmap.py \
  --group "19:00:camera_A=data/result/rtmart-002_bbox.txt,camera_B=data/result/rtmart-004_bbox.txt" \
  --group "20:00:camera_A=data/result/rtmart-001_bbox.txt,camera_B=data/result/rtmart-003_bbox.txt" \
  --calib camera_A=data/calibration/homography_A.json,camera_B=data/calibration/homography_B.json \
  --plane_image data/calibration/floor_plan.png \
  --plane_width 1920 --plane_height 1080 \
  --window_duration_seconds 300 \
  --output_dir data/result/
  # 产出: data/result/plane_heatmap_1900.png, data/result/plane_heatmap_2000.png
```

#### 1.2.5 关键技术决策 / Key Design Decisions

| 决策点 | 方案 | 理由 |
|---|---|---|
| bbox 到地面的映射点 | 使用 bbox 底边中点 `(x + w/2, y + h)` 作为 foot point | 行人检测 bbox 的底边通常对应脚部/地面位置, 是透视投影最合理的锚点 |
| 聚合策略 | 5min 窗口内直接累加, 不做时间衰减 | 窗口足够短, 衰减无意义; 直接累加 = 该时段内的总"人·秒"密度, 物理含义更清晰 |
| 多 camera 融合 | 简单叠加 (后续可升级为可信度加权) | 方案文档中的可信度地图需要额外的 geometric 建模, 首版先用等权叠加快速验证效果 |
| 2D 平面坐标系 | 以 floor plan 图片像素为坐标系 | 无需额外定义物理单位, 直接对应渲染输出 |
| 窗口分组方式 | CLI `--group` 参数显式分组 | 灵活支持任意数量的时间窗口和 camera 组合, 不硬编码 19:00/20:00 |

#### 1.2.6 同环比分析与 Dashboard 展示 / YoY/HoH Analysis & Dashboard

##### 分析维度

两个时间窗口 (19:00 vs 20:00) 的平面热力图可以产生以下分析:

| 分析类型 | 产出 | 业务价值 |
|---|---|---|
| **原始热力图** | 19:00 热力 PNG + 20:00 热力 PNG | 直观展示两个时段的客流空间分布 |
| **差值热力图 (Δ Heatmap)** | 20:00 - 19:00 的逐像素差值, 红色=增, 蓝色=减 | 一眼看出哪些区域人流"涨了"或"跌了" |
| **变化率热力图** | (20:00 - 19:00) / max(19:00, ε) 的相对变化率 | 排除绝对量级影响, 关注结构性变化 |
| **区域统计对比** | 按预定义 ROI (如入口、货架区、收银台) 汇总热力值 | 量化各功能区的时段差异 |

##### 同环比 (YoY/HoH) 应用场景

- **环比分析 (HoH)**: 19:00→20:00 是相邻时段, 对比可发现:
  - 晚高峰人流迁徙方向 (如: 19:00 集中在生鲜区, 20:00 转移到日用品区)
  - 收银台排队压力是否在 20:00 显著上升
  - 促销活动的时段效果差异

- **同比分析 (YoY)**: 后续积累多天数据后:
  - 同一时段 (如每天 19:00-19:05) 跨天对比, 发现周期性规律
  - 周末 vs 工作日同一时段差异
  - 活动日 vs 非活动日对比

##### 建议的 Dashboard 展示方案

```
┌─────────────────────────────────────────────────────────┐
│  2D Plane Heatmap Dashboard                             │
├──────────────────────┬──────────────────────────────────┤
│  19:00-19:05         │  20:00-20:05                     │
│  [热力图 PNG]        │  [热力图 PNG]                    │
│                      │                                  │
├──────────────────────┴──────────────────────────────────┤
│  Δ 差值热力图 (20:00 - 19:00)                           │
│  [差值热力图 PNG]    红色↑ = 人流增加 / 蓝色↓ = 人流减少 │
│                      │                                  │
├─────────────────────────────────────────────────────────┤
│  区域统计对比表                                          │
│  ┌──────────┬────────┬────────┬─────────┬────────┐     │
│  │ 区域     │ 19:00  │ 20:00  │ Δ 差值   │ 变化率  │     │
│  ├──────────┼────────┼────────┼─────────┼────────┤     │
│  │ 入口区   │ 1,234  │ 1,567  │ +333    │ +27%   │     │
│  │ 生鲜区   │ 2,100  │ 1,850  │ -250    │ -12%   │     │
│  │ 收银区   │   890  │ 1,420  │ +530    │ +60%   │     │
│  │ ...      │   ...  │   ...  │   ...   │  ...   │     │
│  └──────────┴────────┴────────┴─────────┴────────┘     │
└─────────────────────────────────────────────────────────┘
```

> [thinking] Dashboard 可用简单 HTML 页面实现: 内嵌 PNG + 表格, 无需后端。后续可升级为 WebSocket 实时推送。

#### 1.2.7 实施步骤 / Implementation Steps

1. **准备标定素材**: 从 `rtmart-001.mp4` 和 `rtmart-003.mp4` 中各抽取一帧包含清晰地面特征的画面作为标定参考图
2. **编写 `calibrate_camera.py`**: 交互式双图点击标定工具, 输出 Homography JSON
3. **手工标定**: 运行标定工具, 为 camera_A 和 camera_B 分别生成 `homography_A.json` 和 `homography_B.json`
4. **编写 `plane_heatmap.py`** (核心脚本):
   - 实现 `--group` 窗口分组解析
   - 实现 foot-point 投影 + 窗口内累加
   - 实现热力图渲染 (applyColorMap + 叠加底图)
   - 输出每个窗口的独立 PNG
5. **编写差值热力图生成**: 在 `plane_heatmap.py` 中增加 `--delta` 模式, 对两个已生成的网格矩阵做差值
6. **编写 Dashboard HTML**: 简单的静态 HTML, 展示两张原始热力图 + 差值图 + 区域统计表
7. **验证**:
   - 19:00 窗口 (rtmart-002 + rtmart-004) → `plane_heatmap_1900.png`
   - 20:00 窗口 (rtmart-001 + rtmart-003) → `plane_heatmap_2000.png`
   - 差值图 `plane_heatmap_delta_1900_2000.png`
   - 在 HTML dashboard 中对比查看

### 1.3 已实现 / Implemented

| 文件 | 说明 |
|---|---|
| `data/calibration/rtmart-001_frame0.jpg` | camera_A 标定参考帧 (960×544) |
| `data/calibration/rtmart-003_frame0.jpg` | camera_B 标定参考帧 (960×544) |
| `data/calibration/roi_sample.json` | ROI 区域定义示例文件, 可按需修改区域名和坐标 |
| `calibrate_camera.py` | 交互式标定工具: 左右键点击收集 ≥4 组对应点, RANSAC 计算 Homography, 输出 JSON |
| `plane_heatmap.py` | 核心热力图脚本, 支持: `--group` 窗口分组、foot-point 投影 + 高斯扩散累加、窗口内直接累加(无衰减)、差值热力图 `--delta`、变化率热力图 `--rate_of_change`、ROI 区域统计 `--roi_json` |
| `data/result/dashboard.html` | 静态仪表盘 HTML: 深色主题, 左右对比两窗口热力图 + 差值图 + 变化率图 + ROI 统计表, 纯前端无后端依赖 |

#### 使用流程 / Usage Flow

```shell
# Step 1: 交互式标定 (每个 camera 执行一次, 需人工点击对应点)
python calibrate_camera.py \
  --camera_name camera_A \
  --camera_image data/calibration/rtmart-001_frame0.jpg \
  --plane_image data/calibration/floor_plan.png \
  --output data/calibration/homography_A.json

python calibrate_camera.py \
  --camera_name camera_B \
  --camera_image data/calibration/rtmart-003_frame0.jpg \
  --plane_image data/calibration/floor_plan.png \
  --output data/calibration/homography_B.json

# Step 2: 生成平面热力图 (两窗口 + 差值 + 变化率)
python plane_heatmap.py \
  --group "19:00:camera_A=data/result/rtmart-002_bbox.txt,camera_B=data/result/rtmart-004_bbox.txt" \
  --group "20:00:camera_A=data/result/rtmart-001_bbox.txt,camera_B=data/result/rtmart-003_bbox.txt" \
  --calib "camera_A=data/calibration/homography_A.json,camera_B=data/calibration/homography_B.json" \
  --plane_image data/calibration/floor_plan.png \
  --plane_width 1920 --plane_height 1080 \
  --window_duration_seconds 300 \
  --delta --rate_of_change \
  --output_dir data/result/

# Step 3: 在浏览器中打开 dashboard 查看结果
# open data/result/dashboard.html
```

> **注意**: 使用前需准备好 `data/calibration/floor_plan.png` (超市平面图) 并完成 Step 1 的交互式标定。
