"""
plane_heatmap.py

将多路 camera bbox 投影到 2D 平面并生成热力图。
支持窗口聚合模式 (生成静态 PNG) 和差值热力图。

---
Project multi-camera bbox detections onto a 2D floor plan and render heatmaps.
Supports window-aggregation mode (static PNG) and delta heatmap generation.
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def _parse_group_spec(group_str: str) -> Tuple[str, Dict[str, str]]:
    """解析 --group 参数: 'window_label:camera_name=path,camera_name=path'
    支持 label 内含冒号 (如 '19:00'), 通过定位第一个 '=' 之前的最后一个 ':' 来拆分.
    Parse --group argument into (window_label, {camera_name: bbox_path}).
    Supports labels containing colons (e.g., '19:00') by splitting at the last colon
    before the first '=' character."""
    if '=' not in group_str:
        raise ValueError(f'无效的 group 格式, 缺少 camera=path 对 / Invalid group format: {group_str}')
    # 定位第一个 '=' 之前的最后一个 ':', 作为 label 与 camera 列表的分界
    # Find the last ':' before the first '=' to split label from camera list
    eq_pos = group_str.index('=')
    sep_pos = group_str.rfind(':', 0, eq_pos)
    if sep_pos == -1:
        raise ValueError(f'无效的 group 格式, 需要 label:xxx=yyy / Invalid group format: {group_str}')
    label = group_str[:sep_pos].strip()
    rest = group_str[sep_pos + 1:]
    camera_map = {}
    for pair in rest.split(','):
        pair = pair.strip()
        if not pair or '=' not in pair:
            continue
        cam_name, bbox_path = pair.split('=', 1)
        camera_map[cam_name.strip()] = bbox_path.strip()
    if not camera_map:
        raise ValueError(f'group 中没有有效的 camera=path 对 / No valid camera=path in group: {group_str}')
    return label, camera_map


def _parse_calib_spec(calib_str: str) -> Dict[str, str]:
    """解析 --calib 参数: 'camera_name=json_path,camera_name=json_path'
    Parse --calib argument into {camera_name: calib_json_path}."""
    result = {}
    for pair in calib_str.split(','):
        pair = pair.strip()
        if not pair or '=' not in pair:
            continue
        cam_name, json_path = pair.split('=', 1)
        result[cam_name.strip()] = json_path.strip()
    if not result:
        raise ValueError(f'calib 中没有有效的 camera=path 对 / No valid camera=path in calib: {calib_str}')
    return result


def load_calibration(json_path: str) -> dict:
    """加载标定 JSON, 返回包含 homography_matrix (3×3 np.array) 等字段的字典.
    Load calibration JSON, returns dict with homography_matrix as 3×3 np.array."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data['homography_matrix'] = np.array(data['homography_matrix'], dtype=np.float64)
    if data['homography_matrix'].shape != (3, 3):
        raise ValueError(f'Homography 矩阵形状异常 / Unexpected homography shape: '
                         f'{data["homography_matrix"].shape} in {json_path}')
    return data


def load_bbox_file(bbox_path: str) -> Tuple[List[dict], bool]:
    """加载 bbox 文件, 返回 (bbox列表, 是否归一化).
    每项 bbox: {frame_id, cls_id, x, y, w, h, conf}
    Load bbox file, returns (list of bbox dicts, normalized flag)."""
    bboxes = []
    normalized = False
    with open(bbox_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                # 解析 header: '# normalized: true' / Parse header for normalized flag
                if 'normalized:' in line:
                    normalized = 'true' in line.lower().split('normalized:')[1]
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 7:
                continue
            bboxes.append({
                'frame_id': int(parts[0]),
                'cls_id': int(parts[1]),
                'x': float(parts[2]),
                'y': float(parts[3]),
                'w': float(parts[4]),
                'h': float(parts[5]),
                'conf': float(parts[6]),
            })
    return bboxes, normalized


def _foot_point(bbox: dict, im_w: int, im_h: int, normalized: bool) -> Tuple[float, float]:
    """计算 bbox 底边中点在原图上的像素坐标 (foot point).
    Compute the pixel coordinate of the bbox bottom-center (foot point) on the source frame.
    公式 / Formula: foot_x = x + w/2, foot_y = y + h
    若 bbox 为归一化坐标, 先反归一化到像素 / If normalized, un-normalize to pixels first."""
    x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
    if normalized:
        x *= im_w
        y *= im_h
        w *= im_w
        h *= im_h
    foot_x = x + w / 2.0
    foot_y = y + h
    return foot_x, foot_y


def project_foot_points(bboxes: List[dict], H: np.ndarray,
                        im_w: int, im_h: int, normalized: bool) -> List[Tuple[float, float, float]]:
    """将 bbox 列表的 foot point 通过 Homography 投影到平面坐标系.
    返回 [(plane_x, plane_y, conf), ...]
    Project all bbox foot points through homography to plane coordinates.
    Returns [(plane_x, plane_y, confidence), ...]."""
    if not bboxes:
        return []

    # 计算所有 foot point 像素坐标 / Compute all foot points in pixels
    src_pts = []
    confs = []
    for b in bboxes:
        fx, fy = _foot_point(b, im_w, im_h, normalized)
        src_pts.append([fx, fy])
        confs.append(b['conf'])

    src_arr = np.array(src_pts, dtype=np.float32).reshape(-1, 1, 2)

    # 透视变换到平面坐标 / Perspective transform to plane coordinates
    dst_arr = cv2.perspectiveTransform(src_arr, H)
    dst_arr = dst_arr.reshape(-1, 2)

    return [(float(dst_arr[i][0]), float(dst_arr[i][1]), confs[i]) for i in range(len(dst_arr))]


def accumulate_heat_grid(plane_points: List[Tuple[float, float, float]],
                         grid_w: int, grid_h: int) -> np.ndarray:
    """在 2D 平面网格上累加 bbox 置信度 (以每个 foot point 为中心的高斯扩散).
    Accumulate bbox confidences onto a 2D plane grid with Gaussian spreading around each foot point.
    返回 / Returns: (grid_h, grid_w) float32 热力矩阵."""
    heat = np.zeros((grid_h, grid_w), dtype=np.float32)
    if not plane_points:
        return heat

    # 高斯核, 模拟人体在地面上占据的约 0.5m 半径区域 / Gaussian kernel for ~0.5m human footprint area
    kernel_size = 31  # 奇数 / Odd
    sigma = 8.0       # 高斯 sigma, 约覆盖一个人在地面的区域 / Approx one person footprint
    kx = cv2.getGaussianKernel(kernel_size, sigma)
    ky = cv2.getGaussianKernel(kernel_size, sigma)
    kernel = kx @ ky.T  # (kernel_size, kernel_size)
    half_k = kernel_size // 2

    for px, py, conf in plane_points:
        if conf <= 0:
            continue
        # 将 foot point 位置映射到网格坐标 / Map foot point to grid coordinates
        cx = int(round(px))
        cy = int(round(py))
        # 边界裁剪 / Clip to grid bounds
        x1 = cx - half_k
        y1 = cy - half_k
        x2 = x1 + kernel_size
        y2 = y1 + kernel_size

        # 计算 kernel 和 grid 的重叠区域 / Compute overlap region between kernel and grid
        gx1 = max(x1, 0)
        gy1 = max(y1, 0)
        gx2 = min(x2, grid_w)
        gy2 = min(y2, grid_h)
        if gx2 <= gx1 or gy2 <= gy1:
            continue

        kx1 = gx1 - x1
        ky1 = gy1 - y1
        kx2 = kx1 + (gx2 - gx1)
        ky2 = ky1 + (gy2 - gy1)

        heat[gy1:gy2, gx1:gx2] += kernel[ky1:ky2, kx1:kx2] * conf

    return heat


def render_heatmap(heat_grid: np.ndarray, plane_image: Optional[np.ndarray],
                   alpha: float = 0.55) -> np.ndarray:
    """将热力网格渲染为伪彩色图, 可选叠加到平面底图上.
    Render heat grid as color-mapped image, optionally blended onto floor plan base.
    返回 BGR uint8 图像 / Returns BGR uint8 image."""
    h, w = heat_grid.shape
    heat_max = float(heat_grid.max())

    if heat_max <= 1e-8:
        # 无热力数据时返回纯色冷图或底图 / Return cold blank or base image when no heat data
        if plane_image is not None:
            return plane_image.copy()
        cold = np.zeros((h, w, 3), dtype=np.uint8)
        cold[:] = (20, 20, 40)  # 深蓝黑底 / Dark blue-black background
        return cold

    # 归一化到 [0, 255] 并应用 JET 色图 / Normalize to [0,255] and apply JET colormap
    heat_norm = np.clip(heat_grid / heat_max, 0.0, 1.0)
    heat_u8 = (heat_norm * 255.0).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

    if plane_image is not None:
        # 确保平面底图与热力图尺寸一致 / Ensure plane image matches heatmap dimensions
        if plane_image.shape[:2] != (h, w):
            plane_bg = cv2.resize(plane_image, (w, h), interpolation=cv2.INTER_AREA)
        else:
            plane_bg = plane_image.copy()
        return cv2.addWeighted(plane_bg, 1.0 - alpha, heat_color, alpha, 0.0)
    else:
        return heat_color


def add_colorbar(image: np.ndarray, vmin: float = 0.0, vmax: float = 1.0) -> np.ndarray:
    """在图像右侧添加颜色条 (Colorbar) 指示热度范围.
    Add a vertical colorbar on the right side of the image indicating heat value range."""
    h, w = image.shape[:2]
    bar_width = 30
    bar_margin = 15
    bar_height = h - 2 * bar_margin

    # 创建色条: 从上到下 red→blue / Create color bar: top red → bottom blue
    bar = np.linspace(255, 0, bar_height, dtype=np.uint8).reshape(bar_height, 1)
    bar_color = cv2.applyColorMap(bar, cv2.COLORMAP_JET)
    bar_color = cv2.resize(bar_color, (bar_width, bar_height), interpolation=cv2.INTER_NEAREST)

    # 创建扩展画布 / Create extended canvas
    canvas = np.zeros((h, w + bar_width + bar_margin * 2, 3), dtype=np.uint8)
    canvas[:, :w] = image
    canvas[bar_margin:bar_margin + bar_height, w + bar_margin:w + bar_margin + bar_width] = bar_color

    # 标注数值 / Label the colorbar with values
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    text_color = (255, 255, 255)

    # 最大值 (红/热) / Max (red/hot)
    cv2.putText(canvas, f'{vmax:.1f}s', (w + bar_margin + bar_width + 3, bar_margin + 12),
                font, font_scale, text_color, 1, cv2.LINE_AA)
    # 最小值 (蓝/冷) / Min (blue/cold)
    cv2.putText(canvas, f'{vmin:.0f}s', (w + bar_margin + bar_width + 3, bar_margin + bar_height - 5),
                font, font_scale, text_color, 1, cv2.LINE_AA)
    # 中间值 / Mid value
    mid_y = bar_margin + bar_height // 2
    mid_val = (vmax + vmin) / 2.0
    cv2.putText(canvas, f'{mid_val:.0f}s', (w + bar_margin + bar_width + 3, mid_y),
                font, font_scale, text_color, 1, cv2.LINE_AA)

    return canvas


def plane_heatmap():
    # --- 解析参数 / Parse arguments ---
    groups: List[Tuple[str, Dict[str, str]]] = []
    for gs in opt.group:
        groups.append(_parse_group_spec(gs))

    calib_map = _parse_calib_spec(opt.calib)
    calibrations: Dict[str, dict] = {}
    for cam_name, json_path in calib_map.items():
        calibrations[cam_name] = load_calibration(json_path)
        print(f'已加载标定 {cam_name}: {json_path} | '
              f'Homography 重投影误差={calibrations[cam_name]["reprojection_error_px"]}px')

    # 加载平面底图 (可选) / Load floor plan base image (optional)
    plane_image = None
    if opt.plane_image and os.path.isfile(opt.plane_image):
        plane_image = cv2.imread(opt.plane_image)
        if plane_image is not None:
            plane_image = cv2.resize(plane_image, (opt.plane_width, opt.plane_height),
                                     interpolation=cv2.INTER_AREA)
            print(f'已加载平面底图: {opt.plane_image} -> {opt.plane_width}x{opt.plane_height} / '
                  f'Loaded floor plan: {opt.plane_image} -> {opt.plane_width}x{opt.plane_height}')

    os.makedirs(opt.output_dir, exist_ok=True)

    # 每个窗口的累积热力网格 (用于后续差值计算) / Per-window accumulated heat grids (for delta computation)
    window_heat_grids: Dict[str, np.ndarray] = {}

    print('-' * 50)

    # --- 对每个时间窗口独立处理 / Process each time window independently ---
    for window_label, camera_bbox_map in groups:
        print(f'\n处理窗口: {window_label} / Processing window: {window_label}')

        # 初始化平面热力网格 / Initialize plane heat grid
        plane_heat = np.zeros((opt.plane_height, opt.plane_width), dtype=np.float32)

        # 遍历该窗口内的所有 camera / Iterate all cameras in this window
        for cam_name, bbox_path in camera_bbox_map.items():
            if cam_name not in calibrations:
                print(f'  ✗ 缺少标定数据: {cam_name}, 跳过 / Missing calibration for {cam_name}, skipping')
                continue
            if not os.path.isfile(bbox_path):
                print(f'  ✗ bbox 文件不存在: {bbox_path}, 跳过 / Bbox file not found: {bbox_path}, skipping')
                continue

            calib = calibrations[cam_name]
            im_w = calib['image_width']
            im_h = calib['image_height']
            H = calib['homography_matrix']

            # 若目标平面尺寸与标定时不同, 缩放 Homography 矩阵以匹配新尺寸
            # Scale homography to match target plane dimensions if different from calibration
            calib_plane_w = calib.get('plane_width', opt.plane_width)
            calib_plane_h = calib.get('plane_height', opt.plane_height)
            if calib_plane_w != opt.plane_width or calib_plane_h != opt.plane_height:
                sx = opt.plane_width / calib_plane_w
                sy = opt.plane_height / calib_plane_h
                S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
                H = S @ H
                print(f'  {cam_name}: Homography 已从标定平面 ({calib_plane_w}x{calib_plane_h}) '
                      f'缩放到目标平面 ({opt.plane_width}x{opt.plane_height}) / '
                      f'Homography scaled from calibration plane to target plane')

            bboxes, normalized = load_bbox_file(bbox_path)
            if not bboxes:
                print(f'  ! {cam_name}: bbox 文件为空 / Empty bbox file: {bbox_path}')
                continue

            # 按 frame_id 分组 bbox / Group bboxes by frame_id
            frame_bboxes: Dict[int, List[dict]] = defaultdict(list)
            for b in bboxes:
                frame_bboxes[b['frame_id']].append(b)

            print(f'  {cam_name}: {len(bboxes)} bboxes, {len(frame_bboxes)} 帧 / '
                  f'{len(bboxes)} bboxes, {len(frame_bboxes)} frames')

            # 逐帧投影并累加 / Project per frame and accumulate
            total_frames = 0
            for frame_id in sorted(frame_bboxes.keys()):
                frame_pts = project_foot_points(frame_bboxes[frame_id], H, im_w, im_h, normalized)
                frame_heat = accumulate_heat_grid(frame_pts, opt.plane_width, opt.plane_height)
                plane_heat += frame_heat
                total_frames += 1

        # 除以 FPS: 将 "检测-帧" 转为 "人·秒 (person-seconds)"
        # Divide by FPS: convert detection-frames to person-seconds
        plane_heat = plane_heat / max(opt.fps, 1.0)

        # 保存该窗口的热力网格 / Store heat grid for this window
        window_heat_grids[window_label] = plane_heat

        # 渲染并保存 PNG / Render and save PNG
        heat_max = float(plane_heat.max()) if plane_heat.max() > 0 else 1.0
        rendered = render_heatmap(plane_heat, plane_image, alpha=opt.heatmap_alpha)
        rendered = add_colorbar(rendered, vmin=0.0, vmax=heat_max)

        safe_label = window_label.replace(':', '_')  # 冒号在 URL/文件名中可能引起问题
        output_path = os.path.join(opt.output_dir, f'plane_heatmap_{safe_label}.png')
        cv2.imwrite(output_path, rendered)
        print(f'  ✓ 输出: {output_path} (max={heat_max:.1f}s / 该像素点最多累计人·秒)')

    # --- 差值热力图 (Delta Heatmap) / Delta between two windows ---
    if opt.delta and len(window_heat_grids) >= 2:
        labels = list(window_heat_grids.keys())
        # 取最后两个窗口做差值 (后 - 前) / Subtract the last two windows (later - earlier)
        label_a, label_b = labels[0], labels[1]
        print(f'\n生成差值热力图: {label_b} - {label_a} / Generating delta heatmap: {label_b} - {label_a}')

        heat_a = window_heat_grids[label_a]
        heat_b = window_heat_grids[label_b]
        delta = heat_b - heat_a

        # 差值热力图: 使用发散色图 RdBu (红=增, 蓝=减) 或自定义 / Diverging colormap (red=increase, blue=decrease)
        delta_abs_max = max(abs(float(delta.min())), abs(float(delta.max())), 1e-8)
        # 映射到 [0, 1]: 中间值 0.5 对应差值为 0 / Map to [0,1]: 0.5 = zero delta
        delta_norm = np.clip(delta / (2.0 * delta_abs_max) + 0.5, 0.0, 1.0)
        delta_u8 = (delta_norm * 255.0).astype(np.uint8)
        delta_color = cv2.applyColorMap(delta_u8, cv2.COLORMAP_JET)

        # 叠加到底图 / Blend with base image
        if plane_image is not None:
            if plane_image.shape[:2] != (opt.plane_height, opt.plane_width):
                plane_bg = cv2.resize(plane_image, (opt.plane_width, opt.plane_height),
                                      interpolation=cv2.INTER_AREA)
            else:
                plane_bg = plane_image.copy()
            delta_color = cv2.addWeighted(plane_bg, 1.0 - opt.heatmap_alpha,
                                          delta_color, opt.heatmap_alpha, 0.0)

        # 添加对称色条 / Add symmetric colorbar
        delta_color = add_colorbar(delta_color, vmin=-delta_abs_max, vmax=+delta_abs_max)

        # 在图像上标注变化方向 / Annotate direction on the image
        cv2.putText(delta_color, f'Red(+) = {label_b} > {label_a}  |  Blue(-) = {label_b} < {label_a}',
                    (12, delta_color.shape[0] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        delta_path = os.path.join(opt.output_dir,
                                  f'plane_heatmap_delta_{label_a.replace(":", "_")}_{label_b.replace(":", "_")}.png')
        cv2.imwrite(delta_path, delta_color)
        print(f'  ✓ 差值图输出: {delta_path} (Δ范围: {-delta_abs_max:.1f}s ~ +{delta_abs_max:.1f}s)')

    # --- 变化率热力图 (可选) / Rate-of-change heatmap (optional) ---
    if opt.rate_of_change and len(window_heat_grids) >= 2:
        labels = list(window_heat_grids.keys())
        label_a, label_b = labels[0], labels[1]
        print(f'\n生成变化率热力图: ({label_b} - {label_a}) / max({label_a}, eps)')

        heat_a = window_heat_grids[label_a]
        heat_b = window_heat_grids[label_b]
        eps = 1e-6
        denominator = np.maximum(heat_a, eps)
        roc = (heat_b - heat_a) / denominator  # 相对变化率 / Relative rate of change

        roc_abs_max = max(abs(float(roc.min())), abs(float(roc.max())), 1e-3)
        roc_norm = np.clip(roc / (2.0 * roc_abs_max) + 0.5, 0.0, 1.0)
        roc_u8 = (roc_norm * 255.0).astype(np.uint8)
        roc_color = cv2.applyColorMap(roc_u8, cv2.COLORMAP_JET)

        if plane_image is not None:
            if plane_image.shape[:2] != (opt.plane_height, opt.plane_width):
                plane_bg = cv2.resize(plane_image, (opt.plane_width, opt.plane_height),
                                      interpolation=cv2.INTER_AREA)
            else:
                plane_bg = plane_image.copy()
            roc_color = cv2.addWeighted(plane_bg, 1.0 - opt.heatmap_alpha,
                                        roc_color, opt.heatmap_alpha, 0.0)

        roc_color = add_colorbar(roc_color, vmin=-roc_abs_max, vmax=+roc_abs_max)
        cv2.putText(roc_color, f'Red(+) = {label_b} growth  |  Blue(-) = {label_b} decline',
                    (12, roc_color.shape[0] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        roc_path = os.path.join(opt.output_dir,
                                f'plane_heatmap_roc_{label_a.replace(":", "_")}_{label_b.replace(":", "_")}.png')
        cv2.imwrite(roc_path, roc_color)
        print(f'  ✓ 变化率图输出: {roc_path} / Rate-of-change saved: {roc_path}')

    # --- 区域统计 JSON 输出 + 注入 Dashboard / ROI statistics JSON output + inject into dashboard ---
    if opt.roi_json and os.path.isfile(opt.roi_json) and len(window_heat_grids) >= 2:
        _compute_roi_stats(opt.roi_json, window_heat_grids, opt.output_dir)
        _inject_roi_into_dashboard(opt.output_dir)

    print('\n全部完成 / Done.')


def _compute_roi_stats(roi_json_path: str, window_heat_grids: Dict[str, np.ndarray], output_dir: str):
    """按 ROI 定义计算各区域的统计值并输出 JSON 和 CSV 表格.
    Compute per-ROI statistics from heat grids and output JSON + CSV."""
    with open(roi_json_path, 'r', encoding='utf-8') as f:
        roi_data = json.load(f)

    # 兼容两种 JSON 格式: {"regions": [...]} 或直接 [...]
    # Support both JSON formats: {"regions": [...]} or bare [...]
    if isinstance(roi_data, list):
        rois = roi_data
    elif isinstance(roi_data, dict):
        rois = roi_data.get('regions', roi_data.get('rois', []))
    else:
        rois = []

    if not rois:
        print('  ! ROI JSON 中没有找到 regions/rois 数组 / No regions array found in ROI JSON')
        return

    labels = list(window_heat_grids.keys())
    results = []
    for roi in rois:
        name = roi.get('name', 'unknown')
        # ROI 定义为边界框 (x1, y1, x2, y2) / ROI defined as bounding box
        x1, y1, x2, y2 = roi['x1'], roi['y1'], roi['x2'], roi['y2']
        row = {'区域/ROI': name}
        for label in labels:
            grid = window_heat_grids[label]
            h, w = grid.shape
            # 裁剪到有效范围 / Clip to valid range
            gx1 = max(int(x1), 0)
            gy1 = max(int(y1), 0)
            gx2 = min(int(x2), w)
            gy2 = min(int(y2), h)
            if gx2 > gx1 and gy2 > gy1:
                roi_sum = float(grid[gy1:gy2, gx1:gx2].sum())
            else:
                roi_sum = 0.0
            row[label] = round(roi_sum, 2)
        # 计算差值和变化率 / Compute delta and rate of change
        if len(labels) >= 2:
            val_a = row[labels[0]]
            val_b = row[labels[1]]
            row['Δ 差值/Delta'] = round(val_b - val_a, 2)
            row['变化率%/RoC'] = round((val_b - val_a) / max(val_a, 1e-6) * 100.0, 1)
        results.append(row)

    # 输出 JSON / Save as JSON
    stats_path = os.path.join(output_dir, 'roi_statistics.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'  ✓ ROI 统计 JSON: {stats_path}')

    # 输出 CSV (方便 Excel 打开) / Save as CSV (easy for Excel)
    csv_path = os.path.join(output_dir, 'roi_statistics.csv')
    if results:
        import csv
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f'  ✓ ROI 统计 CSV: {csv_path}')


def _inject_roi_into_dashboard(output_dir: str):
    """将 roi_statistics.json 数据动态注入 dashboard.html, 避免 file:// 协议下 fetch CORS 限制.
    Inject roi_statistics.json data into dashboard.html to bypass CORS under file:// protocol."""
    stats_path = os.path.join(output_dir, 'roi_statistics.json')
    dashboard_path = os.path.join(output_dir, 'dashboard.html')

    if not os.path.isfile(stats_path) or not os.path.isfile(dashboard_path):
        return

    with open(stats_path, 'r', encoding='utf-8') as f:
        stats_json_str = f.read().strip()

    with open(dashboard_path, 'r', encoding='utf-8') as f:
        html = f.read()

    html = html.replace('__ROI_STATS_PLACEHOLDER__', stats_json_str)

    with open(dashboard_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f'  ✓ Dashboard 已更新 ROI 统计数据: {dashboard_path} / Dashboard updated with ROI stats')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='多相机 2D 平面热力图生成器 / Multi-camera 2D plane heatmap generator')

    parser.add_argument('--group', type=str, action='append', default=[],
                        help='时间窗口分组: "label:camera_name=bbox_path,..." '
                             '(可多次指定, 每个 --group 对应一个时间窗口 / repeatable, one per time window)')
    parser.add_argument('--calib', type=str, required=True,
                        help='标定 JSON 映射: "camera_name=json_path,..." / Calibration JSON map')
    parser.add_argument('--plane_image', type=str, default=None,
                        help='2D 平面底图路径 (可选) / Floor plan base image path (optional)')
    parser.add_argument('--plane_width', type=int, default=1920,
                        help='平面热力图宽度 (px) / Plane heatmap width in pixels')
    parser.add_argument('--plane_height', type=int, default=1080,
                        help='平面热力图高度 (px) / Plane heatmap height in pixels')
    parser.add_argument('--window_duration_seconds', type=float, default=300.0,
                        help='窗口时长 (秒), 用于跨窗口归一化 / Window duration in seconds for cross-window normalization')
    parser.add_argument('--fps', type=float, default=25.0,
                        help='视频帧率 (用于将累积值转为物理秒) / Video FPS for converting accumulation to physical seconds')
    parser.add_argument('--heatmap_alpha', type=float, default=0.55,
                        help='热力图叠加透明度 [0, 1] / Heatmap overlay alpha')
    parser.add_argument('--delta', action='store_true',
                        help='生成差值热力图 (最后两窗口之差) / Generate delta heatmap (last two windows difference)')
    parser.add_argument('--rate_of_change', action='store_true',
                        help='生成变化率热力图 / Generate rate-of-change heatmap')
    parser.add_argument('--roi_json', type=str, default=None,
                        help='ROI 定义 JSON 路径 (可选, 用于区域统计) / ROI definition JSON path for zone statistics')
    parser.add_argument('--output_dir', type=str, default='data/result/',
                        help='输出目录 / Output directory')

    opt = parser.parse_args()

    if not opt.group:
        parser.error('至少需要指定一个 --group / At least one --group is required')

    plane_heatmap()
