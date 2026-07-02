"""
calibrate_camera.py

交互式相机标定工具: 用户在 camera 画面和 2D 平面图上点击对应点,
计算并保存透视变换矩阵 (Homography Matrix)。

---
Interactive camera calibration tool: click corresponding points on
camera image and 2D floor plan, compute and save the perspective
transform (Homography) matrix.
"""

import argparse
import json
import os
import sys
from typing import List, Tuple

import cv2
import numpy as np

# 标定点绘制样式 / Calibration point drawing style
_POINT_RADIUS = 5
_POINT_COLOR_CAMERA = (0, 255, 0)   # 绿 / Green for camera image
_POINT_COLOR_PLANE = (255, 0, 0)    # 蓝 / Blue for plane image
_LINE_COLOR = (0, 255, 255)         # 黄 / Yellow for connection lines
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.6
_FONT_COLOR = (255, 255, 255)


class CalibrationCollector:
    """管理单次标定的点收集与交互 / Manages point collection and interaction for a single calibration."""

    def __init__(self, window_name: str, image: np.ndarray, title: str, point_color: Tuple[int, int, int]):
        self.window_name = window_name
        self.image_original = image.copy()
        self.image_display = image.copy()
        self.title = title
        self.point_color = point_color
        self.points: List[Tuple[float, float]] = []  # 收集的点击点 (x, y) / Collected click points

    @staticmethod
    def _on_mouse(event, x, y, flags, param):
        """鼠标回调: 左键添加点, 右键删除最近点 / Mouse callback: left-click to add, right-click to undo."""
        if event == cv2.EVENT_LBUTTONDOWN:
            collector = param
            collector.points.append((float(x), float(y)))
            collector._redraw()

        elif event == cv2.EVENT_RBUTTONDOWN:
            collector = param
            if collector.points:
                collector.points.pop()
                collector._redraw()

    def _redraw(self):
        """重绘画面: 底图 + 标定点 + 序号标签 + 连线 / Redraw: base image + points + indices + connecting lines."""
        self.image_display = self.image_original.copy()
        # 绘制已点击的点与序号 / Draw collected points with indices
        for i, (px, py) in enumerate(self.points):
            cv2.circle(self.image_display, (int(px), int(py)), _POINT_RADIUS, self.point_color, -1)
            cv2.putText(
                self.image_display, str(i + 1), (int(px) + 8, int(py) - 8),
                _FONT, _FONT_SCALE, _FONT_COLOR, 2, cv2.LINE_AA,
            )
        # 用线连接标定点, 便于观察顺序 / Connect points with lines for visual clarity
        if len(self.points) >= 2:
            pts_arr = np.array([[int(p[0]), int(p[1])] for p in self.points], dtype=np.int32)
            cv2.polylines(self.image_display, [pts_arr], isClosed=False, color=_LINE_COLOR, thickness=2)

        # 叠加标题与操作提示 / Overlay title and usage hints
        hint_lines = [
            f'{self.title} | 已选 {len(self.points)} 点 / Selected',
            '[左键] 添加 / Add  [右键] 撤销 / Undo  [Enter] 确认 / Confirm  [Esc] 退出 / Quit',
        ]
        y0 = 25
        for line in hint_lines:
            cv2.putText(self.image_display, line, (10, y0), _FONT, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(self.image_display, line, (10, y0), _FONT, 0.55, _FONT_COLOR, 1, cv2.LINE_AA)
            y0 += 22

        cv2.imshow(self.window_name, self.image_display)

    def collect(self, min_points: int = 4) -> List[Tuple[float, float]]:
        """交互式收集点, 返回点击坐标列表 / Interactive point collection, returns list of (x, y) coordinates."""
        self._redraw()
        cv2.setMouseCallback(self.window_name, self._on_mouse, self)

        print(f'  [{self.title}] 请点击至少 {min_points} 个对应点, 按 Enter 确认, Esc 退出 / '
              f'Click at least {min_points} corresponding points, Enter to confirm, Esc to quit')

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter 键确认 / Enter to confirm
                if len(self.points) < min_points:
                    print(f'  ! 需要至少 {min_points} 个点, 当前 {len(self.points)} / '
                          f'Need at least {min_points} points, got {len(self.points)}')
                    continue
                print(f'  ✓ 确认 {len(self.points)} 个点 / Confirmed {len(self.points)} points')
                break
            elif key == 27:  # Esc 退出 / Esc to quit
                print('  ✗ 用户取消 / User cancelled')
                return []

        return self.points


def compute_homography(src_pts: List[Tuple[float, float]],
                       dst_pts: List[Tuple[float, float]]) -> Tuple[np.ndarray, float]:
    """使用 RANSAC 计算透视变换矩阵并返回重投影误差 / Compute homography via RANSAC and return reprojection error."""
    src = np.array(src_pts, dtype=np.float32).reshape(-1, 1, 2)
    dst = np.array(dst_pts, dtype=np.float32).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        raise RuntimeError('Homography 计算失败, 请检查对应点是否合理 / Homography computation failed, check point correspondences')

    # 计算重投影误差, 评估标定质量 / Compute reprojection error to assess calibration quality
    projected = cv2.perspectiveTransform(src, H)
    errors = np.linalg.norm(projected - dst, axis=2).flatten()
    mean_error = float(np.mean(errors))
    return H, mean_error


def calibrate_camera():
    # 加载 camera 截图 / Load camera frame
    camera_image = cv2.imread(opt.camera_image)
    if camera_image is None:
        raise FileNotFoundError(f'无法读取 camera 截图 / Cannot read camera image: {opt.camera_image}')
    cam_h, cam_w = camera_image.shape[:2]

    # 加载 2D 平面图 / Load floor plan image
    plane_image = cv2.imread(opt.plane_image)
    if plane_image is None:
        raise FileNotFoundError(f'无法读取平面图 / Cannot read plane image: {opt.plane_image}')
    plane_h, plane_w = plane_image.shape[:2]

    print(f'Camera 截图: {cam_w}x{cam_h} | 平面图: {plane_w}x{plane_h} / '
          f'Camera frame: {cam_w}x{cam_h} | Floor plan: {plane_w}x{plane_h}')
    print(f'标定相机: {opt.camera_name} / Calibrating camera: {opt.camera_name}')
    print('-' * 50)

    # Step 1: 在 camera 截图上点击标定点 / Collect points on camera image
    cam_collector = CalibrationCollector(
        window_name='Camera Image - Click Points',
        image=camera_image,
        title=f'Camera: {opt.camera_name}',
        point_color=_POINT_COLOR_CAMERA,
    )
    cam_points = cam_collector.collect(min_points=4)
    if not cam_points:
        cv2.destroyAllWindows()
        sys.exit(0)
    cv2.destroyWindow('Camera Image - Click Points')

    # Step 2: 在平面图上点击对应点 (顺序必须一致) / Collect corresponding points on plane (same order)
    plane_collector = CalibrationCollector(
        window_name='Floor Plan - Click Corresponding Points',
        image=plane_image,
        title='2D Floor Plan (按相同顺序点击 / Click in same order)',
        point_color=_POINT_COLOR_PLANE,
    )
    plane_points = plane_collector.collect(min_points=4)
    if not plane_points:
        cv2.destroyAllWindows()
        sys.exit(0)
    cv2.destroyWindow('Floor Plan - Click Corresponding Points')

    # Step 3: 计算 Homography / Compute homography
    H, reproj_error = compute_homography(cam_points, plane_points)

    # 统计 inlier 数量 / Count inliers (within RANSAC threshold)
    src = np.array(cam_points, dtype=np.float32).reshape(-1, 1, 2)
    dst = np.array(plane_points, dtype=np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(src, H)
    errors = np.linalg.norm(projected - dst, axis=2).flatten()
    inliers = int(np.sum(errors < 5.0))

    print('-' * 50)
    print(f'Homography 矩阵 (3×3):\n{H}')
    print(f'重投影误差均值: {reproj_error:.4f} px | Inliers: {inliers}/{len(cam_points)} / '
          f'Mean reprojection error: {reproj_error:.4f} px | Inliers: {inliers}/{len(cam_points)}')

    # Step 4: 保存标定结果 / Save calibration result
    calib_data = {
        'camera_name': opt.camera_name,
        'image_width': cam_w,
        'image_height': cam_h,
        'plane_width': plane_w,
        'plane_height': plane_h,
        'homography_matrix': H.tolist(),
        'reprojection_error_px': round(reproj_error, 4),
        'num_correspondences': len(cam_points),
        'inliers': inliers,
        'correspondences': [
            {'image_xy': [cam_points[i][0], cam_points[i][1]],
             'plane_xy': [plane_points[i][0], plane_points[i][1]]}
            for i in range(len(cam_points))
        ],
    }

    with open(opt.output, 'w', encoding='utf-8') as f:
        json.dump(calib_data, f, indent=2, ensure_ascii=False)
    print(f'\n标定结果已保存 / Calibration saved to: {opt.output}')

    # 可选: 打开预览窗口, 展示变换后的叠加效果 / Optional: display overlay preview
    if opt.show_preview:
        cam_preview = cv2.resize(camera_image, (min(cam_w, 640), min(cam_h, 480)))
        plane_preview = cv2.resize(plane_image, (min(plane_w, 640), min(plane_h, 480)))
        cv2.imshow('Camera Points', cam_preview)
        cv2.imshow('Floor Plan Points', plane_preview)
        print('按任意键关闭预览 / Press any key to close preview')
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='交互式相机标定工具 / Interactive Camera Calibration Tool')
    parser.add_argument('--camera_name', type=str, required=True,
                        help='Camera 标识 / Camera identifier, e.g. camera_A')
    parser.add_argument('--camera_image', type=str, required=True,
                        help='Camera 截图路径 / Camera frame image path')
    parser.add_argument('--plane_image', type=str, required=True,
                        help='2D 平面图路径 / Floor plan image path')
    parser.add_argument('--output', type=str, required=True,
                        help='Homography JSON 输出路径 / Output JSON path')
    parser.add_argument('--show_preview', action='store_true',
                        help='标定完成后展示预览窗口 / Show preview after calibration')

    opt = parser.parse_args()
    calibrate_camera()
