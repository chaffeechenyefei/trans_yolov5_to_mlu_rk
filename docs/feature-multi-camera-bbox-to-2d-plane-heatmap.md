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
## [new] 1. 生成2d heatmap plane的计划
### 1.1 现状与期望
- camera_A `data/videos/rtmart-001.mp4` 与 `data/videos/rtmart-002.mp4` 是同一个摄像机不同时刻的video
- camera_B `data/videos/rtmart-003.mp4` 与 `data/videos/rtmart-004.mp4` 是另一个摄像机不同时刻的video
- `data/videos/rtmart-001.mp4` 与 `data/videos/rtmart-003.mp4` 都是 20 点开始的 5min数据
- `data/videos/rtmart-002.mp4` 与 `data/videos/rtmart-004.mp4` 都是 19 点开始的 5min数据
- 他们对应的bbox在 [result目录下](../data/result)
- 具体bbox的使用请参考 [yolo检测迭代记录](../detect_video.md), 和合成代码 - [bbox_video_synth](../bbox_video_synth.py)
- 需要分别将camera A 和 B, 与2D平面进行点的映射标定
