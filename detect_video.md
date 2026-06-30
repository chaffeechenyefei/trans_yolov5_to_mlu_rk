# Modification Rules
- 每次改动(增加代码), 代码中必须要有注释, 中英文双语都需要
- 改动结束后, 需要判断是否更新README.md
- [thinking]或者<thinking></thinking>中的内容, 是我希望你(coding agent)自己考虑并作出判断的内容, 这里之所以特别标注出来, 是为了提示你进行判断, 因为我也不确定哪个方案更好
- 改动结束后, 如果发现本文档中其他部分与现有内容有冲突, 请修改本文档其他部分的描述, 确保二者相互匹配

## 状态更新规则
- 一般只需要对[new]或者<new></new>中的内容进行改动, 改动结束后, 标记为[modified]或者对应的<modified></modified>, 并根据实际改动内容, 对[new]或者<new></new>进行改动, 主要是对修改的说明,便于之后coding agent理解. 即:完成实现后，请将本文档各段落前缀从 `[new]` 改为 `[modified]`（可按 section 级别更新）。

---

# Task
对应代码[detect_video.py](./detect_video.py)
从视频流抽frame,进行检测,检测结果存为某个新的视频流

# Requirements
- 参考[detect.py](./detect.py)代码文件, 我已经跑通这个文件了, 使用以下命令测试无误
```shell
python detect.py --weights weights/yolov5s-conv-head-20220121.pt --source data/images --save_dir data/result --img_size 736 416 --conf_thres 0.2 --iou_thres 0.3 --device cpu
```
- 需要参考上述代码改造成对视频流的, 输入某段视频流, 输出一段视频流
- 检测时, 支持ctrl+c进行中断, 中断后不影响已有结果的保存
- 入参上继续支持 `--weights --img_size --conf_thres --iou_thres --device`
- 支持抽frame的fps设置, 默认1fps
- 检测时可以显示实时fps
- 不要改动项目中其他代码, 可以新建代码, 出发点是保证兼容性
- 在命令行显示剩余video中尚未处理的frame的百分比, 或者已处理的frame的百分比
- [modified] 已增加减小结果文件体积的方案: 支持 `--codec` 编码器选择(默认 `auto`, 自动回退 `avc1/H264/mp4v/XVID`)；支持 `--output_scale` 输出分辨率缩放(如 `0.75` / `0.5`)。两者可叠加使用以进一步减小文件。

## Incremental Requirement
较大改动会在这里, 每个section可以理解为一个较大的需求

### [modified] HeatMap
- 已支持可选热力图输出: 通过 `--enable_heatmap` 开启。未开启时保持原有行为不变(仅输出检测结果视频)。
- 热力图由每帧多个bbox累计得到: bbox区域按置信度累加，位置重叠越多/置信度越高，颜色越红。
- 已支持时间衰减: 通过 `--heat_decay_seconds` 控制衰减窗口，默认 `60` 秒；超过窗口的历史贡献权重为0。
- 已支持叠加透明度: 通过 `--heatmap_alpha` 控制热力图与原视频alpha叠放强度，默认 `0.35`。

### [modified] bbox 文件输出
- 已支持通过 `--bbox_output` CLI flag 开启"仅 bbox"模式: 该模式下不再输出视频文件, 仅输出 `<video_name>_bbox.txt`。
- 输出格式(每行 7 个字段, 逗号分隔, 文件首行为 `#` 注释 header):
  ```
  # frame_id, cls_id, x, y, w, h, conf
  ```
  其中:
  - `frame_id` = 该帧在**原视频**中的整数索引(并非采样后的连续序号), 即 `frame_id % frame_interval == 0` 的那些帧。
  - `x, y` = bbox 在**原画面分辨率**下的 top-left 坐标(已通过 `scale_coords` 还原到 `im0` 尺寸)。
  - `w, h` = bbox 的宽与高, `w = x2 - x1`, `h = y2 - y1`。
  - `conf` = 检测置信度, 保留 6 位小数。
- 该模式下不会执行 bbox 绘制、resize、热力图、视频编码、`cv2.imshow` 等任何与画面合成相关的操作, 推理吞吐更高。
- 与 `--enable_heatmap` 互斥: 同时设置时, `--bbox_output` 优先(heatmap 在 bbox-only 模式下不生效)。
- 目的是, 将来可以将 bbox 和原始 video 再次合成带 bbox 的 video, 文件体积远小于 in-line 绘制的检测结果视频。
- 中断(Ctrl+C)安全性: `finally` 中显式 `flush + close` bbox 文件句柄, 已写入的 bbox 行一定落盘。

## [modified] bbox 文件 + video 合成
- 已新增独立脚本 [bbox_video_synth.py](./bbox_video_synth.py), 将 `detect_video.py --bbox_output` 产出的 bbox 文本与原始视频合成带 bbox 的结果视频, 便于直接查看。
- 不依赖 torch / 权重, 纯 OpenCV 写入, 可在轻量环境下运行, 不会修改项目中其它代码。
- bbox 解析严格遵循 `detect_video.py` 的输出格式 `# frame_id, cls_id, x, y, w, h, conf`:
  - `frame_id` 视为原视频整数帧索引, 与 `--sample_fps` 配合计算 `frame_interval = round(src_fps / sample_fps)`;
  - `(x, y, w, h)` 视为 top-left + 宽高, 还原为 `(x1, y1, x2 = x+w, y2 = y+h)` 后直接在原画面绘制。
- **关键约束**: 调用本脚本时 `--sample_fps` 必须与生成 bbox 时使用的值一致, 否则 bbox 会被错位投影到错误的帧。
- 非采样帧(即 bbox 文件中没有对应行的中间帧)处理策略, 通过 `--non_sampled_strategy` 控制:
  - `hold_last`(默认): 沿用最近一次采样帧的 bbox, 视觉上呈现 "跟踪" 效果;
  - `passthrough`: 不绘制任何 bbox, 仅输出原画面。
- 复用 `detect_video.py` 的 `_create_video_writer` / `_make_even`, 输出编码器与分辨率缩放行为与 `detect_video.py` 完全一致, 便于跨脚本一致。
  - 当 `detect_video.py` 不在同目录时, 脚本内自带 fallback, 仍可独立运行。
- 可选 `--conf_thres` 二次过滤 bbox、`--label` 控制画板上的文字(`none`/`conf`/`cls`/`cls_conf`)、`--names` 提供类名映射(如 `"0:head,1:person"`)。
- 支持 Ctrl+C 中断, 已写入的视频帧仍会落盘(`finally` 中显式 `cap.release()` + `writer.release()`)。
- 命令行同样输出进度百分比与平均 FPS, 与 `detect_video.py` 体感一致。

# CLI
- 第一次测试
```shell
python detect_video.py --weights weights/yolov5s-conv-head-20220121.pt --source data/videos/rtmart-001.mp4 --save_dir data/result --img_size 736 416 --conf_thres 0.5 --iou_thres 0.3 --device cpu --sample_fps 1
```

- 编解码压缩输出

head detection
```shell
python detect_video.py --weights weights/yolov5s-conv-head-20220121.pt --source data/videos/rtmart-001.mp4 --save_dir data/result --img_size 736 416 --conf_thres 0.5 --iou_thres 0.3 --device cpu --sample_fps 25 --codec auto --output_scale 0.75

python detect_video.py --weights weights/yolov5s-conv-head-20220121.pt --source data/videos/rtmart-002.mp4 --save_dir data/result --img_size 736 416 --conf_thres 0.5 --iou_thres 0.3 --device cpu --sample_fps 25 --codec auto --output_scale 0.75

python detect_video.py --weights weights/yolov5s-conv-head-20220121.pt --source data/videos/rtmart-003.mp4 --save_dir data/result --img_size 736 416 --conf_thres 0.5 --iou_thres 0.3 --device cpu --sample_fps 25 --codec auto --output_scale 0.75

python detect_video.py --weights weights/yolov5s-conv-head-20220121.pt --source data/videos/rtmart-004.mp4 --save_dir data/result --img_size 736 416 --conf_thres 0.5 --iou_thres 0.3 --device cpu --sample_fps 25 --codec auto --output_scale 0.75
```

people detection
```shell
python detect_video.py --weights weights/yolov5s-people.pt --source data/videos/rtmart-001.mp4 --save_dir data/result --img_size 736 416 --conf_thres 0.5 --iou_thres 0.3 --device cpu --sample_fps 25 --codec auto --output_scale 0.75

python detect_video.py --weights weights/yolov5s-people.pt --source data/videos/rtmart-002.mp4 --save_dir data/result --img_size 736 416 --conf_thres 0.5 --iou_thres 0.3 --device cpu --sample_fps 25 --codec auto --output_scale 0.75

# HeatMap(60秒衰减 + alpha叠加)
python detect_video.py --weights weights/yolov5s-people.pt --source data/videos/rtmart-001.mp4 --save_dir data/result --img_size 736 416 --conf_thres 0.5 --iou_thres 0.3 --device cpu --sample_fps 25 --codec auto --output_scale 0.75 --enable_heatmap --heat_decay_seconds 60 --heatmap_alpha 0.35
```


head detection + heatmap
```shell
python detect_video.py --weights weights/yolov5s-conv-head-20220121.pt --source data/videos/rtmart-001.mp4 --save_dir data/result --img_size 736 416 --conf_thres 0.5 --iou_thres 0.3 --device cpu --sample_fps 25 --codec auto --output_scale 0.75 --enable_heatmap

python detect_video.py --weights weights/yolov5s-conv-head-20220121.pt --source data/videos/rtmart-002.mp4 --save_dir data/result --img_size 736 416 --conf_thres 0.5 --iou_thres 0.3 --device cpu --sample_fps 25 --codec auto --output_scale 0.75 --enable_heatmap
```

people detection + heatmap
```shell
python detect_video.py --weights weights/yolov5s-people.pt --source data/videos/rtmart-001.mp4 --save_dir data/result --img_size 736 416 --conf_thres 0.5 --iou_thres 0.3 --device cpu --sample_fps 25 --codec auto --output_scale 0.75 --enable_heatmap

python detect_video.py --weights weights/yolov5s-people.pt --source data/videos/rtmart-002.mp4 --save_dir data/result --img_size 736 416 --conf_thres 0.5 --iou_thres 0.3 --device cpu --sample_fps 25 --codec auto --output_scale 0.75 --enable_heatmap

python detect_video.py --weights weights/yolov5s-people.pt --source data/videos/rtmart-003.mp4 --save_dir data/result --img_size 736 416 --conf_thres 0.5 --iou_thres 0.3 --device cpu --sample_fps 25 --codec auto --output_scale 0.75 --enable_heatmap

python detect_video.py --weights weights/yolov5s-people.pt --source data/videos/rtmart-004.mp4 --save_dir data/result --img_size 736 416 --conf_thres 0.5 --iou_thres 0.3 --device cpu --sample_fps 25 --codec auto --output_scale 0.75 --enable_heatmap
```

bbox only 输出
```shell
# 仅输出 bbox 文件, 跳过视频写入, 用于后续与原视频合成 / Output bbox text only, skip video writing, re-composable later
python detect_video.py --weights weights/yolov5s-people.pt --source data/videos/rtmart-001.mp4 --save_dir data/result --img_size 736 416 --conf_thres 0.5 --iou_thres 0.3 --device cpu --sample_fps 25 --bbox_output

python detect_video.py --weights weights/yolov5s-people.pt --source data/videos/rtmart-002.mp4 --save_dir data/result --img_size 736 416 --conf_thres 0.5 --iou_thres 0.3 --device cpu --sample_fps 25 --bbox_output

python detect_video.py --weights weights/yolov5s-conv-head-20220121.pt --source data/videos/rtmart-001.mp4 --save_dir data/result --img_size 736 416 --conf_thres 0.5 --iou_thres 0.3 --device cpu --sample_fps 25 --bbox_output
```

bbox 文件 + video 合成
```shell
# 基础用法: 与 detect_video.py 同 sample_fps, 自动定位同目录下的 <video_name>_bbox.txt / Basic usage: auto-resolve bbox next to source
python bbox_video_synth.py --source data/videos/rtmart-001.mp4 --save_dir data/result --sample_fps 25 --codec auto --output_scale 0.75

# 显式指定类别名, 画面上叠加 "类名 + 置信度" 标签 / Provide class names so label shows "cls conf"
python bbox_video_synth.py --source data/videos/rtmart-001.mp4 --save_dir data/result --sample_fps 25 --codec auto --output_scale 0.75 --names "0:head" --label cls_conf

# head 检测的合成可视化, 同时叠加 frame_id 水印便于与 bbox txt 对照 / Head detection synthesis with frame_id watermark
python bbox_video_synth.py --source data/videos/rtmart-001.mp4 --save_dir data/result --sample_fps 25 --codec auto --output_scale 0.75 --names head --show_frame_id

# people 检测合成, 中间帧 bbox 全透明透传 / People synthesis with passthrough for non-sampled frames
python bbox_video_synth.py --source data/videos/rtmart-002.mp4 --save_dir data/result --sample_fps 25 --codec auto --output_scale 0.75 --names person --non_sampled_strategy passthrough
```