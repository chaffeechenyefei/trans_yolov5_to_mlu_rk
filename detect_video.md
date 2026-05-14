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

# CLI
```shell
python detect_video.py --weights weights/yolov5s-conv-head-20220121.pt --source data/videos/rtmart-001.mp4 --save_dir data/result --img_size 736 416 --conf_thres 0.5 --iou_thres 0.3 --device cpu --sample_fps 1
```

```shell
python detect_video.py --weights weights/yolov5s-conv-head-20220121.pt --source data/videos/rtmart-002.mp4 --save_dir data/result --img_size 736 416 --conf_thres 0.5 --iou_thres 0.3 --device cpu --sample_fps 25

python detect_video.py --weights weights/yolov5s-conv-head-20220121.pt --source data/videos/rtmart-003.mp4 --save_dir data/result --img_size 736 416 --conf_thres 0.5 --iou_thres 0.3 --device cpu --sample_fps 25

python detect_video.py --weights weights/yolov5s-conv-head-20220121.pt --source data/videos/rtmart-004.mp4 --save_dir data/result --img_size 736 416 --conf_thres 0.5 --iou_thres 0.3 --device cpu --sample_fps 25
```