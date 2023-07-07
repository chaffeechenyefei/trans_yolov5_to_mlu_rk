训练: python3.8 train.py --config ./motor/config_v5s_conv.yaml

测试: python3.8 test.py --config ./motor/config_v5s_conv.yaml

推理: python3.8 detect.py --weights /path/weights.pt --source /path/imgs_dir --save_dir /path/save_dir --img_size 736 416 --conf_thres 0.2 --iou_thres 0.3


# I. RKNN量化
## 项目页面
https://ushare.ucloudadmin.com/pages/viewpage.action?pageId=119935816


## 步骤：
### 1) pt模型转onnx
```commandline
python models/export_onnx_for_rknn.py --model_key yolov5s-conv-9-20211104 --mode 0(or 4)
```
* 需要在pytorch版本较高的docker环境下运行，可以导出nn.SiLU并且替换为x*sigmoid(x)
* --model_key 需要在config.py中事先配置好
* 目前模型用的地址在config.py中配置
* 目前仅支持mode 0,4, 其中0不支持fast precompile, 4支持fast precompile, 但是, 需要额外传入anchors的数值. 分析原因, 应该是anchors是内部FP16的数值, precompile没有考虑这种情况.

### 2) onnx模型转rknn
```commandline
python export_rknn.py --model_key yolov5s-conv-9-20211104 --mode 0(or 4)
```
* 需要在rknn-toolkit-1.7.1版本的docker下运行
* rknn模型存在./rknn/目录下
* 可以通过--img导入测试图片，对比输出boxes的数量判断效果，如果不设置则使用随机数

### 3) onnx模型转rknn load加速
```commandline
python export_rknn.py --model_key yolov5s-conv-9-20211104 --mode 4 --fast 
```
* 如果在2）中验证成功，则可以生成precompile的模型，在C++上加载时，速度明显变快，其他无差别

## 适配RKNN_v1.7.1对yolov5的修改
1) yolo.py: 在class Detect的forward中，加入self.export的前置判断，如果设置export，则触发rknn模型下的输出
 
2) yolo.py: 直接采用如下方式进行取数赋值，存在问题。rknn导入onnx时，会报错：scatter_ND的问题，实际是数组下标的问题。属于通用问题。
```commandline
y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]
```
正确方式如下， 通过分段方式赋值给不同的变量，不进行自身的赋值
```commandline
xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]
wh = (y[..., 2:4] * 2) ** 2 * anchor_grid
conf = y[..., 4:]
```
3) yolo.py: 不能直接使用```self.anchor_grid[]```，否则会报如下错，判断是数组维度被修改了，通过间接的方式获取就没问题了。属于rknn独有问题。
```commandline
Dimensions must be equal, but are 52 and 3 for 'Mul_Mul_242_18/Mul_Mul_242_18' (op: 'Mul') with input shapes: [1,3,52,92,2], [2,1,3,1,1].
```

4) yolo.py: 输出的结果不能合并，只能分3个head输出，如果合并会报如下错误：属于rknn独有问题，有时会报CONCAT维度错误。
```commandline
E [op_optimize:428]CONCAT, uid 3 must have same quantize parameter!
```
```commandline
xys = torch.cat(xys,dim=1)
whs = torch.cat(whs,dim=1)
# xywhs = torch.cat([xys,whs], dim=-1) # if doing so: E [op_optimize:428]CONCAT, uid 3 must have same quantize parameter!
confs = torch.cat(confs,dim=1)
```

# II. mlu量化
## 项目页面
https://ushare.ucloudadmin.com/pages/viewpage.action?pageId=113783103

## 步骤：
### 1) pt转pth
需要在pytorch1.9.0以上版本的环境中运行
```commandline
python export_pth_for_mlu --model_key yolov5s-conv-9-20211104
```
将得到的模型传输到mlu环境的机器中； 以后可以在mlu环境的机器中部署一个高版本的pytorch，减少传输

### 2) pth转int8模型
需要在带mlu环境的docker中运行
```commandline
python mlu_yolov5.py --mlu 0 --quantization 1 --half_input 1 --data ./data/images/ --batch_size 2 --w 736 --h 416 --model_key yolov5s-conv-9-20211104
```

### 3) int8模型转cambricon离线模型
分别产出mlu270与mlu220的模型
```commandline
python genoff.py -fake_device 0 -input_format 0 -half_input 1 -in_width 736 -in_height 416 -model yolov5s-conv-9-20211104 -core_number 1
python genoff.py -fake_device 1 -mcore MLU220 -input_format 0 -half_input 1 -in_width 736 -in_height 416 -model yolov5s-conv-9-20211104 -core_number 1
```

## 适配mlu对yolov5的修改

1) models/yolo.py Detect class中增加export_mode=10，支持寒武纪的导出方式

2) api_for_mlu.py 增加通过config文件的模型加载

3) models.experimental中的attempt_load，可以选择是否进行fuse，寒武纪模式下不进行fuse

4) 注释class AutoShape

# III. yolov5-face的量化



