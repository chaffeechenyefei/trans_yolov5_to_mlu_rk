# YOLOv5 YOLO-specific modules

import argparse
import logging
import sys
from copy import deepcopy

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None


class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx/mlu export
    export_mode = 0

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        # print('initial Detect')
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.stride_ = []
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        # self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.anchor_grid = nn.Parameter(a.clone().view(self.nl, 1, -1, 1, 1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

        self.conv1x1 = nn.Conv2d(1,1,1,1,0, bias=False)
        self.conv1x1.weight = nn.Parameter(torch.ones(1,1,1,1).float())

        #mlu
        self.grid0=None
        self.grid1=None
        self.grid2=None
        self.grid3=None

    def get_gridN(self, i):
        return eval('self.grid{:d}'.format(i))

    def forward(self, x):
        # print('grid.size: ', self.nl, self.na)
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        print('--> Detect Layer:', self.training, self.export, self.export_mode )
        if self.export:
            """
            针对rknn输出的特殊处理
            """
            print('stride = ', self.stride)
            if self.export_mode == 0:
                print('--> export rknn mode 0')
                xys = []
                whs = []
                confs = []
                for i in range(self.nl):
                    x[i] = self.m[i](x[i])  # conv
                    # print(x[i].shape)
                    bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
                    x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
                    # Dimensions must be equal, but are 52 and 3 for 'Mul_Mul_242_18/Mul_Mul_242_18' (op: 'Mul') with input shapes: [1,3,52,92,2], [2,1,3,1,1].
                    # epxort rknn时，self.anchor_grid的shape会错乱
                    anchor_grid = self.anchor_grid[i].reshape(1,self.na,1,1,2)
                    if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                        # print('grid: ', nx, ny)
                        self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                    y = x[i].sigmoid()
                    # print(i, y.shape, self.grid[i].shape, self.stride[i].shape, self.anchor_grid[i].shape)
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]
                    wh = (y[..., 2:4] * 2) ** 2 * anchor_grid
                    conf = y[..., 4:]
                    # print(xy.shape,wh.shape,conf.shape)
                    xys.append(xy.reshape(bs,-1,2))
                    whs.append(wh.reshape(bs,-1,2))
                    confs.append(conf.reshape(bs,-1,self.nc+1))

                xys = torch.cat(xys,dim=1)
                whs = torch.cat(whs,dim=1)
                # xywhs = torch.cat([xys,whs], dim=-1) # if doing so: E [op_optimize:428]CONCAT, uid 3 must have same quantize parameter!
                confs = torch.cat(confs,dim=1)
                        # y = torch.cat([xy, wh, conf], dim=4)
                print('--> export')
                print('<-- Detect Layer:', self.training, self.export, self.export_mode)
                # return [xywhs,confs]
                return [xys,whs,confs]
            elif self.export_mode == 1:
                print('--> export rknn mode 1')
                xys = []
                whs = []
                confs = []
                anchor_grid = self.anchor_grid.reshape(self.nl,self.na,2);
                for i in range(self.nl):
                    x[i] = self.m[i](x[i])  # conv
                    # print(x[i].shape)
                    bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
                    x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
                    y = x[i].sigmoid()
                    # print(i, y.shape, self.grid[i].shape, self.stride[i].shape, self.anchor_grid[i].shape)
                    xy = y[..., 0:2]
                    wh = y[..., 2:4]
                    conf = y[..., 4:]
                    # print(xy.shape,wh.shape,conf.shape)
                    xys.append(xy.reshape(bs, -1, 2))
                    whs.append(wh.reshape(bs, -1, 2))
                    confs.append(conf.reshape(bs, -1, self.nc + 1))

                xys = torch.cat(xys, dim=1)
                whs = torch.cat(whs, dim=1)
                # xywhs = torch.cat([xys,whs], dim=-1) # if doing so: E [op_optimize:428]CONCAT, uid 3 must have same quantize parameter!
                confs = torch.cat(confs, dim=1)
                # y = torch.cat([xy, wh, conf], dim=4)
                print('--> export')
                print('<-- Detect Layer:', self.training, self.export, self.export_mode)
                # return [xywhs,confs]
                return [xys, whs, confs, anchor_grid]
            elif self.export_mode == 2:
                print('--> export rknn mode 2')
                ys = []
                anchor_grid = self.anchor_grid.reshape(self.nl, self.na, 2);
                for i in range(self.nl):
                    x[i] = self.m[i](x[i])  # conv
                    # print(x[i].shape)
                    bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
                    # x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
                    y = x[i].sigmoid()
                    # print(i, y.shape, self.grid[i].shape, self.stride[i].shape, self.anchor_grid[i].shape)
                    # y [bs, na*no, ny, nx]
                    ys.append(y)
                # y = torch.cat([xy, wh, conf], dim=4)
                print('--> export')
                print('<-- Detect Layer:', self.training, self.export, self.export_mode)
                # return [xywhs,confs]
                ys.append(anchor_grid)
                return ys
            elif self.export_mode == 3:
                print('--> export rknn mode 3')
                ys  = []
                anchor_grid = self.anchor_grid.reshape(1,1, 1,self.nl*self.na*2)
                anchor_grid = self.conv1x1(anchor_grid)
                for i in range(self.nl):
                    x_i = self.m[i](x[i])
                    ys.append(x_i)
                ys.append(anchor_grid)
                print('anchors: ', anchor_grid.view(-1))
                print('<-- Detect Layer:', self.training, self.export, self.export_mode)
                return ys
            elif self.export_mode == 4:
                print('--> export rknn mode 4')
                ys  = []
                anchor_grid = self.anchor_grid.reshape(self.nl, self.na, 2)
                for i in range(self.nl):
                    x_i = self.m[i](x[i])
                    ys.append(x_i)
                # ys.append(anchor_grid)
                print('**please copy the value to anchor!!!')
                print('**anchors: ', anchor_grid.view(-1))
                print('<-- Detect Layer:', self.training, self.export, self.export_mode)
                return ys
            elif self.export_mode == 10:
                print('--> export mlu')
                output = []
                for i in range(self.nl):
                    x[i] = self.m[i](x[i])  # conv
                    bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
                    x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2)  # .contiguous()
                    y = x[i].sigmoid()
                    # same problem in onnx ATT.
                    # y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid3) * self.stride_[i]  # xy
                    # y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    xy = (y[..., 0:2] * 2. - 0.5 + self.get_gridN(i)) * self.stride_[i]
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                    conf = y[..., 4:]
                    y = torch.cat([xy, wh, conf], dim=4)
                    output.append(y.view(bs, -1, self.no))
                return torch.cat(output, 1)

        print('--> normal')
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            # print(x[i].shape)
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            # print(x[i].shape)
            # print(self.training, self.export)
            if not self.training:  # inference
                # print('inference')
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    # print('grid: ', nx, ny)
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                    print(self.grid[i].shape)

                y = x[i].sigmoid()
                # print(i, y.shape, self.grid[i].shape, self.stride[i].shape, self.anchor_grid[i].shape)
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))
        print('<-- normal')
        print('<-- Detect Layer:', self.training, self.export)
        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model_yolov5x6(nn.Module):
    """
    input_sz = [w,h]
    """
    def __init__(self, cfg='models/hub/yolov5x6.yaml',input_sz=[1024,1024],stride=[8., 16., 32., 64] , ch=3, nc=None, anchors=None):
        super(Model_yolov5x6, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict
        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names

        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            m.stride_ = stride
            m.stride = torch.tensor(m.stride_)#torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            """
            只有注册成Parameter才能进行量化，否则torch变量在量化时会出现问题
            """
            m.grid0 = nn.Parameter(m._make_grid( int(input_sz[0]/stride[0]), int(input_sz[1]/stride[0])))
            m.grid1 = nn.Parameter(m._make_grid( int(input_sz[0]/stride[1]), int(input_sz[1]/stride[1])))
            if len(stride) > 2:
                m.grid2 = nn.Parameter(m._make_grid( int(input_sz[0]/stride[2]), int(input_sz[1]/stride[2])))
            if len(stride) > 3:
                m.grid3 = nn.Parameter(m._make_grid( int(input_sz[0]/stride[3]), int(input_sz[1]/stride[3])))
            self.stride =  torch.tensor(m.stride_)
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        # self.info()
        logger.info('')

    def forward(self, x, augment=False, profile=False):
        return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        # print('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def autoshape(self):  # add autoShape module
        return self

class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            # a = self.forward_once((x, profile))
            # print('a: ', a)
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        # print('once: ', len(x), self.save, len(y))
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        # print('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        print('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP,
                 C3, C3TR]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, C3TR]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
    # y = model(img, profile=True)

    # Tensorboard
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter()
    # print("Run 'tensorboard --logdir=models/runs' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(model.model, img)  # add model to tensorboard
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
