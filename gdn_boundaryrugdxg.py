import torch
from torch import nn
import torch.nn.functional as F
# from model/resnet import ResNet as models
# import model.resnet as models
# import resnet as models
# import resnet as models
from torch.nn.utils import weight_norm
import time
from math import exp
import numpy as np


class ConvBNReLU(nn.Module):
    '''Module for the Conv-BN-ReLU tuple.'''

    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation,
                 use_relu=True):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            c_in, c_out, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        if use_relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class CARAFE(nn.Module):
    def __init__(self, c, c_mid=64, scale=2, k_up=5, k_enc=3):
        """ The unofficial implementation of the CARAFE module.

        The details are in "https://arxiv.org/abs/1905.02188".

        Args:
            c: The channel number of the input and the output.
            c_mid: The channel number after compression.
            scale: The expected upsample scale.
            k_up: The size of the reassembly kernel.
            k_enc: The kernel size of the encoder.

        Returns:
            X: The upsampled feature map.
        """
        super(CARAFE, self).__init__()
        self.scale = scale

        self.comp = ConvBNReLU(c, c_mid, kernel_size=1, stride=1,
                               padding=0, dilation=1)
        self.enc = ConvBNReLU(c_mid, (scale * k_up) ** 2, kernel_size=k_enc,
                              stride=1, padding=k_enc // 2, dilation=1,
                              use_relu=False)
        self.pix_shf = nn.PixelShuffle(scale)

        self.upsmp = nn.Upsample(scale_factor=scale, mode='nearest')
        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale,
                                padding=k_up // 2 * scale)

    def forward(self, X):
        b, c, h, w = X.size()
        h_, w_ = h * self.scale, w * self.scale

        W = self.comp(X)  # b * m * h * w
        W = self.enc(W)  # b * 100 * h * w
        W = self.pix_shf(W)  # b * 25 * h_ * w_
        W = F.softmax(W, dim=1)  # b * 25 * h_ * w_

        X = self.upsmp(X)  # b * c * h_ * w_
        X = self.unfold(X)  # b * 25c * h_ * w_
        X = X.view(b, c, -1, h_, w_)  # b * 25 * c * h_ * w_

        X = torch.einsum('bkhw,bckhw->bchw', [W, X])  # b * c * h_ * w_
        return X


class conv3x3_resume(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()
        self.conv3x3 = Conv(nIn // 2, nIn // 2, kSize, 1, padding=1, bn_acti=True)
        self.conv1x1_resume = Conv(nIn // 2, nIn, 1, 1, padding=0, bn_acti=False)

    def forward(self, input):
        output = self.conv3x3(input)
        output = self.conv1x1_resume(output)
        return output

def Split(x):
    c = int(x.size()[1])
    c1 = round(c * 0.5)
    x1 = x[:, :c1, :, :].contiguous()
    x2 = x[:, c1:, :, :].contiguous()
    return x1, x2

class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output

class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output
# 计算一维的高斯分布向量
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
# 可以设定channel参数拓展为3通道
def create_window(window_size, channel=3):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


# 计算SSIM
# 直接使用SSIM的公式，但是在计算均值时，不是直接求像素平均值，而是采用归一化的高斯核卷积来代替。
# 在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
# 正如前面提到的，上面求期望的操作采用高斯核卷积代替。
def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 3
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


# from .common import ShiftMean


class ResBlock(nn.Module):
    def __init__(self, n_feats, expansion_ratio, res_scale=1.0, low_rank_ratio=0.75):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.pool = nn.AvgPool2d(2, 2)
        self.gw1 = weight_norm(nn.Conv2d(n_feats, int(n_feats * expansion_ratio / 2), kernel_size=1))
        self.gw2 = weight_norm(nn.Conv2d(n_feats, int(n_feats * expansion_ratio / 2), kernel_size=3, padding=1))
        self.re = nn.PReLU()
        self.module = nn.Sequential(
            weight_norm(nn.Conv2d(n_feats * expansion_ratio, int(n_feats * low_rank_ratio), kernel_size=3, padding=1)),
            weight_norm(nn.Conv2d(int(n_feats * low_rank_ratio), n_feats, kernel_size=3, padding=1))
        )

    def forward(self, x):
        gw1 = self.gw1(x)
        gw2 = self.gw2(x)
        gw = torch.cat([gw1, gw2], dim=1)
        gw = self.re(gw)
        return x + self.module(gw) * self.res_scale

class ResBlocklv(nn.Module):
    def __init__(self, n_feats, expansion_ratio, res_scale=1.0, low_rank_ratio=0.75):#64/6/1.0/0.75 #192=64*6/2
        super(ResBlocklv, self).__init__()
        self.res_scale = res_scale
        self.pool = nn.AvgPool2d(2, 2)

        self.gw1 = weight_norm(nn.Conv2d(n_feats, int(n_feats * expansion_ratio / 2), kernel_size=1))
        self.gw2 = weight_norm(nn.Conv2d(n_feats, int(n_feats * expansion_ratio / 2), kernel_size=3, padding=1))
        self.re = nn.PReLU()
        # self.module = nn.Sequential(
        #     weight_norm(nn.Conv2d(n_feats * expansion_ratio, int(n_feats * low_rank_ratio), kernel_size=3, padding=1)),
        #     weight_norm(nn.Conv2d(int(n_feats * low_rank_ratio), n_feats, kernel_size=3, padding=1))
        # )#384-->48; 48-->64
        self.module = nn.Sequential(
            weight_norm(nn.Conv2d(n_feats * expansion_ratio, n_feats, kernel_size=3, padding=1))
        )#384-->64

    def forward(self, x):
        gw1 = self.gw1(x)
        gw2 = self.gw2(x)
        gw = torch.cat([gw1, gw2], dim=1) #192+192
        gw = self.re(gw)
        return x + self.module(gw) * self.res_scale


class ResBlock_2(nn.Module):
    def __init__(self, n_feats, expansion_ratio, res_scale=1.0, low_rank_ratio=0.75):
        super(ResBlock_2, self).__init__()
        self.res_scale = res_scale
        self.pool = nn.AvgPool2d(2, 2)
        self.gw1 = weight_norm(nn.Conv2d(n_feats, int(n_feats * expansion_ratio / 2), kernel_size=1))
        self.gw2 = weight_norm(nn.Conv2d(n_feats, int(n_feats * expansion_ratio / 2), kernel_size=3, padding=1))
        self.re = nn.PReLU()
        self.module = nn.Sequential(
            weight_norm(nn.Conv2d(n_feats * expansion_ratio, int(n_feats * low_rank_ratio), kernel_size=3, padding=1)),
            weight_norm(nn.Conv2d(int(n_feats * low_rank_ratio), n_feats, kernel_size=3, padding=1))
        )

    def forward(self, x):
        x1 = self.pool(x)
        gw1 = self.gw1(x1)
        gw2 = self.gw2(x1)
        gw = torch.cat([gw1, gw2], dim=1)
        gw = self.re(gw)
        gw = self.module(gw) * self.res_scale
        gw = nn.functional.interpolate(gw, scale_factor=2, mode='bilinear', align_corners=True)
        return x + gw

class ResBlock_2lv(nn.Module):
    def __init__(self, n_feats, expansion_ratio, res_scale=1.0, low_rank_ratio=0.75, d=1):
        super(ResBlock_2lv, self).__init__()
        self.res_scale = res_scale
        self.pool = nn.AvgPool2d(2, 2)
        # self.dconv_right = Conv(n_feats, n_feats * expansion_ratio / 2, (dkSize, dkSize), 1,
        #                         padding=(1 * d, 1 * d), groups=n_feats // 4, dilation=(d * 2, d * 2), bn_acti=True)
        # self.dconv_left = Conv(n_feats, n_feats * expansion_ratio / 2, (dkSize, dkSize), 1, padding=(1, 1), groups=n_feats // 4, bn_acti=True)
        self.gw1 = weight_norm(nn.Conv2d(n_feats, int(n_feats * expansion_ratio / 2), kernel_size=1,dilation=(d *2,  d*2)))
        self.gw2 = weight_norm(nn.Conv2d(n_feats, int(n_feats * expansion_ratio / 2), kernel_size=3, padding=1))
        self.re = nn.PReLU()
        # self.module = nn.Sequential(
        #     weight_norm(nn.Conv2d(n_feats * expansion_ratio, int(n_feats * low_rank_ratio), kernel_size=3, padding=1)),
        #     weight_norm(nn.Conv2d(int(n_feats * low_rank_ratio), n_feats, kernel_size=3, padding=1))
        # )
        self.module = nn.Sequential(
            weight_norm(nn.Conv2d(n_feats * expansion_ratio, n_feats, kernel_size=3, padding=1)))
        self.CARAFE = CARAFE(n_feats, c_mid=64, scale=2, k_up=5, k_enc=3)

    def forward(self, x):
        x1 = self.pool(x) #torch.Size([8, 64, 64, 64])
        # print(x.size())
        gw1 = self.gw1(x1) #torch.Size([8, 384, 32, 32])
        # gw1 = self.dconv_right(x1)
        gw2 = self.gw2(x1)
        # print(gw2.size()) #torch.Size([8, 192, 32, 32])
        # gw2 = self.dconv_left(x1)
        gw = torch.cat([gw1, gw2], dim=1)
        gw = self.re(gw)
        gw = self.module(gw) * self.res_scale
        # print(gw.size)
        # gw = nn.functional.interpolate(gw, scale_factor=2, mode='bilinear', align_corners=True)
        gw = self.CARAFE(gw) #torch.Size([8, 64, 64, 64])
        # print(gw.size()) #torch.Size([8, 64, 64, 64])
        # print("WWWWWWWW")
        return x + gw

class Body_1(nn.Module):
    def __init__(self):
        super(Body_1, self).__init__()
        w_residual = [ResBlocklv(64, 6, 1.0, 0.75)
                      for _ in range(6)]
        self.module = nn.Sequential(*w_residual)

    def forward(self, x):
        x1 = self.module(x)
        return x + x1


class Body_2(nn.Module):
    def __init__(self):
        super(Body_2, self).__init__()
        w_residual = [ResBlock_2lv(64, 6, 1.0, 0.75)
                      for _ in range(6)]
        self.module = nn.Sequential(*w_residual)

    def forward(self, x):
        x1 = self.module(x)
        x2 = x + x1
        x3 = nn.functional.interpolate(x2, scale_factor=4, mode='nearest')
        return x3


class FFM(nn.Module):
    def __init__(self):
        super(FFM, self).__init__()
        self.module = nn.Sequential(
            weight_norm(nn.Conv2d(128, 128, kernel_size=1)),
            nn.PReLU(),
            weight_norm(nn.Conv2d(128, 128, kernel_size=1)),
            nn.PReLU()
        )

    def forward(self, x, y):
        x1 = torch.cat([x, y], dim=1)
        x1 = self.module(x1)
        return x1


class ResBlock_3(nn.Module):
    def __init__(self, n_feats):
        super(ResBlock_3, self).__init__()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = weight_norm(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=2, padding=1))
        self.pool3 = weight_norm(nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1))

    def forward(self, x):
        p1 = self.pool1(x)
        p2 = self.pool2(x)
        p3 = self.pool3(p2)
        return p1 + p3

class ResBlock_3lv(nn.Module):
    def __init__(self, n_feats, d=2, kSize=3, dkSize=3):
        super(ResBlock_3lv, self).__init__()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = weight_norm(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=2, padding=1))
        self.pool3 = weight_norm(nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1))
        self.conv3x3 = Conv(n_feats, n_feats // 2, kSize, 1, padding=1, bn_acti=True)
        self.dconv_left = Conv(n_feats // 4, n_feats // 4, (dkSize, dkSize), 1,
                              padding=(1, 1), groups=n_feats // 4, bn_acti=True)

        self.dconv_right = Conv(n_feats // 4, n_feats // 4, (dkSize, dkSize), 1,
                                padding=(1 * d, 1 * d), groups=n_feats // 4, dilation=(d, d), bn_acti=True)

        self.bn_relu_1 = BNPReLU(n_feats)
        self.conv3x3_resume = conv3x3_resume(n_feats, n_feats, (dkSize, dkSize), 1,
                                             padding=(1, 1), bn_acti=True)

    def forward(self, x):
        p1 = self.pool1(x)
        # print("WWWWWWWW")
        # print(p1.size())
        p2 = self.pool2(x)
        output = self.conv3x3(p2)
        x1, x2 = Split(output)
        letf = self.dconv_left(x1)
        right = self.dconv_right(x2)
        output = torch.cat((letf, right), 1)
        output = self.conv3x3_resume(output)
        p3 = self.pool3(output)
        # p3 = self.pool3(p2)
        return self.bn_relu_1(p1 + p3)


class Body_3(nn.Module):
    def __init__(self):
        super(Body_3, self).__init__()
        self.con = weight_norm(nn.Conv2d(3, 32, kernel_size=3, padding=1))
        w_residual = [ResBlock_3lv(32)
                      for _ in range(3)]
        self.module = nn.Sequential(*w_residual)

    def forward(self, x):
        x = self.con(x)
        x = self.module(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class GDN(nn.Module):
    def __init__(self, layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=19, zoom_factor=8, use_ppm=True,
                 criterion=nn.CrossEntropyLoss(ignore_index=255), pretrained=False):
        super(GDN, self).__init__()
        assert layers in [50, 101, 152]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.criterion = criterion
        self.boundary_loss_function = nn.L1Loss()
        self.layers = layers
        self.classses = classes

        head = [weight_norm(nn.Conv2d(32, 64, kernel_size=3, padding=1))]
        waist = [weight_norm(nn.ConvTranspose2d(64, 64, stride=2, kernel_size=3, padding=1, output_padding=1)),
                 nn.PReLU(),
                 weight_norm(nn.ConvTranspose2d(64, 64, stride=2, kernel_size=3, padding=1, output_padding=1)),
                 nn.PReLU()]
        boundary = [weight_norm(nn.Conv2d(64, 64, kernel_size=3, padding=1)),
                    nn.PReLU(),
                    weight_norm(nn.Conv2d(64, 64, kernel_size=3, padding=1)),
                    nn.PReLU(),
                    weight_norm(nn.Conv2d(64, 32, kernel_size=3, padding=1)),
                    nn.PReLU(),
                    weight_norm(nn.Conv2d(32, 16, kernel_size=3, padding=1)),
                    nn.PReLU(),
                    weight_norm(nn.Conv2d(16, 16, kernel_size=3, padding=1)),
                    nn.PReLU(),
                    weight_norm(nn.Conv2d(16, 1, kernel_size=3, padding=1)),
                    weight_norm(nn.Conv2d(1, 1, kernel_size=3, padding=1))]
        tail = [weight_norm(nn.Conv2d(128, 32, kernel_size=3, padding=1)),
                nn.PReLU(),
                weight_norm(nn.Conv2d(32, 20, kernel_size=3, padding=1)),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.PReLU(),
                weight_norm(nn.Conv2d(20, classes, kernel_size=3, padding=1)),
                weight_norm(nn.Conv2d(classes, classes, kernel_size=3, padding=1))]
        taillv = [weight_norm(nn.Conv2d(128, classes, kernel_size=3, padding=1))]
        seg1 = [weight_norm(nn.Conv2d(64, 32, kernel_size=3, padding=1)),
                nn.PReLU(),
                weight_norm(nn.Conv2d(32, 20, kernel_size=3, padding=1)),
                nn.PReLU(),
                weight_norm(nn.Conv2d(20, classes, kernel_size=3, padding=1)),
                weight_norm(nn.Conv2d(classes, classes, kernel_size=3, padding=1))]
        seg2 = [weight_norm(nn.Conv2d(128, 32, kernel_size=3, padding=1)),
                nn.PReLU(),
                weight_norm(nn.Conv2d(32, 20, kernel_size=3, padding=1)),
                nn.PReLU(),
                weight_norm(nn.Conv2d(20, classes, kernel_size=3, padding=1)),
                weight_norm(nn.Conv2d(classes, classes, kernel_size=3, padding=1))]
        # sr_head = [weight_norm(nn.Conv2d(30, 48, kernel_size=3, padding=1)),
        #            nn.PReLU(),
        #            weight_norm(nn.Conv2d(48, 64, kernel_size=3, padding=1)),
        #            nn.PReLU()]
        sr_head = [weight_norm(nn.Conv2d(32, 64, kernel_size=3, padding=1)),
                   nn.PReLU()]

        sr_body = [ResBlocklv(64, 6, 1.0, 0.75)
                   for _ in range(5)]
        sr_tail = [weight_norm(nn.Conv2d(64, 192, kernel_size=3, padding=1)),
                   nn.PReLU(),
                   weight_norm(nn.Conv2d(192, 192, kernel_size=3, padding=1)),
                   nn.PixelShuffle(8)]
        self.pre = Body_3()
        self.head = nn.Sequential(*head)
        self.body_1 = Body_1()
        self.body_2 = Body_2()
        self.ffm = FFM()
        self.waist = nn.Sequential(*waist)
        self.boundary = nn.Sequential(*boundary)
        self.tail = nn.Sequential(*tail)
        self.taillv = nn.Sequential(*taillv)
        self.seg1 = nn.Sequential(*seg1)
        self.seg2 = nn.Sequential(*seg2)
        # self.skip = nn.Sequential(*skip)
        self.sr_head = nn.Sequential(*sr_head)
        self.sr_body = nn.Sequential(*sr_body)
        self.sr_tail = nn.Sequential(*sr_tail)
        self.CARAFE = CARAFE(128, c_mid=128, scale=2, k_up=5, k_enc=3)

    def forward(self, x, y=None, z=None):
        img = x #torch.Size([8, 3, 512, 512])
        x = self.pre(x) #torch.Size([8, 30, 64, 64])  torch.Size([8, 32, 64, 64])
        # print(x.size())
        img_sr = self.sr_head(x)
        img_sr = self.sr_body(img_sr)
        img_sr = self.sr_tail(img_sr)
        # print(img_sr.size())
        # print("WWWWWWWWWW")
        x = self.head(x) #torch.Size([8, 54, 64, 64])  torch.Size([8, 64, 64, 64])
        # print(x.size())
        wai = self.waist(x) #torch.Size([8, 54, 256, 256]) torch.Size([8, 64, 256, 256])
        # print(wai.size())
        boundary = self.boundary(wai)  #torch.Size([8, 1, 256, 256])
        # print(boundary.size())
        body1 = self.body_1(x) # torch.Size([8, 54, 64, 64]) torch.Size([8, 64, 64, 64])
        # print(body1.size())
        segment1 = self.seg1(body1)
        # print(segment1.size())
        body2 = self.body_2(body1) #torch.Size([8, 54, 256, 256]) torch.Size([8, 64, 256, 256])
        # print(body2.size())
        ff = self.ffm(wai, body2) #torch.Size([8, 108, 256, 256])
        # print(ff.size()) #torch.Size([2, 128, 256, 256])
        segment2 = self.seg2(ff)
        # print(segment2.size())
        # ff = self.CARAFE(ff)
        # x = self.taillv(ff)
        x = self.tail(ff)
        # print(x.size()) #([8, 6, 512, 512])
        if self.training:
            # print(y)
            y1 = (F.interpolate(torch.unsqueeze(y.float(), dim=1), size=[64, 64], mode='nearest')).long() #size=[90, 120]
            # print(y1)
            y1 = torch.squeeze(y1, dim=1)
            # print(y1)
            y2 = (F.interpolate(torch.unsqueeze(y.float(), dim=1), size=[256, 256], mode='nearest')).long()#size=[360, 480]
            y2 = torch.squeeze(y2, dim=1)
            if z is None:
                z1 = (F.interpolate(torch.unsqueeze(y.float(), dim=1), size=[256, 512], mode='nearest')).long()
                # print(z1)
                # print(z)
                # z1 = (F.interpolate(torch.unsqueeze(z.float(), dim=1), size=[360, 480], mode='nearest')).long()
                z1 = torch.squeeze(z1, dim=1)
            # print("LWWWWWWWWWW")
            # z1 = (F.interpolate(torch.unsqueeze(z.float(), dim=1), size=[360, 480], mode='nearest')).long()
            # print(z)
            # z1 = (F.interpolate(torch.unsqueeze(z.float(), dim=1), size=[360, 480], mode='nearest')).long()
            # z1 = torch.squeeze(z1, dim=1)
            # y2 = (F.interpolate(y.float(), scale_factor=0.5)).long()
            main_loss0 = self.criterion(x, y)
            main_loss1 = self.criterion(segment1, y1)
            main_loss2 = self.criterion(segment2, y2)
            main_loss = main_loss0 + 0.125 * main_loss1 + 0.5 * main_loss2
            # aux_loss = 1 - ssim(img_sr, img) + 0.25 * self.boundary_loss_function(boundary, z1)  # If there is no GT of boundary information,it is recommended not to add boundary loss
            aux_loss = 1 - ssim(img_sr, img)   # If there is no GT of boundary information,it is recommended not to add boundary loss
            f = open('lossrugd.txt', 'a')
            f.write(str(aux_loss))
            f.write('\n')
            f.close()
            return x.max(1)[1], main_loss, aux_loss
        else:
            return x


class GDN1(nn.Module):
    def __init__(self, pretrained=False):
        super(GDN1, self).__init__()
        head = [weight_norm(nn.Conv2d(30, 54, kernel_size=3, padding=1))]
        tail1 = [nn.Upsample(scale_factor=4, mode='nearest'),
                 weight_norm(nn.Conv2d(108, 108, kernel_size=1)),
                 nn.PReLU(),
                 weight_norm(nn.Conv2d(108, 108, kernel_size=1)),
                 nn.PReLU()]
        tail2 = [weight_norm(nn.Conv2d(108, 32, kernel_size=3, padding=1)),
                 nn.PReLU(),
                 weight_norm(nn.Conv2d(32, 20, kernel_size=3, padding=1)),
                 nn.Upsample(scale_factor=2, mode='nearest'),
                 nn.PReLU(),
                 weight_norm(nn.Conv2d(20, 11, kernel_size=3, padding=1)),
                 weight_norm(nn.Conv2d(11, 11, kernel_size=3, padding=1))]
        self.pre = Body_3()
        self.head = nn.Sequential(*head)
        self.body_1 = Body_1()
        self.body_2 = Body_2()
        self.tail1 = nn.Sequential(*tail1)
        self.tail2 = nn.Sequential(*tail2)

    def forward(self, x, y=None):
        x = self.pre(x)
        x = self.head(x)
        x1 = self.body_1(x)
        x = self.body_2(x1, x)
        # x = self.upsample(x)
        x2 = self.tail1(x)
        x = self.tail2(x2)
        return x


if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    # input = torch.rand(4, 3, 473, 473).cuda()
    # model = PSPNet(layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=21, zoom_factor=1, use_ppm=True, pretrained=True).cuda()
    # model.eval()
    # print(model)
    # output = model(input)
    # print('PSPNet', output.size())
