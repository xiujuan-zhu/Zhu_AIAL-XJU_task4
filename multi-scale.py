import torch
from torch import nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


def shuffle(x, groups):
    N, C, H, W = x.size()
    out = x.view(N, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)
    return out


class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class SplAtConv2d(nn.Module):
    """Split-Attention Conv2d
    """
    def __init__(self, in_channels, channels, kernel_size=(3, 3), stride=1, padding=(0, 0),
                 dilation=(1, 1), groups=1, bias=True,
                 radix=2,  norm_layer=True,
                 dropblock_prob=0.0, **kwargs):
        super(SplAtConv2d, self).__init__()
        padding = _pair(padding)
        inter_channels = channels//2
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob
        self.conv = nn.Conv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                               groups=groups*radix, bias=bias, **kwargs)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = nn.BatchNorm2d(channels*radix)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(inter_channels)
        self.fc2 = nn.Conv2d(inter_channels, channels*radix, 1, groups=self.cardinality)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            if torch.__version__ < '1.5':
                splited = torch.split(x, int(rchannel//self.radix), dim=1)
            else:
                splited = torch.split(x, rchannel//self.radix, dim=1)
            gap = sum(splited)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)

        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)

        if self.radix > 1:
            if torch.__version__ < '1.5':
                attens = torch.split(atten, int(rchannel//self.radix), dim=1)
            else:
                attens = torch.split(atten, rchannel//self.radix, dim=1)
            out = sum([att*split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x
        return out.contiguous()


class msblock_h(nn.Module):

    def __init__(self, inplans, planes, conv_kernels=[3, 5], stride=1, conv_groups=[1, 4]):
        super(msblock_h, self).__init__()
        self.conv_1 = SplAtConv2d(inplans, planes, kernel_size=conv_kernels[0], padding=conv_kernels[0] // 2,
                           stride=stride, groups=conv_groups[0])
        self.conv_2 = SplAtConv2d(inplans, planes, kernel_size=(conv_kernels[1], conv_kernels[0]),
                           padding=(conv_kernels[1] // 2, conv_kernels[0] // 2),
                           stride=stride, groups=conv_groups[1])
        self.conv1x1 = nn.Conv2d(2*planes, planes, 1, 1)

    def forward(self, x):
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        feats1 = torch.cat((x1, x2), dim=1)
        out = shuffle(feats1, groups=4)
        out = self.conv1x1(out)
        return out


class Resblock_h(nn.Module):
    def __init__(self, inplanes, planes, stride=1, conv_kernels=[3, 5], conv_groups=[1, 4]):
        super(Resblock_h, self).__init__()
        self.conv1 = msblock_h(inplanes, planes, stride=stride, conv_kernels=conv_kernels, conv_groups=conv_groups)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = out + identity
        out = self.relu(out)
        return out


class Highlayers(nn.Module):
    def __init__(self):
        super(Highlayers, self).__init__()
        self.conv5 = Resblock_h(128, 128, stride=1, conv_kernels=[3, 5], conv_groups=[1,1])
        self.conv6 = Resblock_h(128, 128, stride=1, conv_kernels=[3, 5], conv_groups=[1, 1])
        self.conv7 = Resblock_h(128, 128, stride=1, conv_kernels=[3, 5], conv_groups=[1, 1])
        self.pool3 = nn.AvgPool2d([1, 2])
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x5 = self.drop(self.pool3(self.conv5(x)))
        x6 = self.drop(self.pool3(self.conv6(x5)))
        x7 = self.drop(self.pool3(self.conv7(x6)))
        return x7


class msblock_l(nn.Module):
    def __init__(self, inplans, planes, conv_kernels=[3, 5], stride=1, conv_groups=[1, 1]):
        super(msblock_l, self).__init__()
        self.conv_1 = SplAtConv2d(inplans, planes, kernel_size=conv_kernels[0], padding=(conv_kernels[0]//2, conv_kernels[0]//2),
                            stride=stride, groups=conv_groups[0])
        self.conv_2 = SplAtConv2d(inplans, planes, kernel_size=conv_kernels[1], padding=(conv_kernels[1]//2, conv_kernels[1]//2),
                            stride=stride, groups=conv_groups[1])
        self.conv1x1 = nn.Conv2d(2*planes, planes, 1, 1)

    def forward(self, x):
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        feats1 = torch.cat((x1, x2), dim=1)
        out = shuffle(feats1, groups=4)
        out = self.conv1x1(out)
        return out


class Resblock_l(nn.Module):

    def __init__(self, inplanes, planes, stride=1, conv_kernels=[3, 5], conv_groups=[1, 4]):
        super(Resblock_l, self).__init__()
        self.conv1 = msblock_l(inplanes, planes, stride=stride, conv_kernels=conv_kernels, conv_groups=conv_groups)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = msblock_l(planes, planes, stride=stride, conv_kernels=conv_kernels, conv_groups=conv_groups)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.shortcut = nn.Sequential(
            nn.Conv2d(inplanes, planes, (1, 1), stride=stride, bias=False),
            nn.BatchNorm2d(planes),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        identity = self.shortcut(x)
        out = out + identity

        out = self.relu(out)
        return out


class pre_cnn(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(pre_cnn, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channel, out_channel, (3, 3), (1, 1), (1, 1))
        self.bn3x3 = nn.BatchNorm2d(out_channel)
        self.conv5x5 = nn.Conv2d(in_channel, out_channel, (5, 5), (1, 1), (2, 2))
        self.bn5x5 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv1x1 = nn.Conv2d(32, 16, (1, 1), (1, 1))

    def forward(self, x):
        x_3x3 = self.relu(self.bn3x3(self.conv3x3(x)))
        x_5x5 = self.relu(self.bn5x5(self.conv5x5(x)))
        feats1 = torch.cat((x_3x3, x_5x5), dim=1)
        out = shuffle(feats1, 4)
        out = self.conv1x1(out)
        return out


class Lowlayers(nn.Module):
    def __init__(self):
        super(Lowlayers, self).__init__()
        self.conv_pre = pre_cnn(in_channel=1, out_channel=16)
        self.conv2 = Resblock_l(16, 32, stride=1, conv_kernels=[3, 5], conv_groups=[1, 1])
        self.conv3 = Resblock_l(32, 64, stride=1,  conv_kernels=[3, 5], conv_groups=[1, 1])
        self.conv4 = Resblock_l(64, 128, stride=1,  conv_kernels=[3, 5], conv_groups=[1, 1])
        self.pool1 = nn.AvgPool2d([1, 2])
        self.pool2 = nn.AvgPool2d([2, 2])
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x1 = self.drop(self.pool1(self.conv_pre(x)))
        x2 = self.drop(self.pool1(self.conv2(x1)))
        x3 = self.drop(self.pool2(self.conv3(x2)))
        x4 = self.drop(self.pool2(self.conv4(x3)))
        return x4


class mssanet(nn.Module):
    def __init__(self):
        super(mssanet, self).__init__()
        self.lowlayers = Lowlayers()
        self.highlayers = Highlayers()

    def forward(self, x):
        x_l = self.lowlayers(x)
        x_h = self.highlayers(x_l)
        return x_h


# a = torch.randn(2, 1, 4, 128)
# ms = mssanet()
# output = ms(a)
# print(ms)

