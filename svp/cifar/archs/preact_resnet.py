'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class PreActBlock(nn.Module):
#     '''Pre-activation version of the BasicBlock.'''
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1):
#         super(PreActBlock, self).__init__()
#         self.bn1 = nn.BatchNorm2d(in_planes)
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(x))
#         shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
#         out = self.conv1(out)
#         out = self.conv2(F.relu(self.bn2(out)))
#         out += shortcut
#         return out


# class PreActBottleneck(nn.Module):
#     '''Pre-activation version of the original Bottleneck module.'''
#     expansion = 4

#     def __init__(self, in_planes, planes, stride=1):
#         super(PreActBottleneck, self).__init__()
#         self.bn1 = nn.BatchNorm2d(in_planes)
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(x))
#         shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
#         out = self.conv1(out)
#         out = self.conv2(F.relu(self.bn2(out)))
#         out = self.conv3(F.relu(self.bn3(out)))
#         out += shortcut
#         return out


# class PreActResNet(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=10):
#         super(PreActResNet, self).__init__()
#         self.in_planes = 64

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.linear = nn.Linear(512*block.expansion, num_classes)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = F.avg_pool2d(out, 4)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out


# def PreActResNet18():
#     return PreActResNet(PreActBlock, [2,2,2,2])

# def PreActResNet20(numClasses):
#     return PreActResNet(PreActBlock, [], num_classes=numClasses)

# def PreActResNet34():
#     return PreActResNet(PreActBlock, [3,4,6,3])

# def PreActResNet50():
#     return PreActResNet(PreActBottleneck, [3,4,6,3])

# def PreActResNet101():
#     return PreActResNet(PreActBottleneck, [3,4,23,3])

# def PreActResNet152():
#     return PreActResNet(PreActBottleneck, [3,8,36,3])


# def test():
#     net = PreActResNet18()
#     y = net((torch.randn(1,3,32,32)))
#     print(y.size())

# test()

import math
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride != 1 or inplanes != (planes * self.expansion):
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion, 1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, inputs):
        H = self.conv1(inputs)
        H = self.bn1(H)
        H = F.relu(H)

        H = self.conv2(H)
        H = self.bn2(H)

        H += self.shortcut(inputs)
        outputs = F.relu(H)

        return outputs


class StochasticBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, survival_rate=1):
        super().__init__()
        self.survival_rate = survival_rate
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.increasing = inplanes != (planes * self.expansion)
        if self.increasing:
            assert ((1. * planes * self.expansion) / inplanes) == 2
        if stride != 1:
            self.shortcut = nn.Sequential(nn.AvgPool2d(stride))
        else:
            self.shortcut = nn.Sequential()

    def forward(self, inputs):
        shortcut = self.shortcut(inputs)
        if self.increasing:
            shortcut = torch.cat([shortcut] + [shortcut.mul(0)], 1)

        if not self.training or torch.rand(1)[0] <= self.survival_rate:
            H = self.conv1(inputs)
            H = self.bn1(H)
            H = F.relu(H)

            H = self.conv2(H)
            H = self.bn2(H)

            if self.training:
                H /= self.survival_rate
            H += shortcut
        else:
            H = shortcut
        outputs = F.relu(H)

        return outputs


class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1,
                               bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)

        self.increasing = stride != 1 or inplanes != (planes * self.expansion)
        if self.increasing:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion, 1, stride=stride,
                          bias=False)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, inputs):
        H = self.bn1(inputs)
        H = F.relu(H)
        if self.increasing:
            inputs = H
        H = self.conv1(H)

        H = self.bn2(H)
        H = F.relu(H)
        H = self.conv2(H)

        H += self.shortcut(inputs)
        return H


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        if stride != 1 or inplanes != (planes * self.expansion):
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion, 1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, inputs):
        H = self.conv1(inputs)
        H = self.bn1(H)
        H = F.relu(H)

        H = self.conv2(H)
        H = self.bn2(H)
        H = F.relu(H)

        H = self.conv3(H)
        H = self.bn3(H)

        H += self.shortcut(inputs)
        outputs = F.relu(H)

        return outputs


class ResNeXtBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, cardinality=32,
                 base_width=4):
        super().__init__()

        width = math.floor(planes * (base_width / 64.0))

        self.conv1 = nn.Conv2d(inplanes, width * cardinality, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * cardinality)

        self.conv2 = nn.Conv2d(width * cardinality, width * cardinality, 3,
                               groups=cardinality, padding=1, stride=stride,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(width * cardinality)

        self.conv3 = nn.Conv2d(width * cardinality, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        if stride != 1 or inplanes != (planes * self.expansion):
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion, 1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, inputs):
        H = self.conv1(inputs)
        H = self.bn1(H)
        H = F.relu(H)

        H = self.conv2(H)
        H = self.bn2(H)
        H = F.relu(H)

        H = self.conv3(H)
        H = self.bn3(H)

        H += self.shortcut(inputs)
        outputs = F.relu(H)

        return outputs


class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, stride=stride,
                               bias=False)

        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)

        self.increasing = stride != 1 or inplanes != (planes * self.expansion)
        if self.increasing:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion, 1, stride=stride,
                          bias=False)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, inputs):
        H = self.bn1(inputs)
        H = F.relu(H)
        if self.increasing:
            inputs = H
        H = self.conv1(H)

        H = self.bn2(H)
        H = F.relu(H)
        H = self.conv2(H)

        H = self.bn3(H)
        H = F.relu(H)
        H = self.conv3(H)

        H += self.shortcut(inputs)
        return H


class ResNet(nn.Module):

    def __init__(self, Block, layers, filters, num_classes=10, inplanes=None):
        self.inplanes = inplanes or filters[0]
        super().__init__()

        self.pre_act = 'Pre' in Block.__name__

        self.conv1 = nn.Conv2d(3, self.inplanes, 3, padding=1, bias=False)
        if not self.pre_act:
            self.bn1 = nn.BatchNorm2d(self.inplanes)

        self.num_sections = len(layers)
        for section_index, (size, planes) in enumerate(zip(layers, filters)):
            section = []
            for layer_index in range(size):
                if section_index != 0 and layer_index == 0:
                    stride = 2
                else:
                    stride = 1
                section.append(Block(self.inplanes, planes, stride=stride))
                self.inplanes = planes * Block.expansion
            section = nn.Sequential(*section)
            setattr(self, f'section_{section_index}', section)

        if self.pre_act:
            self.bn1 = nn.BatchNorm2d(self.inplanes)

        self.fc = nn.Linear(filters[-1] * Block.expansion, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')  # noqa: E501
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, inputs):
        H = self.conv1(inputs)

        if not self.pre_act:
            H = self.bn1(H)
            H = F.relu(H)

        for section_index in range(self.num_sections):
            H = getattr(self, f'section_{section_index}')(H)

        if self.pre_act:
            H = self.bn1(H)
            H = F.relu(H)

        H = F.avg_pool2d(H, H.size()[2:])
        H = H.view(H.size(0), -1)
        outputs = self.fc(H)

        return outputs


class StochasticResNet(ResNet):

    def __init__(self, Block, layers, filters, num_classes=10, inplanes=None,
                 min_survival_rate=1.0, decay='linear'):
        super().__init__(Block, layers, filters,
                         num_classes=num_classes,
                         inplanes=inplanes)
        L = sum(layers)
        curr = 1
        for section_index in range(self.num_sections):
            section = getattr(self, f'section_{section_index}')
            for name, module in section.named_children():
                if decay == 'linear':
                    survival_rate = 1 - ((curr / L) * (1 - min_survival_rate))
                elif decay == 'uniform':
                    survival_rate = min_survival_rate
                else:
                    raise NotImplementedError(
                        f"{decay} decay has not been implemented.")
                module.survival_rate = survival_rate
                curr += 1
        assert (curr - 1) == L


# From "Deep Residual Learning for Image Recognition"
def ResNet20(num_classes=10):
    return ResNet(BasicBlock, layers=[3] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)


def ResNet32(num_classes=10):
    return ResNet(BasicBlock, layers=[5] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)


def ResNet44(num_classes=10):
    return ResNet(BasicBlock, layers=[7] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)


def ResNet56(num_classes=10):
    return ResNet(BasicBlock, layers=[9] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)


def ResNet110(num_classes=10):
    return ResNet(BasicBlock, layers=[18] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)


def ResNet1202(num_classes=10):
    return ResNet(BasicBlock, layers=[200] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)


# From "Identity Mappings in Deep Residual Networks"
def PreActResNet110(num_classes=10):
    return ResNet(PreActBlock, layers=[18] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)


def PreActResNet164(num_classes=10):
    return ResNet(PreActBottleneck, layers=[18] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)


def PreActResNet1001(num_classes=10):
    return ResNet(PreActBottleneck, layers=[111] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)


# Based on but not in "Identity Mappings in Deep Residual Networks"
def PreActResNet8(num_classes=10):
    return ResNet(PreActBlock, layers=[1] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)


def PreActResNet14(num_classes=10):
    return ResNet(PreActBlock, layers=[2] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)


def PreActResNet20(num_classes=10):
    return ResNet(PreActBlock, layers=[3] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)


def PreActResNet56(num_classes=10):
    return ResNet(PreActBlock, layers=[9] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)


def PreActResNet164Basic(num_classes=10):
    return ResNet(PreActBlock, layers=[27] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)


# From "Deep Networks with Stochastic Depth"
def StochasticResNet110(num_classes=10):
    return StochasticResNet(StochasticBlock, layers=[18] * 3,
                            filters=[16, 32, 64], min_survival_rate=0.5,
                            decay='linear', num_classes=num_classes)


def StochasticResNet1202(num_classes=10):
    return StochasticResNet(StochasticBlock, layers=[200] * 3,
                            filters=[16, 32, 64], min_survival_rate=0.5,
                            decay='linear', num_classes=num_classes)


# From "Deep Networks with Stochastic Depth" for SVHN Experiments
def ResNet152SVHN(num_classes=10):
    return ResNet(BasicBlock, layers=[25] * 3, filters=[16, 32, 64],
                  num_classes=num_classes)


def StochasticResNet152SVHN(num_classes=10):
    return StochasticResNet(StochasticBlock, layers=[25] * 3,
                            filters=[16, 32, 64], min_survival_rate=0.5,
                            decay='linear', num_classes=num_classes)


# Based on but not in "Deep Networks for Stochastic Depth"
def StochasticResNet56(num_classes=10):
    return StochasticResNet(StochasticBlock, layers=[9] * 3,
                            filters=[16, 32, 64], min_survival_rate=0.5,
                            decay='linear', num_classes=num_classes)


def StochasticResNet56_08(num_classes=10):
    return StochasticResNet(StochasticBlock, layers=[9] * 3,
                            filters=[16, 32, 64], min_survival_rate=0.8,
                            decay='linear', num_classes=num_classes)


# From "Wide Residual Networks"
def WRN(n, k, num_classes=10):
    assert (n - 4) % 6 == 0
    base_filters = [16, 32, 64]
    filters = [num_filters * k for num_filters in base_filters]
    d = (n - 4) / 2  # l = 2
    return ResNet(PreActBlock, layers=[int(d / 3)] * 3, filters=filters,
                  inplanes=16, num_classes=num_classes)


def WRN_40_4(num_classes=10):
    return WRN(40, 4, num_classes=num_classes)


def WRN_16_4(num_classes=10):
    return WRN(16, 4, num_classes=num_classes)


def WRN_16_8(num_classes=10):
    return WRN(16, 8, num_classes=num_classes)


def WRN_28_10(num_classes=10):
    return WRN(28, 10, num_classes=num_classes)


# From "Aggregated Residual Transformations for Deep Neural Networks"
def ResNeXt29(cardinality, base_width, num_classes=10):
    Block = partial(ResNeXtBottleneck, cardinality=cardinality,
                    base_width=base_width)
    Block.__name__ = ResNeXtBottleneck.__name__
    Block.expansion = ResNeXtBottleneck.expansion
    return ResNet(Block, layers=[3, 3, 3], filters=[64, 128, 256],
                  num_classes=num_classes)


# From kunagliu/pytorch
def ResNet18(num_classes=10):
    return ResNet(BasicBlock, layers=[2, 2, 2, 2], filters=[64, 128, 256, 512],
                  num_classes=num_classes)


def ResNet34(num_classes=10):
    return ResNet(BasicBlock, layers=[3, 4, 6, 3], filters=[64, 128, 256, 512],
                  num_classes=num_classes)


def ResNet50(num_classes=10):
    return ResNet(Bottleneck, layers=[3, 4, 6, 3], filters=[64, 128, 256, 512],
                  num_classes=num_classes)


def ResNet101(num_classes=10):
    return ResNet(Bottleneck,
                  layers=[3, 4, 23, 3], filters=[64, 128, 256, 512],
                  num_classes=num_classes)


def ResNet152(num_classes=10):
    return ResNet(Bottleneck,
                  layers=[3, 8, 36, 3], filters=[64, 128, 256, 512])
