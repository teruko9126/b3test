# https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# resblock is basicblock or bottleneck
class ResNet(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet, self).__init__()

        self.now_filters = 64
        self.conv1 = nn.Conv2d(# implement)

        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(# implement)

        self.layer2 = self._make_layer(# implement)

        self.layer3 = self._make_layer(# implement)

        self.layer4 = self._make_layer(# implement)

        self.linear = nn.Linear(# implement)

    def _make_layer(self, block, filters, num_blocks, first_stride):
        strides = [first_stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.now_filters, filters, stride))
            self.now_filters = filters * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # implement


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, now_filters, planes, stride):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(# implement)
        self.bn1 = nn.BatchNorm2d(# implement)
        
        self.conv2 = nn.Conv2d(
            # implement
        )
        self.bn2 = nn.BatchNorm2d(# implement)
        self.conv3 = nn.Conv2d(
            # implement
        )
        self.bn3 = nn.BatchNorm2d(# implement)

        self.shortcut = nn.Sequential()

        if stride != 1 or now_filters != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
# implement
                ),
                nn.BatchNorm2d(# implement),
            )

    def forward(self, x):
      # implement


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])