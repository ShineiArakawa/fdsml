"""ResNet in PyTorch.

License
-------

Copyright (c) 2026 Shinei Arakawa
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

This code is a derivative work based on code from the hirokatsukataoka16/FractalDB-Pretrained-ResNet-PyTorch repository, whose original code is licensed under the MIT License. The original code can be found at https://github.com/hirokatsukataoka16/FractalDB-Pretrained-ResNet-PyTorch.
"""

import typing

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import torch_utils.util as util

# ------------------------------------------------------------------------------------------------------------------------------

__all__ = ['ResNet', 'ResNetFeatureExtractor', 'ReductionType', 'resnet18', 'resnet34', 'resnet50']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


# ------------------------------------------------------------------------------------------------------------------------------
# Basic building blocks

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out

# ------------------------------------------------------------------------------------------------------------------------------
# Bottleneck


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_until_last_nonlinearlity(x)
        x = self.forward_last_nonlinearlity(x)
        return x

    def forward_until_last_nonlinearlity(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual

        return out

    def forward_last_nonlinearlity(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x)

# ------------------------------------------------------------------------------------------------------------------------------
# ResNet


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


class ReductionType(util.StrEnum):
    """Spatial feature reduction types."""

    gap = 'gap'  # Global Average Pooling


def global_average_pooling(x: torch.Tensor) -> torch.Tensor:
    """Global average pooling over spatial dimensions.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (B, C, H, W).

    Returns
    -------
    torch.Tensor
        Output tensor of shape (B, C).
    """

    return x.mean(dim=(-2, -1))


class ResNetFeatureExtractor(ResNet):
    """Feature extractor for ResNet with spatial feature reduction.
    This class extracts features in front of non-linearities at each bottleneck layer and applies spatial reduction to them.
    """

    def __init__(self, reduction: ReductionType = ReductionType.gap, **kwargs):
        """Initialize a ResNet feature extractor.

        Parameters
        ----------
        reduction : ReductionType, optional
            Type of spatial feature reduction to apply. Default is global average pooling.
        **kwargs
            Additional arguments passed to the ResNet initializer.
        """

        super().__init__(**kwargs)

        self.reduction = reduction
        if self.reduction == ReductionType.gap:
            self.reducer = global_average_pooling
        else:
            raise ValueError(f'Unknown reduction type: {self.reduction}')

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass that extracts features at multiple layers.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 3, H, W).

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary mapping feature names to their corresponding tensors.
        """

        feats = {}

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        def forward_and_store(layer: nn.Sequential, name: str, x_: torch.Tensor) -> torch.Tensor:
            # Forward until last non-linearity and extract features
            if len(layer) > 0:
                x_ = layer[:-1](x_)
            x_ = layer[-1].forward_until_last_nonlinearlity(x_)
            feat = x_.clone()
            x_ = layer[-1].forward_last_nonlinearlity(x_)

            # Apply reduction and store
            feat = self.reducer(feat)
            feats[name] = feat

            return x_

        # Forward until non-linearities and store features
        x = forward_and_store(self.layer1, 'layer1', x)
        x = forward_and_store(self.layer2, 'layer2', x)
        x = forward_and_store(self.layer3, 'layer3', x)
        x = forward_and_store(self.layer4, 'layer4', x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        feats['pre_fc'] = x.clone()

        x = self.fc(x)
        feats['logits'] = x

        return feats


# ------------------------------------------------------------------------------------------------------------------------------
# ResNet constructors


def resnet18(pretrained=False, resnet_cls: typing.Type[torch.nn.Module] = ResNet, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = resnet_cls(block=BasicBlock, layers=[2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, resnet_cls: typing.Type[torch.nn.Module] = ResNet, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = resnet_cls(block=BasicBlock, layers=[3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, resnet_cls: typing.Type[torch.nn.Module] = ResNet, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = resnet_cls(block=Bottleneck, layers=[3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

# ------------------------------------------------------------------------------------------------------------------------------
