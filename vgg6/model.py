"""VGG6 for CIFAR-10 with a configurable activation function.

Same architecture as the course template: four 3x3 conv layers
[64, 64, M, 128, 128, M] (optionally with BatchNorm), global average pooling
and a single linear classifier (128 -> 10). ~0.26M parameters.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

VGG6_CFG = [64, 64, "M", 128, 128, "M"]

ACTIVATIONS = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "leaky_relu": lambda: nn.LeakyReLU(0.1),
    "elu": nn.ELU,
}


def make_layers(cfg, activation="relu", batch_norm=True):
    layers, in_ch = [], 3
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue
        layers.append(nn.Conv2d(in_ch, v, kernel_size=3, padding=1))
        if batch_norm:
            layers.append(nn.BatchNorm2d(v))
        layers.append(ACTIVATIONS[activation]())
        in_ch = v
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, features, num_classes=10, width=128):
        super().__init__()
        self.features = features
        self.classifier = nn.Linear(width, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(x)

    def _initialize_weights(self):
        # He-style init, as in the template.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0, math.sqrt(2.0 / n))
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


def vgg6(activation="relu", batch_norm=True, num_classes=10) -> VGG:
    if activation not in ACTIVATIONS:
        raise ValueError(f"unknown activation {activation!r}; choose from {list(ACTIVATIONS)}")
    return VGG(make_layers(VGG6_CFG, activation, batch_norm), num_classes)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
