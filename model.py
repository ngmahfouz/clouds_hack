#!/usr/bin/env python
"""


2020-03-03 08:52:16
"""
import torch
from pathlib import Path
import torch.nn as nn

class Deconvolver(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.n_out = n_out
        self.n_in = n_in

        self.initial_linear = nn.Linear(n_in, n_in * n_out ** 2)
        self.up1 = UNetModule(n_in, 3 * n_in)
        self.up2 = UNetModule(3 * n_in, 9 * n_in)
        self.up3 = UNetModule(9 * n_in, 27 * n_in)
        self.up4 = UNetModule(27 * n_in, n_in)
        self.conv_final = nn.Conv2d(n_in, 1, kernel_size=1)

    def forward(self, x):
        x_ = self.initial_linear(x)
        x_ = x_.reshape(x.shape[0], self.n_in, self.n_out, self.n_out)
        layers = nn.Sequential(
            self.up1,
            self.up2,
            self.up3,
            self.up4,
            self.conv_final
        )
        return layers(x_)


class UNetModule(nn.Module):
    """
    One of the "triple layer" blocks in https://arxiv.org/pdf/1505.04597.pdf
    """
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv1 = nn.Conv2d(n_in, n_out, 3, padding=1)
        self.conv2 = nn.Conv2d(n_out, n_out, 3, padding=1)
        self.activation = nn.ReLU()
        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, x):
        layers = nn.Sequential(
            self.conv1, self.bn, self.activation,
            self.conv2, self.bn, self.activation
        )
        return layers(x)

