#!/usr/bin/env python
"""


2020-03-03 08:52:16
"""
import torch
from pathlib import Path
import torch.nn as nn

class Deconvolver(nn.Module):
    def __init__(self, n_in, n_out, noise_dim=2):
        super().__init__()
        self.n_out = n_out
        self.n_in = n_in
        self.total_n_in = n_in + noise_dim

        self.initial_linear = nn.Linear(self.total_n_in, self.total_n_in * n_out ** 2)
        self.metos_batch_norm = nn.BatchNorm1d(n_in)
        self.up1 = UNetModule(self.total_n_in, 2 * self.total_n_in)
        self.up2 = UNetModule(2 * self.total_n_in, 4 * self.total_n_in)
        self.up3 = UNetModule(4 * self.total_n_in, 8 * self.total_n_in)
        self.up4 = UNetModule(8 * self.total_n_in, self.total_n_in)
        self.conv_final = nn.Conv2d(self.total_n_in, 1, kernel_size=1)

    def forward(self, x, noise):
        x = self.metos_batch_norm(x)
        x = torch.cat([x, noise], axis=1)
        x_ = self.initial_linear(x)
        x_ = x_.reshape(x.shape[0], self.total_n_in, self.n_out, self.n_out)
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

