#!/usr/bin/env python
"""


2020-03-03 08:52:16
"""
import torch
from pathlib import Path
import torch.nn as nn

class Deconvolver(nn.Module):
    def __init__(self, n_in, n_out, n_blocks=4, depth_increase_factor=2, noise_dim=2):
        super().__init__()
        self.n_out = n_out
        self.n_in = n_in
        self.total_n_in = n_in + noise_dim #takes into account both the metos (n_in=8) and the noise (noise_dim) by concatenating their respective vectors

        self.initial_linear = nn.Linear(self.total_n_in, self.total_n_in * n_out ** 2) # we want to generate images of size height=n_out x width=n_out
        self.metos_batch_norm = nn.BatchNorm1d(n_in)
        cblocks_list = []
        unet_input_channels = self.total_n_in
        for i in range(n_blocks):
            cblocks_list.append(UNetModule(unet_input_channels, depth_increase_factor * unet_input_channels))
            unet_input_channels*= 2
        self.conv_final = nn.Conv2d(unet_input_channels, 1, kernel_size=1)
        self.model = nn.ModuleList(cblocks_list + [self.conv_final])

    def forward(self, x, noise):
        x = self.metos_batch_norm(x)
        x = torch.cat([x, noise], axis=1)
        x_ = self.initial_linear(x)
        x_ = x_.reshape(x.shape[0], self.total_n_in, self.n_out, self.n_out)
        for layer in self.model:
            x_ = layer(x_)
        return x_


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

