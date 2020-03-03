#!/usr/bin/env python
"""


2020-03-03 09:08:28
"""
import torch
import torch.nn as nn
from addict import Dict

def train(models, iterator, optimizers, loss_fun, device, noise_dim=2, disc_step=1):
    epoch_losses = Dict({"d": 0, "g": 0, "matching": 0})
    models.g.train()
    models.d.train()

    for sample in iterator:
        losses = Dict({"d": 0, "g": 0, "matching": 0})
        x = sample["metos"].to(device)
        y = sample["real_imgs"].to(device)

        # update discriminator
        optimizers.d.zero_grad()
        for k in range(disc_step):
            noise = torch.randn(x.shape[0], noise_dim).to(device)
            y_hat = models.g(x, noise)
            losses.d += models.d.compute_loss(y, 1) + models.d.compute_loss(y_hat, 0)

        losses.d.backward()
        optimizers.d.step()

        # update generator
        optimizers.g.zero_grad()
        noise = torch.randn(x.shape[0], noise_dim).to(device)
        y_hat = models.g(x, noise)
        losses.g = models.d.compute_loss(y_hat, 1)
        losses.matching = loss_fun(y_hat, y)
        (losses.g + losses.matching).backward()
        optimizers.g.step()

        for k in epoch_losses.keys():
            epoch_losses[k] += losses[k].item() / len(iterator)

    return models, epoch_losses
