#!/usr/bin/env python
"""


2020-03-03 09:08:28
"""
import torch
import torch.nn as nn

def train(models, iterator, optimizers, loss_fun, device, noise_dim=2, disc_step=1, is_training_discriminator=True):
    epoch_loss = Dict({"d": 0, "g": 0, "matching": 0})
    model.train()

    for sample in iterator:
        x = sample["metos"].to(device)
        noise = torch.randn(x.shape[0], noise_dim).to(device)
        y = sample["real_imgs"].to(device)
        y_hat = models.g(x, noise)


        # update discriminator
        losses = Dict({"d": 0, "g": 0, "matching": 0})
        optimizers.d.zero_grad()
        for k in range(disc_step):
            losses.d += models.d.compute_loss(y, 1) + models.d.compute_loss(y_hat, 0)

        losses.d.backward()
        optimizers.d.step()

        # update generator
        optimizers.g.zero_grad()
        losses.g = discriminator.compute_loss(y_hat, 1)
        losses.matching = loss_fun(y_hat, y)
        (losses.g + losses.matching).backward()
        optimizers.g.step()

        for k in epoch_losses.keys():
            epoch_losses[k] += losses[k].item() / len(iterator)

    return models, epoch_losses
