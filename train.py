#!/usr/bin/env python
"""


2020-03-03 09:08:28
"""
import torch
import torch.nn as nn

def train(model, iterator, optimizer, loss_fun, device, discriminator, noise_dim=2, is_training_discriminator=True):
    epoch_loss = 0
    model.train()

    for sample in iterator:
        optimizer.zero_grad()
        x = sample["metos"].to(device)
        noise = torch.randn(x.shape[0], noise_dim).to(device)
        y = sample["real_imgs"].to(device)

        y_hat = model(x, noise)

        if is_training_discriminator:
            loss = discriminator.compute_loss(y, 1) + discriminator.compute_loss(y_hat, 0)
            model_to_return = discriminator
        else:
            loss = discriminator.compute_loss(y_hat, 1) + loss_fun(y_hat, y)
            model_to_return = model

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return model_to_return, epoch_loss / len(iterator)
