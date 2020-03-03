#!/usr/bin/env python
"""


2020-03-03 09:08:28
"""
import torch
import torch.nn as nn

def train(model, iterator, optimizer, loss_fun, device, noise_dim=2):
    epoch_loss = 0
    model.train()

    for sample in iterator:
        optimizer.zero_grad()
        x = sample["metos"].to(device)
        noise = torch.randn(x.shape[0], noise_dim).to(device)
        y = sample["real_imgs"].to(device)

        y_hat = model(x, noise)
        loss = loss_fun(y_hat, y)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return model, epoch_loss / len(iterator)
