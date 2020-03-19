#!/usr/bin/env python
"""


2020-03-03 09:08:28
"""
import torch
import torch.nn as nn
from addict import Dict
import utils

def train(models, iterator, optimizers, loss_fun, device, train_args, model_args, elapsed_epochs, log_this_epoch=False):

    batch_size = train_args["batch_size"]
    noise_dim = model_args["noise_dim"]
    disc_step = train_args["num_D_accumulations"]

    epoch_losses = Dict({"d": 0, "g": 0, "matching": 0})
    models.g.train()
    models.d.train()
    iterator_len = len(iterator)
    disc_noise = torch.randn((iterator_len, disc_step, batch_size, noise_dim)).to(device)
    gen_noise = torch.randn((iterator_len, batch_size, noise_dim)).to(device)

    for idx, sample in enumerate(iterator):
        losses = Dict({"d": 0, "g": 0, "matching": 0})
        x = sample["metos"] #.to(device)
        y = sample["real_imgs"] #.to(device)

        # update discriminator
        optimizers.d.zero_grad()
        for k in range(disc_step):
            #noise = torch.randn(x.shape[0], noise_dim).to(device)
            noise = disc_noise[idx, k, :x.shape[0]]
            y_hat = models.g(x, noise)
            losses.d += models.d.compute_loss(y, 1) + models.d.compute_loss(y_hat, 0)

        losses.d.backward()
        total_steps = elapsed_epochs * iterator_len + idx
        optimizers.d = utils.optim_step(
            optimizers.d, train_args["optimizer"], total_steps, idx
        )

        # update generator
        optimizers.g.zero_grad()
        #noise = torch.randn(x.shape[0], noise_dim).to(device)
        noise = gen_noise[idx, :x.shape[0]]
        y_hat = models.g(x, noise)
        losses.g = models.d.compute_loss(y_hat, 1)
        losses.matching = loss_fun(y_hat, y)
        (losses.g + losses.matching).backward()

        optimizers.g = utils.optim_step(
            optimizers.g, train_args["optimizer"], total_steps, idx
        )

        if log_this_epoch:
            for k in epoch_losses.keys():
                epoch_losses[k] += losses[k].item() / len(iterator)
        else:
            epoch_losses = None

    return models, epoch_losses, optimizers
