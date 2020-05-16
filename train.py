#!/usr/bin/env python
"""


2020-03-03 09:08:28
"""
import torch
import torch.nn as nn
from addict import Dict
import utils

def train(models, iterator, optimizers, loss_fun, device, train_args, model_args, elapsed_epochs, feature_extraction=None, log_this_epoch=False):

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
        x = sample["metos"]#.to(device)
        y = sample["real_imgs"]#s.to(device)

        # update discriminator
        optimizers.d.zero_grad()
        for k in range(disc_step):
            #noise = torch.randn(x.shape[0], noise_dim).to(device)
            noise = disc_noise[idx, k, :x.shape[0]] #x.shape[0] represents the true batch_size (useful for the last batch especially)
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
        noise = gen_noise[idx, :x.shape[0]] #x.shape[0] represents the true batch_size (useful for the last batch especially)
        y_hat = models.g(x, noise)
        losses.g = models.d.compute_loss(y_hat, 1)
        if feature_extraction is None:
            losses.matching = loss_fun(y_hat, y)
        else:
            #import pdb
            rgb_y_hat = torch.cat([y_hat] * 3, dim=1) #converts grayscale to RGB by replicating the single channel 3 times
            rgb_y = torch.cat([y] * 3, dim=1)
            rgb_normalized_y_hat = []
            rgb_normalized_y = []
            for j in range(x.shape[0]):
                rgb_normalized_y_hat.append(feature_extraction["transformations"](rgb_y_hat[j].cpu()))
                rgb_normalized_y.append(feature_extraction["transformations"](rgb_y[j].cpu()))
            #pdb.set_trace()
            rgb_y_hat, rgb_y = torch.stack(rgb_normalized_y_hat).to(rgb_y_hat.device), torch.stack(rgb_normalized_y).to(rgb_y.device)
            feature_maps_y_hat = feature_extraction["extractor"](rgb_y_hat)
            feature_maps_y = feature_extraction["extractor"](rgb_y)
            losses.matching = loss_fun(feature_maps_y_hat, feature_maps_y)
        (train_args["lambda_gan"] * losses.g + train_args["lambda_L"] * losses.matching).backward()

        optimizers.g = utils.optim_step(
            optimizers.g, train_args["optimizer"], total_steps, idx
        )

        if log_this_epoch:
            for k in epoch_losses.keys():
                epoch_losses[k] += losses[k].item() / len(iterator)
        else:
            epoch_losses = None

    return models, epoch_losses, optimizers
