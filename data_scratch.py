#!/usr/bin/env python
"""


2020-03-03 08:56:40
"""
from addict import Dict
from torch.utils.data import DataLoader
import data
import importlib
import model
import numpy as np
import torch
import torch.nn as nn
import train
from torchvision.utils import save_image
from res_discriminator import MultiDiscriminator, Discriminator

importlib.reload(data)
importlib.reload(model)
importlib.reload(train)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
opts = Dict({"lr": 1e-4, "n_epoch": 10001, "save_every" : 500, "noise_dim" : 2, "disc_step" : 1})

clouds = data.LowClouds("/scratch/sankarak/data/low_clouds/", 10)
loader = DataLoader(clouds, batch_size=10)
dec = model.Deconvolver(8, 128, opts["noise_dim"])
dec = dec.to(device)

disc = MultiDiscriminator(in_channels=1, device=device).to(device)
models = Dict({"g": dec, "d": disc})
optimizers = Dict({
    "g": torch.optim.Adam(dec.parameters(), lr=opts["lr"]),
    "d": torch.optim.Adam(disc.parameters(), lr= 0.01 * opts["lr"])
})

def infer(generator, x, noise_dim, device):
    noise = torch.randn(x.shape[0], noise_dim).to(device)
    return generator(x, noise)


def save_images(dec, loader, i, n_infer=5):
    for sample in loader:
        x = sample["metos"].to(device)
        y = sample["real_imgs"].to(device)
        y_mean = y.mean(0)
        print("Loss of the mean image : ", ((y_mean - y) ** 2).mean())

        save_image(y, f"original_imgs_{i}.png")
        save_image(y_mean, f"mean_imgs_{i}.png")
        y_hats = []
        for j in range(n_infer):
            y_hats.append(infer(dec, x, opts.noise_dim, device))

        y_hats = torch.cat(y_hats, axis=0)
        save_image(y_hats, f"predicted_imgs_{i}-{j}.png")


for i in range(opts.n_epoch):
    models, avg_loss = train.train(models, loader, optimizers, nn.MSELoss(), device)
    if i % opts.save_every == 0:
        save_images(dec, loader, i)
    print(f"Discriminator loss : {avg_loss.d} - Generator loss : {avg_loss.g} - Matching loss: {avg_loss.matching}")

torch.save(dec.state_dict(), "dec_test.pth")


# go through loader and look at y_hat
# save images of the y_hats
# do they look like clouds?????


# normalize the metos

# think through other architectures? how to make it a gan?

# launch job on full images (supervised)

