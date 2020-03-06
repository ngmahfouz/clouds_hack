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
import argparse
import yaml

parser = argparse.ArgumentParser()

parser.add_argument("-o", "--output_dir", type=str, help="Where to save files", default=".")
parser.add_argument("-c", "--config_file", type=str, help="YAML configuration file", default="default_training_config.yaml")

args = parser.parse_args()

importlib.reload(data)
importlib.reload(model)
importlib.reload(train)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#opts = Dict({"lr": 1e-4, "n_epochs": 10001, "save_every" : 500, "noise_dim" : 2, "disc_step" : 1})

with open(args.config_file) as f:
    yaml_config = yaml.load(f, Loader=yaml.FullLoader)
    data_args = yaml_config['data']
    model_args = yaml_config["model"]
    val_args = yaml_config["val"]
    train_args = yaml_config["train"]

clouds = data.LowClouds("/scratch/sankarak/data/low_clouds/", 10)
loader = DataLoader(clouds, batch_size=10)
dec = model.Deconvolver(8, 128, model_args["noise_dim"])
dec = dec.to(device)

disc = MultiDiscriminator(in_channels=1, device=device).to(device)
models = Dict({"g": dec, "d": disc})
optimizers = Dict({
    "g": torch.optim.Adam(dec.parameters(), lr=train_args["lr_g"]),
    "d": torch.optim.Adam(disc.parameters(), lr=train_args["lr_d"])
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

        save_image(y, f"{args.output_dir}/original_imgs_{i}.png")
        save_image(y_mean, f"{args.output_dir}/mean_imgs_{i}.png")
        y_hats = []
        for j in range(n_infer):
            y_hats.append(infer(dec, x, model_args["noise_dim"], device))

        y_hats = torch.cat(y_hats, axis=0)
        save_image(y_hats, f"{args.output_dir}/predicted_imgs_{i}-{j}.png")


for i in range(train_args["n_epochs"]):
    models, avg_loss = train.train(models, loader, optimizers, nn.MSELoss(), device)
    if i % train_args["save_every_epochs"] == 0:
        save_images(dec, loader, i, train_args["n_infer"])
    print(f"Discriminator loss : {avg_loss.d} - Generator loss : {avg_loss.g} - Matching loss: {avg_loss.matching}")

torch.save(dec.state_dict(), f"{args.output_dir}/dec_test.pth")


# go through loader and look at y_hat
# save images of the y_hats
# do they look like clouds?????


# normalize the metos

# think through other architectures? how to make it a gan?

# launch job on full images (supervised)

