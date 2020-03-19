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
from preprocessing import ReplaceNans, get_transforms
from optim import get_optimizers

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

dataset_transforms = get_transforms(data_args)
clouds = data.LowClouds(data_args["path"], data_args["load_limit"], transform=dataset_transforms, device=device)
nb_images = len(clouds)
train_clouds, val_clouds = torch.utils.data.random_split(clouds, [nb_images - val_args["set_size"], val_args["set_size"]])
train_loader = DataLoader(train_clouds, batch_size=train_args["batch_size"], pin_memory=False)
val_loader = DataLoader(val_clouds, batch_size=train_args["batch_size"])
dec = model.Deconvolver(8, 128, model_args["noise_dim"])
dec = dec.to(device)

disc = MultiDiscriminator(in_channels=1, device=device).to(device)
models = Dict({"g": dec, "d": disc})
g_optimizer, d_optimizer = get_optimizers(models["g"], models["d"], Dict(yaml_config))
optimizers = Dict({
    "g": g_optimizer,
    "d": d_optimizer
})

def infer(generator, x, noise_dim, device):
    noise = torch.randn(x.shape[0], noise_dim).to(device)
    return generator(x, noise)


def save_images(dec, loader, i, n_infer=5):
    for sample in loader:
        x = sample["metos"].to(device)
        y = sample["real_imgs"].to(device)
        y_mean = y.mean(0)
        print("Loss of the mean image : ", ((y_mean - y) ** 2).mean(), flush=True)
        print("Standard deviation of the mean image : ", y_mean.std(), flush=True)

        save_image(y, f"{args.output_dir}/original_imgs_{i}.png")
        save_image(y_mean, f"{args.output_dir}/mean_imgs_{i}.png")
        y_hats = []
        for j in range(n_infer):
            y_hats.append(infer(dec, x, model_args["noise_dim"], device))

        y_hats = torch.cat(y_hats, axis=0)
        print("Standard deviation of the first image : ", y_hats[0].std(), flush=True)
        save_image(y_hats, f"{args.output_dir}/predicted_imgs_{i}-{j}.png")


for i in range(train_args["n_epochs"]):
    print(f"Epoch {i+1}")
    log_this_epoch = False
    if i % train_args["log_every"] == 0:
        log_this_epoch = True
    models, avg_loss, optimizers = train.train(models, train_loader, optimizers, nn.MSELoss(), device, train_args, model_args, i, log_this_epoch)
    if i % train_args["save_every_epochs"] == 0:
        save_images(dec, train_loader, i, train_args["n_infer"])

    if i % train_args["log_every"] == 0:
        print(f"Discriminator loss : {avg_loss.d} - Generator loss : {avg_loss.g} - Matching loss: {avg_loss.matching}", flush=True)

torch.save(dec.state_dict(), f"{args.output_dir}/dec_test.pth")


# go through loader and look at y_hat
# save images of the y_hats
# do they look like clouds?????


# normalize the metos

# think through other architectures? how to make it a gan?

# launch job on full images (supervised)

