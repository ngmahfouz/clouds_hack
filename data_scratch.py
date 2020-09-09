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
from torchvision import models, transforms
import utils
from torch.autograd import Variable
from torchvision.utils import save_image
from res_discriminator import MultiDiscriminator, Discriminator
import argparse
import yaml
import pandas as pd
from preprocessing import ReplaceNans, get_transforms
from optim import get_optimizers
import os

os.environ['WANDB_MODE'] = 'dryrun'

parser = argparse.ArgumentParser()

parser.add_argument("-o", "--output_dir", type=str, help="Where to save files", default=".")
parser.add_argument("-c", "--config_file", type=str, help="YAML configuration file", default="default_training_config.yaml")

args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)

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

if train_args["use_wandb"]:
    import wandb
    wandb.init(project="clouds_hack", dir=args.output_dir)

if model_args["infogan"]:
    model_args["noise_dim"] = model_args["num_dis_c"] * model_args["dis_c_dim"] + model_args["num_con_c"] + model_args["num_z"]

if model_args["concat_noise_metos"]:
    model_args["noise_dim"]+= 8 # Add the 8 metos

TRAIN_FILENAME_PREFIX = "train_"

dataset_transforms, img_transforms = get_transforms(data_args)
clouds = data.LowClouds(data_args["path"], data_args["load_limit"], transform=dataset_transforms, device=device, img_transforms=img_transforms)
nb_images = len(clouds)
train_clouds, val_clouds = torch.utils.data.random_split(clouds, [nb_images - val_args["set_size"], val_args["set_size"]])
train_loader = DataLoader(train_clouds, batch_size=train_args["batch_size"], pin_memory=False)
val_loader = DataLoader(val_clouds, batch_size=train_args["batch_size"])

if model_args["film_layers"] == "":
    layers_to_film = []
else:
    layers_to_film = [int(item) for item in model_args["film_layers"].split('a')]

if model_args["generator"] == "dcgan":
    dec = model.DCGANGenerator(data_args["image_size"], layers_to_film, model_args["noise_dim"], model_args["ngf"], model_args["nc"])
else:
    dec = model.Deconvolver(8, 128, n_blocks=model_args["n_blocks"], depth_increase_factor=model_args["depth_increase_factor"], noise_dim=model_args["noise_dim"])
dec = dec.to(device)
log_csv_file = pd.DataFrame(columns=["Epoch", "Discriminator_loss", "Generator_loss", "Matching_loss"])

if train_args["feature_extractor_loss"]:
    feature_extractor, feature_extractor_input_size, _ = utils.initialize_model(train_args["feature_extractor_model"])
    feature_extractor = feature_extractor.to(device)
    feature_extractor_transforms = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((feature_extractor_input_size, feature_extractor_input_size)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    feature_extraction = {"extractor" : feature_extractor, "transformations": feature_extractor_transforms}

else:
    feature_extraction = None

if model_args["discriminator"] == "dcgan":
    discrete_latent_dim = model_args["num_dis_c"] * model_args["dis_c_dim"]
    continuous_latent_dim = model_args["num_con_c"]
    disc = model.DCGANDiscriminator(data_args["image_size"], model_args["ndf"], model_args["nc"], model_args["spectral_norm"], discrete_latent_dim, continuous_latent_dim, model_args["predict_metos"])
else:
    disc = MultiDiscriminator(in_channels=1, device=device)

disc = disc.to(device)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
models = Dict({"g": dec, "d": disc})
models.g.apply(weights_init)
#models.d.apply(weights_init) d has it own initialization method because of spectral norm
g_optimizer, d_optimizer = get_optimizers(models["g"], models["d"], Dict(yaml_config))
optimizers = Dict({
    "g": g_optimizer,
    "d": d_optimizer
})

def save(generator, discriminator, optimizers, output_dir, step=0):
    state = {
        "step": step,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "d_optimizer": optimizers.d.state_dict(),
        "g_optimizer": optimizers.g.state_dict(),
    }
    torch.save(state, f"{output_dir}/state_{step}.pt")
    torch.save(state, f"{output_dir}/state_latest.pt")
    if train_args["use_wandb"]:
        wandb.save(f"{output_dir}/state_{step}.pt")

def infer(generator, x, noise_dim, device):
    noise = torch.randn(x.shape[0], noise_dim).to(device)
    return generator(x, noise)


def infer_and_save(dec, loader, i, n_infer=5, filename_prefix="val_"):
    for sample in loader:
        x = sample["metos"].to(device)
        y = sample["real_imgs"].to(device)

        #If we do inference on the training set, no need to generate (large) training batch_size images
        if filename_prefix == TRAIN_FILENAME_PREFIX:
            x = x[:val_args["set_size"]]
            y = y[:val_args["set_size"]]

        y_mean = y.mean(0)
        print("L1 Loss of the mean image : ", ((y_mean - y).abs()).mean(), flush=True)
        print("Standard deviation of the mean image : ", y_mean.std(), flush=True)

        if i == 0:
            save_image(y, f"{args.output_dir}/original_imgs_{i}.png")
            save_image(y_mean, f"{args.output_dir}/mean_imgs_{i}.png")
            # wandb.save(f"{args.output_dir}/original_imgs_{i}.png")
            # wandb.save(f"{args.output_dir}/mean_imgs_{i}.png")
            if train_args["use_wandb"]:
                wandb.log({
                    "epoch" : i,
                    "ground_truth" : [wandb.Image(j) for j in y],
                    "mean_image" : wandb.Image(y_mean)
                })
        y_hats = []
        for j in range(n_infer):
            y_hats.append(infer(dec, x, model_args["noise_dim"], device))

        y_hats = torch.cat(y_hats, axis=0)
        print("Standard deviation of the first image : ", y_hats[0].std(), flush=True)
        save_image(y_hats, f"{args.output_dir}/{filename_prefix}predicted_imgs_{i}-{j}.png", normalize=True)
        # wandb.save(f"{args.output_dir}/{filename_prefix}predicted_imgs_{i}-{j}.png")
        if train_args["use_wandb"]:
            wandb.log({
                "predicted" : [wandb.Image(j) for j in y_hats]
            })

        #If we do inference on the training set, no need to generate images on multiple batches
        if filename_prefix == TRAIN_FILENAME_PREFIX:
            break

if train_args["use_wandb"]:
    wandb.config.update(data_args)
    wandb.config.update(model_args)
    wandb.config.update(train_args, allow_val_change=True)
    wandb.config.update(val_args)

def log_train_stats(epoch, avg_losses):
    g_total_loss = (train_args["lambda_gan"] * avg_losses.g + train_args["lambda_L"] * avg_losses.matching + train_args["lambda_infogan"] * (avg_losses.mi_dis + avg_losses.mi_con))
    wandb.log({
        "epoch" : i,
        "g/loss/total" : g_total_loss,
        "g/loss/matching" : avg_losses.matching,
        "g/loss/disc" : avg_losses.g,
        "g/loss/mi_dis" : avg_losses.mi_dis,
        "g/loss/mi_con" : avg_losses.mi_con,
        "g/loss/metos" : avg_losses.metos,
        "d/loss" : avg_losses.d
        })


print("Generator : ", models.g)
print("Discriminator : ", models.d)
print("Generator optimizer : ", optimizers.g)
print("Discriminator optimizer : ", optimizers.d)

for i in range(train_args["n_epochs"]):
    print(f"Epoch {i+1}", flush=True)
    log_this_epoch = False
    if i % train_args["log_every"] == 0:
        log_this_epoch = True
    models, avg_loss, optimizers = train.train(models, train_loader, optimizers, nn.L1Loss(), device, train_args, model_args, i, feature_extraction, log_this_epoch)
    if i % val_args["infer_every"] == 0:
        print("====== VALIDATION INFERENCE =======")
        infer_and_save(dec, val_loader, i, val_args["n_infer"])
    if i % train_args["infer_every"] == 0:
        print("====== TRAINING INFERENCE =======")
        infer_and_save(dec, train_loader, i, train_args["n_infer"], filename_prefix=TRAIN_FILENAME_PREFIX)

    if i % train_args["log_every"] == 0:
        print(f"Discriminator loss : {avg_loss.d} - Generator loss : {avg_loss.g} - Matching loss: {avg_loss.matching} - MI Discrete : {avg_loss.mi_dis} - MI Continuous : {avg_loss.mi_con} - Metos : {avg_loss.metos}", flush=True)
        print("Logging...")
        log_info = pd.DataFrame(data={"Epoch" : [i], "Discriminator_loss" : [avg_loss.d], "Generator_loss" : [avg_loss.g], "Matching_loss" : [avg_loss.matching]})
        log_csv_file = log_csv_file.append(log_info)
        log_csv_file.to_csv(f"{args.output_dir}/losses.csv")
        if train_args["use_wandb"]:
            log_train_stats(i, avg_loss)
        print("Done")

    if i % train_args["save_every"] == 0:
        print("Saving...")
        save(models.g, models.d, optimizers, args.output_dir, i)
        print("Done")

torch.save(dec.state_dict(), f"{args.output_dir}/dec_test.pth")


# go through loader and look at y_hat
# save images of the y_hats
# do they look like clouds?????


# normalize the metos

# think through other architectures? how to make it a gan?

# launch job on full images (supervised)

