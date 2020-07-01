#!/usr/bin/env python
"""


2020-03-03 08:56:40
"""
from addict import Dict
from torch.utils.data import DataLoader
import data
import importlib
import model
import torch
import torch.nn as nn
import train
import torch.optim as optim
from torchvision import models, transforms
import utils
from torch.autograd import Variable
from torchvision.utils import save_image
from res_discriminator import MultiDiscriminator, Discriminator
from model import MetosRegressor
import argparse
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import ReplaceNans, get_transforms
from optim import get_optimizers
import os

os.environ['WANDB_MODE'] = 'dryrun'

parser = argparse.ArgumentParser()

parser.add_argument("-o", "--output_dir", type=str, help="Where to save files", default="./uni_metos_regressor")
parser.add_argument("-c", "--config_file", type=str, help="YAML configuration file", default="default_training_config.yaml")
parser.add_argument("-t", "--train", type=str, help="Train the regressor or just load the previous one", default="True")
parser.add_argument("-g", "--generator_path", type=str, help="Path to the generator to evaluate", default="state_2000.pt")
parser.add_argument("-n", "--number_targets", type=str, help="Number of values to regress : 1 for mean, 8 for all metos", default="8")

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


TRAIN_FILENAME_PREFIX = "train_"

dataset_transforms, img_transforms = get_transforms(data_args)
clouds = data.LowClouds(data_args["path"], data_args["load_limit"], transform=dataset_transforms, device=device, img_transforms=img_transforms)
nb_images = len(clouds)
train_clouds, val_clouds = torch.utils.data.random_split(clouds, [nb_images - val_args["set_size"], val_args["set_size"]])
train_loader = DataLoader(train_clouds, batch_size=train_args["batch_size"], pin_memory=False)
val_loader = DataLoader(val_clouds, batch_size=train_args["batch_size"])


log_csv_file = pd.DataFrame(columns=["Epoch", "Discriminator_loss", "Generator_loss", "Matching_loss"])

train_args["hidden_features"] = 32
train_args["num_metos"] = int(args.number_targets)
val_args["infer_every"] = 20
model = MetosRegressor(train_args, device).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))


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
    wandb.config.update(train_args)
    wandb.config.update(val_args, allow_val_change=True)

def log_train_stats(epoch, avg_losses):
    g_total_loss = (train_args["lambda_gan"] * avg_losses.g + train_args["lambda_L"] * avg_losses.matching)
    wandb.log({
        "epoch" : i,
        "g/loss/total" : g_total_loss,
        "g/loss/matching" : avg_losses.matching,
        "g/loss/disc" : avg_losses.g,
        "d/loss" : avg_losses.d
        })

previous_best_val_loss = None
if args.train == "True":
    print("====== STARTING TRAINING =======")
    for i in range(train_args["n_epochs"]):
        log_this_epoch = False
        current_epoch_loss = 0
        for idx, sample in enumerate(train_loader):
            optimizer.zero_grad()
            y = sample["metos"].to(device)
            x = sample["real_imgs"].to(device)

            # If the metos regressor is uni-output, the target will be the mean of the 8 metos rather than all of them
            if train_args["num_metos"] == 1:
                y = y.mean(dim=1, keepdims=True)

            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            current_epoch_loss+= loss.item()
            #wandb.log({"step" : i*len(train_loader) + idx, "metos_stats/min" : x.min()})
            #wandb.log({"step" : i*len(train_loader) + idx, "metos_stats/max" : x.max()})

        current_epoch_loss/= len(train_loader)
        print(f"Epoch {i+1} loss : {current_epoch_loss}", flush=True)
        wandb.log({"epoch" : i, "loss/train" : current_epoch_loss})

        if i % val_args["infer_every"] == 0:
            print("====== VALIDATION INFERENCE =======")

            for idx, sample in enumerate(val_loader):
                y = sample["metos"].to(device)
                x = sample["real_imgs"].to(device)

                # If the metos regressor is uni-output, the target will be the mean of the 8 metos rather than all of them
                if train_args["num_metos"] == 1:
                    y = y.mean(dim=1, keepdims=True)

                y_hat = model(x)
                loss = criterion(y_hat, y)

                current_epoch_loss+= loss.item()

            current_epoch_loss/= len(train_loader)
            if previous_best_val_loss is None:
                previous_best_val_loss = current_epoch_loss
            print(f"Validation loss at epoch {i+1} : {current_epoch_loss}", flush=True)
            wandb.log({"epoch" : i, "loss/valid" : current_epoch_loss})
            torch.save(model.state_dict(), f"{args.output_dir}/state_{i}.pt")
            torch.save(model.state_dict(), f"{args.output_dir}/state_latest.pt")
            wandb.save(f"{args.output_dir}/state_{i}.pt")

            if current_epoch_loss <= previous_best_val_loss:
                print(f"New best Validation loss : {current_epoch_loss} (previous was {previous_best_val_loss})")
                torch.save(model.state_dict(), f"{args.output_dir}/state_best.pt")
                wandb.save(f"{args.output_dir}/state_best.pt")
                previous_best_val_loss = current_epoch_loss
        

else:
    model.load_state_dict(torch.load(f"{args.output_dir}/state_best.pt"))

from model import DCGANGenerator
gan_generator = DCGANGenerator(64, layers_to_film=[0,1]).to(device)
gan_generator.load_state_dict(torch.load(args.generator_path)["generator"])
metos_regressor = model
results = {"batch_correlations" : [], "all_metos" : [], "all_generated_imgs_metos" : [], "all_real_imgs_predicted_metos" : []}

def compute_correlation(x, y):
    #x,y have shape (batch_size,num_metos)
    correlation = []
    for metos_index in range(x.shape[1]):
        df = pd.DataFrame({"x": x[:,metos_index], "y": y[:,metos_index]})
        correlation.append(df.corr(method="spearman").iloc[1,0])
    return correlation

for idx, sample in enumerate(val_loader):
    metos = sample["metos"]
    real_imgs = sample["real_imgs"]
    gen_noise = torch.randn((metos.shape[0], model_args["noise_dim"])).to(device)
    with torch.no_grad():
        generated_images = gan_generator(metos, gen_noise)
        generated_imgs_metos = metos_regressor(generated_images).cpu().detach().numpy()
        real_imgs_predicted_metos = metos_regressor(real_imgs).cpu().detach().numpy()

    metos = metos.cpu().detach().numpy()

    #Uni-output regressor case : the target is the mean rather than all 8 metos
    if train_args["num_metos"] == 1:
        generated_imgs_metos = generated_imgs_metos.mean(axis=1, keepdims=True)
        real_imgs_predicted_metos =real_imgs_predicted_metos.mean(axis=1, keepdims=True)
        metos = metos.mean(axis=1, keepdims=True)
    
    results["all_metos"].append(metos)
    results["all_generated_imgs_metos"].append(generated_imgs_metos)
    results["all_real_imgs_predicted_metos"].append(real_imgs_predicted_metos)

    real_generated_correlation = compute_correlation(generated_imgs_metos, metos)
    predicted_generated_correlation = compute_correlation(generated_imgs_metos, real_imgs_predicted_metos)
    real_predicted_correlation = compute_correlation(real_imgs_predicted_metos, metos)
    results["batch_correlations"].append({
        "real_generated_correlation" : real_generated_correlation,
        "predicted_generated_correlation" : predicted_generated_correlation,
        "real_predicted_correlation" : real_predicted_correlation
    })

all_metos = np.concatenate(results["all_metos"])
all_generated_imgs_metos = np.concatenate(results["all_generated_imgs_metos"])
all_real_imgs_predicted_metos = np.concatenate(results["all_real_imgs_predicted_metos"])

for metos_index in range(all_metos.shape[1]):
    
    big_max = max(np.abs(all_metos).max(), np.abs(all_generated_imgs_metos).max(), np.abs(all_real_imgs_predicted_metos).max())

    plt.xlim(-big_max, big_max)
    plt.ylim(-big_max, big_max)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(all_generated_imgs_metos[:, metos_index], all_metos[:,metos_index])
    plt.ylabel(f"Real meto {metos_index}")
    plt.xlabel(f"Generated imgs meto {metos_index} (predicted)")
    plt.savefig(f"{args.output_dir}/generated_real_{metos_index}.png")

    plt.xlim(-big_max, big_max)
    plt.ylim(-big_max, big_max)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(all_generated_imgs_metos[:, metos_index], all_real_imgs_predicted_metos[:,metos_index])
    plt.ylabel(f"Real imgs meto {metos_index} (predicted)")
    plt.xlabel(f"Generated imgs meto {metos_index} (predicted)")
    plt.savefig(f"{args.output_dir}/generated_predicted_{metos_index}.png")

    plt.xlim(-big_max, big_max)
    plt.ylim(-big_max, big_max)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(all_metos[:, metos_index], all_real_imgs_predicted_metos[:,metos_index])
    plt.ylabel(f"Real imgs meto {metos_index} (predicted)")
    plt.xlabel(f"Real meto {metos_index}")
    plt.savefig(f"{args.output_dir}/real_predicted_{metos_index}.png")

results["real_generated_correlation"] = compute_correlation(all_generated_imgs_metos, all_metos)
results["predicted_generated_correlation"] = compute_correlation(all_generated_imgs_metos, all_real_imgs_predicted_metos)
results["real_predicted_correlation"] = compute_correlation(all_metos, all_real_imgs_predicted_metos)

torch.save(results, f"{args.output_dir}/results.pth")
