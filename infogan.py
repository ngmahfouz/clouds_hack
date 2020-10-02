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

os.environ["WANDB_MODE"] = "dryrun"

parser = argparse.ArgumentParser()

parser.add_argument(
    "-o", "--output_dir", type=str, help="Where to save files", default="."
)
parser.add_argument(
    "-c",
    "--config_file",
    type=str,
    help="YAML configuration file",
    default="default_training_config.yaml",
)

args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)

# Fix a seed
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# opts = Dict({"lr": 1e-4, "n_epochs": 10001, "save_every" : 500, "noise_dim" : 2, "disc_step" : 1})

# Load configuration file
with open(args.config_file) as f:
    yaml_config = yaml.load(f, Loader=yaml.FullLoader)
    data_args = yaml_config["data"]
    model_args = yaml_config["model"]
    test_args = yaml_config["test"]
    val_args = yaml_config["val"]
    train_args = yaml_config["train"]

if train_args["use_wandb"]:
    import wandb

    wandb.init(project="clouds_hack", dir=args.output_dir)

# Automatically compute the final dimesion of the noise vector depending on if the model is an infogan and if we concatenate noise and metos
if model_args["infogan"]:
    model_args["noise_dim"] = (
        model_args["num_dis_c"] * model_args["dis_c_dim"]
        + model_args["num_con_c"]
        + model_args["num_z"]
    )

if model_args["concat_noise_metos"]:
    model_args["noise_dim"] += 8  # Add the 8 metos

TRAIN_FILENAME_PREFIX = "train_"

# Load the dataset and create the dataloader
dataset_transforms, img_transforms = get_transforms(data_args)
clouds = data.LowClouds(
    data_args["path"],
    data_args["load_limit"],
    transform=dataset_transforms,
    device=device,
    img_transforms=img_transforms,
)
nb_images = len(clouds)
train_clouds, val_clouds, test_clouds = torch.utils.data.random_split(
    clouds,
    [
        nb_images - val_args["set_size"] - test_args["set_size"],
        val_args["set_size"],
        test_args["set_size"],
    ],
)
train_loader = DataLoader(
    train_clouds, batch_size=train_args["batch_size"], pin_memory=False
)
val_loader = DataLoader(val_clouds, batch_size=train_args["batch_size"])
test_loader = DataLoader(test_clouds, batch_size=train_args["batch_size"])

# Parse the "film_layers" param to know which layers to FiLM (the layers are separated by "a")
if model_args["film_layers"] == "":
    layers_to_film = []
else:
    layers_to_film = [int(item) for item in model_args["film_layers"].split("a")]

if model_args["generator"] == "dcgan":
    dec = model.DCGANGenerator(
        data_args["image_size"],
        layers_to_film,
        model_args["noise_dim"],
        model_args["ngf"],
        model_args["nc"],
    )
else:
    dec = model.Deconvolver(
        8,
        128,
        n_blocks=model_args["n_blocks"],
        depth_increase_factor=model_args["depth_increase_factor"],
        noise_dim=model_args["noise_dim"],
    )
dec = dec.to(device)
log_csv_file = pd.DataFrame(
    columns=["Epoch", "Discriminator_loss", "Generator_loss", "Matching_loss"]
)

# Initialize the model for feature_extractor loss (Matching loss in the feature maps space)
if train_args["feature_extractor_loss"]:
    feature_extractor, feature_extractor_input_size, _ = utils.initialize_model(
        train_args["feature_extractor_model"]
    )
    feature_extractor = feature_extractor.to(device)
    feature_extractor_transforms = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(
                (feature_extractor_input_size, feature_extractor_input_size)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    feature_extraction = {
        "extractor": feature_extractor,
        "transformations": feature_extractor_transforms,
    }

else:
    feature_extraction = None

if model_args["discriminator"] == "dcgan":
    discrete_latent_dim = model_args["num_dis_c"] * model_args["dis_c_dim"]
    continuous_latent_dim = model_args["num_con_c"]
    disc = model.DCGANDiscriminator(
        data_args["image_size"],
        model_args["ndf"],
        model_args["nc"],
        model_args["spectral_norm"],
        discrete_latent_dim,
        continuous_latent_dim,
        model_args["predict_metos"],
    )
else:
    disc = MultiDiscriminator(in_channels=1, device=device)

disc = disc.to(device)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


models = Dict({"g": dec, "d": disc})
models.g.apply(weights_init)
# models.d.apply(weights_init) d has it own initialization method because of spectral norm
g_optimizer, d_optimizer = get_optimizers(models["g"], models["d"], Dict(yaml_config))
optimizers = Dict({"g": g_optimizer, "d": d_optimizer})


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
    if model_args["infogan"]:
        noise, dis_idx = utils.noise_sample(
            model_args["num_dis_c"],
            model_args["dis_c_dim"],
            model_args["num_con_c"],
            model_args["num_z"],
            x.shape[0],
            device,
        )
        noise = noise.squeeze()
    else:
        noise = torch.randn(x.shape[0], noise_dim).to(device)
        dis_idx = [[None]]  # For non-infogan, there is no discrete latent code index
    if model_args["concat_noise_metos"]:
        noise = torch.cat([noise, x], dim=1)
    return generator(x, noise), noise, dis_idx


def infer_and_save(models, loader, i, n_infer=5, filename_prefix="val_"):
    dec = models.g
    epoch_losses = Dict(
        {"d": 0, "g": 0, "matching": 0, "mi_dis": 0, "mi_con": 0, "metos": 0}
    )
    for batch_idx, sample in enumerate(loader):
        print(f"Index {batch_idx} {filename_prefix}")
        x = sample["metos"].to(device)
        y = sample["real_imgs"].to(device)

        # If we do inference on the training set, no need to generate (large) training batch_size images
        if filename_prefix == TRAIN_FILENAME_PREFIX:
            x = x[: val_args["set_size"]]
            y = y[: val_args["set_size"]]

        y_mean = y.mean(0)
        print("L1 Loss of the mean image : ", ((y_mean - y).abs()).mean(), flush=True)
        print("Standard deviation of the mean image : ", y_mean.std(), flush=True)

        y_hats = []
        dis_indices = []
        noises = []
        for j in range(n_infer):
            y_hat, noise, dis_idx = infer(dec, x, model_args["noise_dim"], device)
            y_hats.append(y_hat)
            noises.append(noise)
            dis_indices.append(dis_idx)

        y_hats = torch.cat(y_hats, axis=0)

        dis_idx = np.concatenate(dis_indices, axis=1)
        noise = torch.cat(noises, axis=0)


        losses = Dict(
            {"d": 0, "g": 0, "matching": 0, "mi_dis": 0, "mi_con": 0, "metos": 0}
        )
        losses.d = models.d.compute_loss(y, 1) + models.d.compute_loss(
            y_hats.detach(), 0
        )
        losses.g = models.d.compute_loss(y_hats, 1)
        loss_fun = nn.L1Loss()
        losses.matching = loss_fun(y_hat, y)
        total_loss_g = (
            train_args["lambda_gan"] * losses.g
            + train_args["lambda_L"] * losses.matching
        )

        if model_args["infogan"] or model_args["predict_metos"]:

            # Loss for discrete latent code.
            criterionQ_dis = nn.CrossEntropyLoss()
            # Loss for continuous latent code.
            criterionQ_con = utils.NormalNLLLoss()
            # Loss for metos prediction
            criterion_metos = nn.MSELoss()

            models.d.discriminator_head = False
            q_logits, q_mu, q_var = models.d(y_hats)
            target = torch.LongTensor(dis_idx).to(device)
            dis_loss = torch.zeros(total_loss_g.shape).to(device)
            con_loss = torch.zeros(total_loss_g.shape).to(device)
            metos_loss = torch.zeros(total_loss_g.shape).to(device)

            for j in range(model_args["num_dis_c"]):
                left_index = j * model_args["dis_c_dim"]
                dis_loss += criterionQ_dis(
                    q_logits[:, left_index : left_index + model_args["dis_c_dim"]],
                    target[j],
                )

            if model_args["num_con_c"] != 0:
                left_index = (
                    model_args["num_z"]
                    + model_args["num_dis_c"] * model_args["dis_c_dim"]
                )
                con_loss = (
                    criterionQ_con(
                        noise[
                            :, left_index : left_index + model_args["num_con_c"]
                        ].view(-1, model_args["num_con_c"]),
                        q_mu,
                        q_var,
                    )
                    * 0.1
                )

            if model_args["predict_metos"]:
                metos_loss = criterion_metos(q_mu, torch.cat([x] * n_infer, axis=0))

            losses.mi_dis, losses.mi_con, losses.metos = (
                dis_loss,
                con_loss,
                metos_loss,
            )
            total_loss_g += train_args["lambda_infogan"] * (dis_loss + con_loss)
            total_loss_g += train_args["lambda_metos"] * metos_loss

        img_grid = []
        for j in range(
            min(val_args["max_viz_images"], x.shape[0])
        ):  # For each image. The index corresponds to one row in the grid
            img_grid.append(
                y[j].unsqueeze(0)
            )  # Append the image of index j as first element of the row. We unsqueeze to facilitate the later torch.cat
            img_j_y_hats_indices = [j + k * x.shape[0] for k in range(n_infer)]
            img_grid.append(
                y_hats[img_j_y_hats_indices]
            )  # Append n_infer images corresponding to original image of index j
            img_grid.append(
                y_mean.unsqueeze(0)
            )  # Append (at the end of the row) the mean image

        img_grid = torch.cat(img_grid, axis=0)

        print("Standard deviation of the first image : ", y_hats[0].std(), flush=True)
        save_image(
            img_grid,
            f"{args.output_dir}/{filename_prefix}predicted_imgs_{i}-{batch_idx}-{j+1}.png",
            normalize=True,
            nrow=(1 + n_infer + 1),
        )
        wandb.save(f"{args.output_dir}/{filename_prefix}predicted_imgs_{i}-{batch_idx}-{j+1}.png")

        for k in epoch_losses.keys():
            if isinstance(losses[k], (int, float)):
                loss_val = losses[k]
            else:
                loss_val = losses[k].item()
            epoch_losses[k] += loss_val / len(loader)

        # If we do inference on the training set, no need to generate images on multiple batches
        if filename_prefix == TRAIN_FILENAME_PREFIX:
            break

    log_train_stats(i, epoch_losses, prefix=f"{filename_prefix}infer_")


if train_args["use_wandb"]:
    wandb.config.update(data_args)
    wandb.config.update(model_args)
    wandb.config.update(train_args, allow_val_change=True)
    wandb.config.update(val_args)


def log_train_stats(epoch, avg_losses, prefix="training_"):
    g_total_loss = (
        train_args["lambda_gan"] * avg_losses.g
        + train_args["lambda_L"] * avg_losses.matching
        + train_args["lambda_infogan"] * (avg_losses.mi_dis + avg_losses.mi_con)
        + train_args["lambda_metos"] * avg_loss.metos
    )
    wandb.log(
        {
            "epoch": i,
            f"g/{prefix}loss/total": g_total_loss,
            f"g/{prefix}loss/matching": avg_losses.matching,
            f"g/{prefix}loss/disc": avg_losses.g,
            f"g/{prefix}loss/mi_dis": avg_losses.mi_dis,
            f"g/{prefix}loss/mi_con": avg_losses.mi_con,
            f"g/{prefix}loss/metos": avg_losses.metos,
            f"d/{prefix}loss": avg_losses.d,
        }
    )


print("Generator : ", models.g)
print("Discriminator : ", models.d)
print("Generator optimizer : ", optimizers.g)
print("Discriminator optimizer : ", optimizers.d)

for i in range(train_args["n_epochs"]):
    print(f"Epoch {i+1}", flush=True)
    log_this_epoch = False
    if i % train_args["log_every"] == 0:
        log_this_epoch = True
    models, avg_loss, optimizers = train.train(
        models,
        train_loader,
        optimizers,
        nn.L1Loss(),
        device,
        train_args,
        model_args,
        i,
        feature_extraction,
        log_this_epoch,
    )
    if i % val_args["infer_every"] == 0:
        print("====== VALIDATION INFERENCE =======")
        with torch.no_grad():
            infer_and_save(models, val_loader, i, val_args["n_infer"])
    if i % train_args["infer_every"] == 0:
        with torch.no_grad():
            print("====== TRAINING INFERENCE =======")
            infer_and_save(
                models,
                train_loader,
                i,
                train_args["n_infer"],
                filename_prefix=TRAIN_FILENAME_PREFIX,
            )

    if i % train_args["log_every"] == 0:
        print(
            f"Discriminator loss : {avg_loss.d} - Generator loss : {avg_loss.g} - Matching loss: {avg_loss.matching} - MI Discrete : {avg_loss.mi_dis} - MI Continuous : {avg_loss.mi_con} - Metos : {avg_loss.metos}",
            flush=True,
        )
        print("Logging...")
        if train_args["use_wandb"]:
            log_train_stats(i, avg_loss)
        print("Done")

    if i % train_args["save_every"] == 0:
        print("Saving...")
        save(models.g, models.d, optimizers, args.output_dir, i)
        print("Done")

torch.save(dec.state_dict(), f"{args.output_dir}/dec_test.pth")

