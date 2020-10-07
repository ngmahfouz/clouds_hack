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
import seaborn as sns
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from preprocessing import ReplaceNans, get_transforms_metos_regressor, get_transforms
from optim import get_optimizers
from utils import noise_sample
import os
from PIL import Image
from matplotlib import cm

os.environ["WANDB_MODE"] = "dryrun"

parser = argparse.ArgumentParser()

parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="Where to save files",
    default="./fft_mmr_best_resnet101_ft",
)
parser.add_argument(
    "-c",
    "--config_file",
    type=str,
    help="YAML configuration file",
    default="default_training_config.yaml",
)
parser.add_argument(
    "-t",
    "--train",
    type=str,
    help="Train the regressor or just load the previous one",
    default="False",
)
parser.add_argument(
    "-g",
    "--generator_path",
    type=str,
    help="Path to the generator to evaluate",
    default="ig_state_80.pt",
)
parser.add_argument(
    "-n",
    "--number_targets",
    type=int,
    help="Number of values to regress : 1 for mean, 8 for all metos",
    default=8,
)
parser.add_argument(
    "-v",
    "--best_k_valid",
    type=int,
    help="Do model selection. 0 to disable, k > 0 to output the k best models on validation set",
    default=0,
)
parser.add_argument(
    "-m",
    "--models_directory",
    type=str,
    help="Path to the directory of the model. It should contain a 'training_config.yaml' configuration file and 'state_*.pt' files for model checkpoints to evaluate",
    default=".",
)
parser.add_argument(
    "-i",
    "--metos_indices",
    type=str,
    help="Indices of metos to considerate for validation separated by 'a'. Leave empty to use all metos. E.g: '0a1' will validate using first and second metos",
    default="",
)
parser.add_argument(
    "-s",
    "--summary_metric",
    type=str,
    help="'mean', 'max', 'min' or 'median'. Summary statistic to use on the set of metos correlation to rank the models during validation",
    default="mean",
)

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# opts = Dict({"lr": 1e-4, "n_epochs": 10001, "save_every" : 500, "noise_dim" : 2, "disc_step" : 1})
fft_output_dir = "fft"

if args.best_k_valid == 0:
    config_file_path = args.config_file
else:
    config_file_path = f"{args.models_directory}/training_config.yaml"

print(f"Loading configuration from {config_file_path}")

with open(config_file_path) as f:
    yaml_config = yaml.load(f, Loader=yaml.FullLoader)
    data_args = yaml_config["data"]
    model_args = yaml_config["model"]
    test_args = yaml_config["test"]
    val_args = yaml_config["val"]
    train_args = yaml_config["train"]

if train_args["use_wandb"]:
    import wandb

    wandb.init(project="clouds_hack_mr", dir=args.output_dir)


TRAIN_FILENAME_PREFIX = "train_"

data_args["load_limit"] = -1
val_args["set_size"] = 10000
test_args["set_size"] = 10000
train_args["batch_size"] = 112
train_args["feature_extractor_model"] = "resnet101"
train_args["use_pretrained_ft"] = True
train_args["hidden_features"] = 32
train_args["freeze_ft"] = False
train_args["freeze_first_n"] = -1
train_args["num_metos"] = int(args.number_targets)
dataset_transforms, img_transforms = get_transforms_metos_regressor(data_args)
_, img_gan_transforms = get_transforms(data_args)
clouds = data.LowClouds(
    data_args["path"],
    data_args["load_limit"],
    transform=dataset_transforms,
    device=None,
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
    train_clouds, batch_size=train_args["batch_size"], pin_memory=True
)
val_loader = DataLoader(
    val_clouds, batch_size=train_args["batch_size"], pin_memory=True
)
test_loader = DataLoader(
    test_clouds, batch_size=train_args["batch_size"], pin_memory=True
)

log_csv_file = pd.DataFrame(
    columns=["Epoch", "Discriminator_loss", "Generator_loss", "Matching_loss"]
)

val_args["infer_every"] = 10
model = MetosRegressor(train_args, device).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))

if train_args["use_wandb"]:
    wandb.config.update(data_args)
    wandb.config.update(model_args)
    wandb.config.update(train_args)
    wandb.config.update(val_args, allow_val_change=True)

previous_best_val_loss = None
if args.train == "True":
    print("====== STARTING TRAINING =======")
    for i in range(train_args["n_epochs"]):
        log_this_epoch = False
        current_epoch_loss = 0
        for idx, sample in enumerate(train_loader):
            model.train()
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

            current_epoch_loss += loss.item()
            # wandb.log({"step" : i*len(train_loader) + idx, "metos_stats/min" : x.min()})
            # wandb.log({"step" : i*len(train_loader) + idx, "metos_stats/max" : x.max()})

        current_epoch_loss /= len(train_loader)
        print(f"Epoch {i+1} loss : {current_epoch_loss}", flush=True)
        wandb.log({"epoch": i, "loss/train": current_epoch_loss})

        if i % val_args["infer_every"] == 0:
            model.eval()
            current_epoch_loss = 0
            optimizer.zero_grad()
            print("====== VALIDATION INFERENCE =======")

            with torch.no_grad():

                for idx, sample in enumerate(val_loader):
                    y = sample["metos"].to(device)
                    x = sample["real_imgs"].to(device)

                    # If the metos regressor is uni-output, the target will be the mean of the 8 metos rather than all of them
                    if train_args["num_metos"] == 1:
                        y = y.mean(dim=1, keepdims=True)

                    y_hat = model(x)
                    loss = criterion(y_hat, y)

                    current_epoch_loss += loss.item()
                    torch.cuda.empty_cache()

            current_epoch_loss /= len(val_loader)
            if previous_best_val_loss is None:
                previous_best_val_loss = current_epoch_loss
            print(f"Validation loss at epoch {i+1} : {current_epoch_loss}", flush=True)
            wandb.log({"epoch": i, "loss/valid": current_epoch_loss})
            wandb.log({"epoch": i, "loss/valid_zero": (y ** 2).mean()})
            wandb.log(
                {"epoch": i, "loss/valid_mean": ((y - y.mean(dim=0)) ** 2).mean()}
            )
            torch.save(model.state_dict(), f"{args.output_dir}/state_{i}.pt")
            torch.save(model.state_dict(), f"{args.output_dir}/state_latest.pt")
            wandb.save(f"{args.output_dir}/state_{i}.pt")

            if current_epoch_loss <= previous_best_val_loss:
                print(
                    f"New best Validation loss : {current_epoch_loss} (previous was {previous_best_val_loss})"
                )
                torch.save(model.state_dict(), f"{args.output_dir}/state_best.pt")
                wandb.save(f"{args.output_dir}/state_best.pt")
                previous_best_val_loss = current_epoch_loss


else:
    model.eval()
    model.load_state_dict(torch.load(f"{args.output_dir}/state_best.pt"))


def evaluate(
    generator_weights_path=args.generator_path, exp_output_dir=args.output_dir,
    compute_fft=True, loader=test_loader
):

    from model import DCGANGenerator

    os.makedirs(exp_output_dir, exist_ok=True)
    os.makedirs(f"{exp_output_dir}/{fft_output_dir}", exist_ok=True)

    if model_args["infogan"]:
        model_args["noise_dim"] = (
            model_args["num_dis_c"] * model_args["dis_c_dim"]
            + model_args["num_con_c"]
            + model_args["num_z"]
        )
    if model_args["concat_noise_metos"]:
        model_args["noise_dim"] += 8  # Add the 8 metos
    if model_args["film_layers"] == "":
        layers_to_film = []
    else:
        layers_to_film = [int(item) for item in model_args["film_layers"].split("a")]
    if model_args["generator"] == "dcgan":
        gan_generator = DCGANGenerator(
            data_args["image_size"],
            layers_to_film,
            model_args["noise_dim"],
            model_args["ngf"],
            model_args["nc"],
        ).to(device)
    # gan_generator = DCGANGenerator(64, layers_to_film=[0,1]).to(device)
    gan_generator.load_state_dict(torch.load(generator_weights_path)["generator"])
    metos_regressor = model
    results = {
        "batch_correlations": [],
        "all_metos": [],
        "all_generated_imgs_metos": [],
        "all_real_imgs_predicted_metos": [],
    }

    def compute_correlation(x, y):
        # x,y have shape (batch_size,num_metos)
        correlations = []
        corrs = []
        pvalues = []
        for metos_index in range(x.shape[1]):
            df = pd.DataFrame({"x": x[:, metos_index], "y": y[:, metos_index]})
            correlation_results = spearmanr(df)
            correlations.append(correlation_results)
            corrs.append(correlation_results.correlation)
            pvalues.append(correlation_results.pvalue)

        correlation_results_df = pd.DataFrame(
            {"Correlation": corrs, "P-value": pvalues}
        )
        return correlations, correlation_results_df

    plt.rcParams.update({"font.size": 5})
    metos_regressor.eval()
    for idx, sample in enumerate(loader):
        metos = sample["metos"].to(device)
        real_imgs = sample["real_imgs"].to(device)
        if model_args["infogan"]:
            gen_noise, _ = noise_sample(
                model_args["num_dis_c"],
                model_args["dis_c_dim"],
                model_args["num_con_c"],
                model_args["num_z"],
                metos.shape[0],
                device,
            )
            gen_noise = gen_noise.squeeze()
        else:
            gen_noise = torch.randn((metos.shape[0], model_args["noise_dim"])).to(
                device
            )
        if model_args["concat_noise_metos"]:
            gen_noise = torch.cat([gen_noise, metos], dim=1)
        with torch.no_grad():
            generated_images = gan_generator(metos, gen_noise)
            transformed_generated_images = []
            for generated_image in generated_images:
                transformed_generated_images.append(
                    img_transforms(generated_image.cpu())
                )
            transformed_generated_images = torch.stack(transformed_generated_images).to(
                device
            )
            generated_imgs_metos = (
                metos_regressor(transformed_generated_images).cpu().detach().numpy()
            )
            real_imgs_predicted_metos = (
                metos_regressor(real_imgs).cpu().detach().numpy()
            )

        metos = metos.cpu().detach().numpy()

        # Uni-output regressor case : the target is the mean rather than all 8 metos
        if train_args["num_metos"] == 1:
            generated_imgs_metos = generated_imgs_metos.mean(axis=1, keepdims=True)
            real_imgs_predicted_metos = real_imgs_predicted_metos.mean(
                axis=1, keepdims=True
            )
            metos = metos.mean(axis=1, keepdims=True)

        results["all_metos"].append(metos)
        results["all_generated_imgs_metos"].append(generated_imgs_metos)
        results["all_real_imgs_predicted_metos"].append(real_imgs_predicted_metos)

        real_generated_correlation = compute_correlation(generated_imgs_metos, metos)
        predicted_generated_correlation = compute_correlation(
            generated_imgs_metos, real_imgs_predicted_metos
        )
        real_predicted_correlation = compute_correlation(
            real_imgs_predicted_metos, metos
        )
        results["batch_correlations"].append(
            {
                "real_generated_correlation": real_generated_correlation,
                "predicted_generated_correlation": predicted_generated_correlation,
                "real_predicted_correlation": real_predicted_correlation,
            }
        )

        if compute_fft:

            for i in range(min(10, generated_images.shape[0])):

                image = {
                    "real": utils.to_0_1(real_imgs[i]).squeeze(0).cpu().detach().numpy(),
                    "fake": utils.to_0_1(generated_images[i])
                    .squeeze(0)
                    .cpu()
                    .detach()
                    .numpy(),
                }

                real_img = Image.fromarray(image["real"][0])
                real_img = real_img.resize(
                    (image["fake"].shape[0], image["fake"].shape[0]), Image.ANTIALIAS
                )
                image["real"] = np.array(real_img)
                fft_f = utils.fft(image["fake"])
                fft_r = utils.fft(image["real"])

                plt.subplot(2, 4, 1)
                plt.imshow(image["real"], cmap="gray")
                plt.title("real image")

                plt.subplot(2, 4, 2)

                plt.imshow(fft_r, cmap="gray")
                plt.title("DFT of real image")

                plt.subplot(2, 4, 3)
                sns.distplot(fft_r.flatten())
                plt.title("Histogram of the DFT of the real image")

                ax = plt.subplot(2, 4, 4)
                sns.distplot(fft_r.flatten(), ax=ax)
                sns.distplot(fft_f.flatten(), ax=ax)
                plt.title("Histograms of the DFTs of the real vs generated images")

                plt.subplot(2, 4, 5)

                plt.imshow(image["fake"], cmap="gray")
                plt.title("generated image")

                plt.subplot(2, 4, 6)
                plt.imshow(fft_f, cmap="gray")
                plt.title("DFT of generated image")

                plt.subplot(2, 4, 7)
                sns.distplot(fft_f.flatten())
                plt.title("Histogram of the DFT of the generated image")

                fft_d = np.linalg.norm(fft_f - fft_r) / fft_f.size()
                print("distance between the two DFTs = ", fft_d)
                plt.savefig(f"{exp_output_dir}/{fft_output_dir}/fft_{idx}_{i}.png")
                plt.close()

    all_metos = np.concatenate(results["all_metos"])
    all_generated_imgs_metos = np.concatenate(results["all_generated_imgs_metos"])
    all_real_imgs_predicted_metos = np.concatenate(
        results["all_real_imgs_predicted_metos"]
    )

    for metos_index in range(all_metos.shape[1]):

        big_max = max(
            np.abs(all_metos).max(),
            np.abs(all_generated_imgs_metos).max(),
            np.abs(all_real_imgs_predicted_metos).max(),
        )
        diag = np.linspace(-big_max, big_max, 100)
        std = 0.2
        alpha = 0.1

        plt.xlim(-big_max, big_max)
        plt.ylim(-big_max, big_max)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.scatter(
            all_metos[:, metos_index],
            all_generated_imgs_metos[:, metos_index],
            c="blue",
        )
        plt.ylabel(f"Real meto {metos_index}")
        plt.xlabel(f"Generated imgs meto {metos_index} (predicted)")
        eps = diag * std
        plt.plot(diag, diag)
        plt.fill_between(diag, diag - eps, diag + eps, alpha=alpha)
        plt.savefig(f"{exp_output_dir}/generated_real_{metos_index}.png")

        plt.xlim(-big_max, big_max)
        plt.ylim(-big_max, big_max)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.scatter(
            all_real_imgs_predicted_metos[:, metos_index],
            all_generated_imgs_metos[:, metos_index],
            c="blue",
        )
        plt.ylabel(f"Real imgs meto {metos_index} (predicted)")
        plt.xlabel(f"Generated imgs meto {metos_index} (predicted)")
        eps = diag * std
        plt.plot(diag, diag)
        plt.fill_between(diag, diag - eps, diag + eps, alpha=alpha)
        plt.savefig(f"{exp_output_dir}/generated_predicted_{metos_index}.png")

        plt.xlim(-big_max, big_max)
        plt.ylim(-big_max, big_max)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.scatter(
            all_metos[:, metos_index],
            all_real_imgs_predicted_metos[:, metos_index],
            c="blue",
        )
        plt.ylabel(f"Real imgs meto {metos_index} (predicted)")
        plt.xlabel(f"Real meto {metos_index}")
        eps = diag * std
        plt.plot(diag, diag)
        plt.fill_between(diag, diag - eps, diag + eps, alpha=alpha)
        plt.savefig(f"{exp_output_dir}/real_predicted_{metos_index}.png")

    results["real_generated_correlation"], corr_df = compute_correlation(
        all_generated_imgs_metos, all_metos
    )
    corr_df.to_csv(f"{exp_output_dir}/real_generated_correlation.csv")
    results["predicted_generated_correlation"], corr_df = compute_correlation(
        all_generated_imgs_metos, all_real_imgs_predicted_metos
    )
    corr_df.to_csv(f"{exp_output_dir}/predicted_generated_correlation.csv")
    results["real_predicted_correlation"], corr_df = compute_correlation(
        all_metos, all_real_imgs_predicted_metos
    )
    corr_df.to_csv(f"{exp_output_dir}/real_predicted_correlation.csv")

    torch.save(results, f"{exp_output_dir}/results.pth")

    return results

if args.best_k_valid == 0:
    evaluate()

else:
    import glob
    correlations_summary = []
    model_names = []
    all_results = []
    checkpoints_to_evaluate = glob.glob(f"{args.models_directory}/state_[0-3]*.pt") # Returns the list of all "*.pt" files in "args.models_directory"
    num_models = len(checkpoints_to_evaluate)
    for idx, checkpoint_to_evaluate in enumerate(checkpoints_to_evaluate):
        gan_model_path = checkpoint_to_evaluate
        print(f"{idx+1} / {num_models} : {gan_model_path}", flush=True)
        model_name = gan_model_path.split("/")[-1][:-3] # -1 means the last part after the directory separator "/" i.e. the file name and "-3" to remove the file extension
        model_names.append(model_name)
        results = evaluate(gan_model_path, f"{args.output_dir}/{model_name}", compute_fft=False, loader=val_loader)
        if args.metos_indices == "":
            metos_indices = range(8)
        else:
            metos_indices = [int(item) for item in args.metos_indices.split("a")]

        all_results.append(results)
        corrs = [results["real_generated_correlation"][meto_index].correlation for meto_index in metos_indices]
        if args.summary_metric == "mean":
            corrs_summary = np.mean(corrs)
        elif args.summary_metric == "median":
            corrs_summary = np.median(corrs)
        elif args.summary_metric == "min":
            corrs_summary = np.min(corrs)
        elif args.summary_metric == "max":
            corrs_summary = np.max(corrs)

        correlations_summary.append(corrs_summary)

    best_models_indices = np.flip(np.argsort(correlations_summary))

    for j in range(min(args.best_k_valid, len(best_models_indices))):
        i = best_models_indices[j]
        print(f"Number {j+1} : {model_names[i]} - {args.summary_metric} : {correlations_summary[i]}", flush=True)
        print("Detailed results:", flush=True)
        print(all_results[i]["real_generated_correlation"],flush=True)
        print("Performance on test set :", flush=True)
        test_results = evaluate(checkpoints_to_evaluate[i], f"{args.output_dir}/top_{j+1}_test_{model_names[i]}", compute_fft=False, loader=test_loader)
        print(test_results["real_generated_correlation"], flush=True)
        

        
            
