import numpy as np
import torch
from pathlib import Path
from torchvision import transforms

def get_stats(data_dir):
    return np.load(Path(data_dir, "metos_stats.npy"))

def get_transforms_metos_regressor(data_args):
    dataset_stats = get_stats(data_args["path"])
    replace_nans_transform = ReplaceNans()
    replace_nans_transform.set_stats(dataset_stats)
    dataset_transforms = [replace_nans_transform]
    if data_args["with_stats"]:
        dataset_transforms.append(Standardize(dataset_stats))

    feature_extractor_input_size = 224
    feature_extractor_transforms = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(
                    (feature_extractor_input_size, feature_extractor_input_size)
                ),
                transforms.ToTensor(),
                GrayscaleToRGB(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
            ]
        )

    return transforms.Compose(dataset_transforms), feature_extractor_transforms

def get_transforms(data_args):
    dataset_stats = get_stats(data_args["path"])
    replace_nans_transform = ReplaceNans()
    replace_nans_transform.set_stats(dataset_stats)
    dataset_transforms = [replace_nans_transform]
    if data_args["with_stats"]:
        dataset_transforms.append(Standardize(dataset_stats))
    img_transforms = [transforms.ToPILImage()]

    if data_args["image_size"] != 128:
        img_transforms.append(transforms.Resize(data_args["image_size"]))

    img_transforms.append(transforms.ToTensor())
    if data_args["normalize"]:
        img_transforms.append(transforms.Normalize((0.5,), (0.5,)))

    return transforms.Compose(dataset_transforms), transforms.Compose(img_transforms)

class ReplaceNans:

    def __init__(self, nan_value="Mean"):
        self.nan_value = nan_value

    def set_stats(self, stats):
        self.means, self.stds = stats[:,0], stats[:,1]

    def __call__(self, sample):
        for c in range(sample["metos"].shape[0]):
            if self.nan_value == "Mean":
                sample["metos"][c][torch.isnan(sample["metos"][c])] = self.means[c]

        return sample

class Standardize:

    def __init__(self, stats):
        self.means, self.stds = stats[:,0], stats[:,1]

    def __call__(self, sample):
        device = sample["metos"].device
        dtype = sample["metos"].dtype
        means = torch.tensor(self.means, device=device, dtype=dtype)
        stds = torch.tensor(self.stds, device=device, dtype=dtype)
        sample["metos"] = (sample["metos"] - means) / stds

        return sample

class GrayscaleToRGB(object):
    
    def __init__(self):
        pass
    
    def __call__(self, sample):
        return torch.cat([sample] * 3, dim=0)