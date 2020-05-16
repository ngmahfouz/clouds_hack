import numpy as np
import torch
from pathlib import Path
from torchvision import transforms

def get_stats(data_dir):
    return np.load(Path(data_dir, "metos_stats.npy"))

def get_transforms(data_args):
    dataset_stats = get_stats(data_args["path"])
    replace_nans_transform = ReplaceNans()
    replace_nans_transform.set_stats(dataset_stats)
    dataset_transforms = [replace_nans_transform]
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