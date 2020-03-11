import numpy as np
import torch
from pathlib import Path
from torchvision.transforms import Compose

def get_stats(data_dir):
    return np.load(Path(data_dir, "metos_stats.npy"))

def get_transforms(data_args):
    dataset_stats = get_stats(data_args["path"])
    replace_nans_transform = ReplaceNans()
    replace_nans_transform.set_stats(dataset_stats)
    dataset_transforms = [replace_nans_transform]

    return Compose(dataset_transforms)

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