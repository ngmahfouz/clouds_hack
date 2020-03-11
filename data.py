#!/usr/bin/env python
"""


2020-03-03 08:52:44
"""
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from pathlib import Path
from itertools import compress
import re
plt.ion()


class LowClouds(Dataset):
    """
    Low Clouds / Metereology Data

    Each index corresponds to one 128 x 128 low cloud image, along with 8
    meteorological variables. A separate metadata field stores parsed
    information from the filenames.

    Example
    -------
    >>> clouds = LowClouds("/scratch/sankarak/data/low_clouds/")
    """

    def __init__(
            self,
            data_dir,
            load_limit=-1,
            val_ids=set([]),
            is_val=False,
            transform=None
    ):
        super(LowClouds).__init__()

        # validation logic
        files = np.load(Path(data_dir, "files.npy"))
        self.transform = transform
        self.ids = [Path(str(f)).name for f in files]
        self.ids = [re.search("([0-9]+.*)(?=_Block)", f).group() for f in self.ids]
        self.val_ids = val_ids
        self.is_val = is_val
        val_ids = set(str(v) for v in val_ids)

        if is_val:
            subset_ix = [s in val_ids for s in self.ids]
        else:
            subset_ix = [s not in val_ids for s in self.ids]

        # read data and subset
        self.ids = list(compress(self.ids, subset_ix))
        self.data = {
            "metos": np.load(Path(data_dir, "meto.npy")).transpose((1, 0))[subset_ix],
            "real_imgs": np.load(Path(data_dir, "train.npy")).transpose((2, 0, 1))[subset_ix],
        }

        # some metadata
        if load_limit != -1:
            self.ids = self.ids[:load_limit]
            self.data["metos"] = self.data["metos"][:load_limit]
            self.data["real_imgs"] = self.data["real_imgs"][:load_limit]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        data = {
            "metos": torch.tensor(self.data["metos"][i], dtype=torch.float),
            "real_imgs": torch.tensor(self.data["real_imgs"][i], dtype=torch.float).unsqueeze(0),
        }
        if self.transform:
            data = self.transform(data)
        return data
