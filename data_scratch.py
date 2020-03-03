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
importlib.reload(data)
importlib.reload(model)
importlib.reload(train)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
opts = Dict({"lr": 1e-4, "n_iter": 10000})

clouds = data.LowClouds("/scratch/sankarak/data/low_clouds/", 10)
loader = DataLoader(clouds, batch_size=2)
x = next(iter(loader))["metos"]

dec = model.Deconvolver(8, 128)
dec = dec.to(device)

optimizer = torch.optim.Adam(dec.parameters(), lr=opts["lr"])
dec, avg_loss = train.train(dec, loader, optimizer, nn.MSELoss(), device)

for _ in range(opts.n_iter):
    dec, avg_loss = train.train(dec, loader, optimizer, nn.MSELoss(), device)
    print(f"loss: {avg_loss}")

# torch.save(dec.state_dict(), "dec_test.pth")


# go through loader and look at y_hat
# save images of the y_hats
# do they look like clouds?????


# normalize the metos

# think through other architectures? how to make it a gan?

# launch job on full images (supervised)

