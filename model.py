#!/usr/bin/env python
"""


2020-03-03 08:52:16
"""
import torch
from pathlib import Path
import torch.nn as nn
from film import FiLM
import math
from torchvision import models, transforms
import utils

class Deconvolver(nn.Module):
    def __init__(self, n_in, n_out, n_blocks=4, depth_increase_factor=2, noise_dim=2):
        super().__init__()
        self.n_out = n_out
        self.n_in = n_in
        self.total_n_in = n_in + noise_dim #takes into account both the metos (n_in=8) and the noise (noise_dim) by concatenating their respective vectors

        self.initial_linear = nn.Linear(self.total_n_in, self.total_n_in * n_out ** 2) # we want to generate images of size height=n_out x width=n_out
        self.metos_batch_norm = nn.BatchNorm1d(n_in)
        cblocks_list = []
        unet_input_channels = self.total_n_in
        for i in range(n_blocks):
            cblocks_list.append(UNetModule(unet_input_channels, depth_increase_factor * unet_input_channels))
            unet_input_channels*= 2
        self.conv_final = nn.Conv2d(unet_input_channels, 1, kernel_size=1)
        self.model = nn.ModuleList(cblocks_list + [self.conv_final])

    def forward(self, x, noise):
        x = self.metos_batch_norm(x)
        x = torch.cat([x, noise], axis=1)
        x_ = self.initial_linear(x)
        x_ = x_.reshape(x.shape[0], self.total_n_in, self.n_out, self.n_out)
        for layer in self.model:
            x_ = layer(x_)
        return x_


class UNetModule(nn.Module):
    """
    One of the "triple layer" blocks in https://arxiv.org/pdf/1505.04597.pdf
    """
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv1 = nn.Conv2d(n_in, n_out, 3, padding=1)
        self.conv2 = nn.Conv2d(n_out, n_out, 3, padding=1)
        self.activation = nn.ReLU()
        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, x):
        layers = nn.Sequential(
            self.conv1, self.bn, self.activation,
            self.conv2, self.bn, self.activation
        )
        return layers(x)

class DCGANDiscriminator(nn.Module):
    def __init__(self, img_size=128, ndf=64, nc=1):
        super().__init__()

        self.log_first_out_channels = int(math.log2(img_size)) - 3 # For 64=2^6 the first output is 8 = 2^3 . For 128 = 2^7 it's 16=2^4 . We remove 3 each time
        cblocks_list = []
        self.criterion = nn.BCELoss()
        
        stride = 2
        padding = 1
        prev_out_channels = nc
        
        
        for i in range(self.log_first_out_channels + 1):
            current_out_channels = ndf * 2 ** i

            cblocks_list.append(nn.Conv2d(prev_out_channels, current_out_channels, 4, stride, padding, bias=False))
            cblocks_list.append(nn.BatchNorm2d(current_out_channels))
            
            cblocks_list.append(nn.LeakyReLU(0.2, inplace=True))
            prev_out_channels = current_out_channels

        self.model = nn.ModuleList(cblocks_list + [nn.Conv2d(current_out_channels, 1, 4, 1, 0, bias=False), nn.Sigmoid()])


    def compute_loss(self, x, gt):
        """Computes the BCELoss between model output and scalar gt"""
        label = torch.full((x.shape[0],), gt, device=x.device)
        loss = self.criterion(self.forward(x), label)
        return loss
    
    def forward(self, input):
        output = input
        for layer in self.model:
            output = layer(output)

        return output.view(-1, 1).squeeze(1)


class DCGANGenerator(nn.Module):
    def __init__(self, img_size, layers_to_film=[], nz=100, ngf=64, nc=1):
        super().__init__()
        self.log_first_out_channels = int(math.log2(img_size)) - 3 # For 64=2^6 the first output is 8 = 2^3 . For 128 = 2^7 it's 16=2^4 . We remove 3 each time
        self.layers_to_film = layers_to_film
        cblocks_list = []
        
        stride = 1
        padding = 0
        prev_out_channels = nz
        
        
        for i in range(self.log_first_out_channels, -1, -1):
            current_out_channels = ngf * 2 ** i

            cblocks_list.append(nn.ConvTranspose2d(prev_out_channels, current_out_channels, 4, stride, padding, bias=False))
            cblocks_list.append(nn.BatchNorm2d(current_out_channels))
            if (self.log_first_out_channels - i) in self.layers_to_film: # Index of the layer to film. If we want to FiLM the first layer (index 0), it correspond to i = self.log_first_out_channels
                cblocks_list.append(FiLM(output_dim=current_out_channels))
            cblocks_list.append(nn.ReLU(True))
            stride = 2
            padding = 1
            prev_out_channels = current_out_channels

        self.model = nn.ModuleList(cblocks_list + [nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False), nn.Tanh()])

        

    def forward(self, metos, noise):
        x_ = noise.unsqueeze(-1).unsqueeze(-1)
        for layer in self.model:
            if isinstance(layer, FiLM):
                x_ = layer(metos, x_)
            else:
                x_ = layer(x_)
        return x_

class MetosRegressor(nn.Module):

    def __init__(self, train_args, device):
        super().__init__()
        self.device = device
        feature_extractor, feature_extractor_input_size, self.num_features = utils.initialize_model(train_args["feature_extractor_model"])
        feature_extractor = feature_extractor.to(device)
        feature_extractor_transforms = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((feature_extractor_input_size, feature_extractor_input_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])

        self.feature_extraction = {"extractor" : feature_extractor, "transformations": feature_extractor_transforms}

        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.num_features, out_features=train_args["hidden_features"]),
            nn.ReLU(),
            nn.Linear(in_features=train_args["hidden_features"], out_features=train_args["hidden_features"]),
            nn.ReLU(),
            nn.Linear(in_features=train_args["hidden_features"], out_features=train_args["num_metos"])
        )

    def forward(self, x):
        
        rgb_x = torch.cat([x] * 3, dim=1) #converts grayscale to RGB by replicating the single channel 3 times
        rgb_normalized_x = []
        for j in range(x.shape[0]):
            rgb_normalized_x.append(self.feature_extraction["transformations"](rgb_x[j].cpu()))
        rgb_x = torch.stack(rgb_normalized_x).to(self.device)
        feature_maps_x = self.feature_extraction["extractor"](rgb_x)

        return self.regression_head(feature_maps_x)