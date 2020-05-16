import torch
import torch.nn as nn

class FiLM(nn.Module):

    def __init__(self, output_dim, input_dim=8): #input_dim=8 because there are 8 metos
        super().__init__()
        self.common_network_output_dim = 32
        self.common_network = nn.Sequential(nn.Linear(in_features=input_dim, out_features=self.common_network_output_dim))
        self.gamma_out = nn.Linear(in_features=self.common_network_output_dim, out_features=output_dim)
        self.beta_out = nn.Linear(in_features=self.common_network_output_dim, out_features=output_dim)

    def forward(self, conditioning_infos, to_film):
        out = self.common_network(conditioning_infos)
        gamma = self.gamma_out(out).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_out(out).unsqueeze(-1).unsqueeze(-1)

        return gamma * to_film + beta
