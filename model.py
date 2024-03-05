#!/usr/bin/env python3
#---------------------------------------------------------------------------------------------
# Description: The AE data fusion model to integrate VF and OCT measurements
# Author     : Leo(Yan) Li
# Date       : January 22, 2024
# version    : 1.0
# License    : MIT License
# Encoder network (MLP_Encoder) --> Encoding --> Decoder network (MLP_Decoder)
#       input_dim:      Input vector dimension
#       hidden_dim:     Hidden layer dimension
#       z_dim:          Encoding space dimension
#       out_dim:        Output vector dimension (Note: input_dim == out_dim)
#---------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
from torchsummary import summary


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0)

class MLP_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim, is_norm=False):
        super(MLP_Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, z_dim),
        )
        self.is_norm = is_norm
        self.apply(init_weights)
    def forward(self, x):
        z = torch.sigmoid(self.model(x)) if self.is_norm else self.model(x)
        return z

class MLP_Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, out_dim, is_norm):
        super(MLP_Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.is_norm = is_norm
        self.apply(init_weights)
    def forward(self, z):
        out = torch.sigmoid(self.model(z)) if self.is_norm else self.model(z)
        return out


if __name__ == '__main__':
    #-------------------------------------------------
    # Test the model with mock data
    #-------------------------------------------------
    data_x = torch.rand((10, 52+256+1))
    in_dim, hid_dim, z_dim = data_x.shape[1], 200, 52
    encoder = MLP_Encoder(in_dim, hid_dim, z_dim, True)
    decoder = MLP_Decoder(z_dim, hid_dim, in_dim, True)
    print('Input:',data_x.shape)
    encoding = encoder(data_x)
    print('Encoding:', encoding.shape)
    output = decoder(encoding)
    print('Output:', output.shape)
    #-------------------------------------------------
    # Show the number of parameters in each network
    #-------------------------------------------------
    print(summary(encoder.cuda(), (1, in_dim)))
    print(summary(decoder.cuda(), (1, z_dim)))

    