# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 19:07:39 2023

@author: qq102
"""

##vision2
import torch
from torch import nn
from torch.nn import functional as F

class ContinuousResidualVAE(nn.Module):
    class ResBlock(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.fc = nn.Linear(in_dim, out_dim)
            self.bn = nn.BatchNorm1d(out_dim)
            self.dropout = nn.Dropout(0.3)
            if in_dim != out_dim:
                self.downsample = nn.Linear(in_dim, out_dim)
            else:
                self.downsample = None

        def forward(self, x):
            out = F.leaky_relu(self.bn(self.fc(x)))
            out = self.dropout(out)
            if self.downsample is not None:
                x = self.downsample(x)
            return out + x
    

    def __init__(self, input_dim, hidden_dim, z_dim,loss_type='RMSE',reduction='sum'):
        super().__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.3)
        self.resblock1 = self.ResBlock(hidden_dim, hidden_dim // 2)
        self.resblock2 = self.ResBlock(hidden_dim // 2, hidden_dim // 4)
        # Latent space
        self.fc21 = nn.Linear(hidden_dim // 4, z_dim)  # mu layer
        self.fc22 = nn.Linear(hidden_dim // 4, z_dim)  # logvariance layer
        # Decoder
        self.fc3 = nn.Linear(z_dim, hidden_dim // 4)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 4)
        self.dropout3 = nn.Dropout(0.3)
        self.resblock3 = self.ResBlock(hidden_dim // 4, hidden_dim // 2)
        self.resblock4 = self.ResBlock(hidden_dim // 2, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        
        # Add attributes for loss type and reduction type
        self.loss_type = loss_type
        self.reduction = reduction
        
        if reduction not in ['mean', 'sum']:
            raise ValueError("Invalid reduction type. Expected 'mean' or 'sum', but got %s" % reduction)


    def encode(self, x):
        h = F.leaky_relu(self.bn1(self.fc1(x)))
        h = self.dropout1(h)
        h = self.resblock1(h)
        h = self.resblock2(h)
        return self.fc21(h), self.fc22(h)  # mu, logvariance

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)##取形状与std相同，且平均值为0，标准差为1的正态分布中进行采样
        return mu + eps * std

    def decode(self, z):
        h = F.leaky_relu(self.bn3(self.fc3(z)))
        h = self.dropout3(h)
        h = self.resblock3(h)
        h = self.resblock4(h)
        return self.fc4(h) # No sigmoid here

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, x.shape[1]))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, beta=1.0):
        
        if self.loss_type == 'MSE':
            self.REC = F.mse_loss(recon_x, x.view(-1, x.shape[1]), reduction=self.reduction)
        elif self.loss_type == 'RMSE':
            self.REC = torch.sqrt(F.mse_loss(recon_x, x.view(-1, x.shape[1]), reduction=self.reduction))
        else:
            raise ValueError(f'Invalid loss type: {self.loss_type}')

        if self.reduction == 'mean':
            self.KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        else: 
            self.KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return beta * self.REC + self.KLD
    


    def print_neurons(self):
        print("Encoder neurons:")
        print(f"Input: {self.fc1.in_features}, Output: {self.fc1.out_features}")
        print(f"ResBlock1 - Input: {self.resblock1.fc.in_features}, Output: {self.resblock1.fc.out_features}")
        print(f"ResBlock2 - Input: {self.resblock2.fc.in_features}, Output: {self.resblock2.fc.out_features}")
        
        print("Latent neurons:")
        print(f"mu - Input: {self.fc21.in_features}, Output: {self.fc21.out_features}")
        print(f"logvar - Input: {self.fc22.in_features}, Output: {self.fc22.out_features}")

        print("Decoder neurons:")
        print(f"Input: {self.fc3.in_features}, Output: {self.fc3.out_features}")
        print(f"ResBlock3 - Input: {self.resblock3.fc.in_features}, Output: {self.resblock3.fc.out_features}")
        print(f"ResBlock4 - Input: {self.resblock4.fc.in_features}, Output: {self.resblock4.fc.out_features}")
        print(f"Output: {self.fc4.in_features}, Output: {self.fc4.out_features}")

    def get_model_inference_z(self, x, seed=None):
        """
        This function takes input x and returns the corresponding latent vectors z.
        If a seed is provided, it is used to make the random number generator deterministic.
        """
        self.eval()  # switch to evaluation mode
        if seed is not None:
            torch.manual_seed(seed)
        with torch.no_grad():  # disable gradient computation
            mu, logvar = self.encode(x.view(-1, x.shape[1]))
            z = self.reparameterize(mu, logvar)
        return z

