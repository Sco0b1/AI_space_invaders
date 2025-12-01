import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NoisyLinear(nn.Module):
    """
    Noisy Linear Layer
    """
    
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        
        #Learnable parameters
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize learnable parameters"""
        mu_range = 1.0 / np.sqrt(self.in_features)
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        """Sample new noise"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size):
        """Factorized Gaussian noise"""
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()
    
    def forward(self, x):
        """Forward pass with noisy weights"""
        if self.training:
            weight_sigma_clamped = torch.clamp(self.weight_sigma, min=1e-3)
            bias_sigma_clamped = torch.clamp(self.bias_sigma, min=1e-3)
            
            weight = self.weight_mu + weight_sigma_clamped * self.weight_epsilon
            bias = self.bias_mu + bias_sigma_clamped * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)

    def disable_noise(self):
        """Disable noise for evaluation"""
        self.weight_epsilon.zero_()
        self.bias_epsilon.zero_()