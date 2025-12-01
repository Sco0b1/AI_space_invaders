import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from noisy_layers import NoisyLinear
from config import USE_DUELING, USE_NOISY_NETS, USE_DISTRIBUTIONAL, N_ATOMS, V_MIN, V_MAX


class DQN(nn.Module):
    """Basic DQN with CNN architecture"""
    
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        
        self.n_actions = n_actions
        
        #Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        #Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    
    def forward(self, x):
        """Forward pass returning Q-values"""
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class DuelingDQN(nn.Module):
    """Dueling architecture separating value and advantage streams"""
    
    def __init__(self, n_actions):
        super(DuelingDQN, self).__init__()
        
        self.n_actions = n_actions
        
        #Shared convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        #Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        #Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    
    def forward(self, x):
        """Dueling forward pass combining value and advantage"""
        #Shared features
        features = self.conv(x)
        features = features.view(features.size(0), -1)

        #Value and advantage
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        #Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


class RainbowDQN(nn.Module):
    """Rainbow DQN with dueling, distributional RL, and noisy networks"""
    
    def __init__(self, n_actions, n_atoms=N_ATOMS, v_min=V_MIN, v_max=V_MAX):
        super(RainbowDQN, self).__init__()
        
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        #Support for distributional RL
        self.register_buffer('atoms', torch.linspace(v_min, v_max, n_atoms))
        
        #Shared convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        feature_size = 7 * 7 * 64
        
        #Choose layer type based on config
        if USE_NOISY_NETS:
            from config import NOISY_STD
            #Noisy linear layers with configured sigma
            def LinearLayer(in_features, out_features):
                return NoisyLinear(in_features, out_features, sigma_init=NOISY_STD)
        else:
            LinearLayer = nn.Linear
        
        if USE_DUELING:
            #Dueling architecture with distributional output
            if USE_DISTRIBUTIONAL:
                self.value_stream = nn.Sequential(
                    LinearLayer(feature_size, 512),
                    nn.ReLU(),
                    LinearLayer(512, n_atoms)
                )
                
                #Advantage stream
                self.advantage_stream = nn.Sequential(
                    LinearLayer(feature_size, 512),
                    nn.ReLU(),
                    LinearLayer(512, n_actions * n_atoms)
                )
            else:
                #Value stream (scalar)
                self.value_stream = nn.Sequential(
                    LinearLayer(feature_size, 512),
                    nn.ReLU(),
                    LinearLayer(512, 1)
                )
                
                #Advantage stream (per action)
                self.advantage_stream = nn.Sequential(
                    LinearLayer(feature_size, 512),
                    nn.ReLU(),
                    LinearLayer(512, n_actions)
                )
        else:
            #Non-dueling architecture
            if USE_DISTRIBUTIONAL:
                self.fc = nn.Sequential(
                    LinearLayer(feature_size, 512),
                    nn.ReLU(),
                    LinearLayer(512, n_actions * n_atoms)
                )
            else:
                self.fc = nn.Sequential(
                    LinearLayer(feature_size, 512),
                    nn.ReLU(),
                    LinearLayer(512, n_actions)
                )
    
    def forward(self, x):
        """Forward pass returning distributions or Q-values"""
        batch_size = x.size(0)
        features = self.conv(x)
        features = features.view(batch_size, -1)
        
        if USE_DUELING:
            if USE_DISTRIBUTIONAL:
                #Value distribution
                value = self.value_stream(features).view(batch_size, 1, self.n_atoms)
                
                #Advantage distribution
                advantage = self.advantage_stream(features).view(batch_size, self.n_actions, self.n_atoms)
                
                #Combine
                q_atoms = value + (advantage - advantage.mean(dim=1, keepdim=True))
                
                #Apply softmax to get probabilities
                q_dist = F.softmax(q_atoms, dim=2)
                
                return q_dist
            else:
                #Non-distributional dueling
                value = self.value_stream(features)
                advantage = self.advantage_stream(features)
                q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
                return q_values
        else:
            if USE_DISTRIBUTIONAL:
                #Non-dueling distributional
                q_atoms = self.fc(features).view(batch_size, self.n_actions, self.n_atoms)
                q_dist = F.softmax(q_atoms, dim=2)
                return q_dist
            else:
                #Basic Q-values
                return self.fc(features)
    
    def reset_noise(self):
        """Reset noise for noisy layers"""
        if USE_NOISY_NETS:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()
    
    def get_q_values(self, x):
        """Get Q-values from distribution for action selection"""
        if USE_DISTRIBUTIONAL:
            dist = self.forward(x)
            q_values = (dist * self.atoms).sum(dim=2)
            return q_values
        else:
            return self.forward(x)