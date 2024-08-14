# agents/actor.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[400, 300], activation=F.relu, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        self.activation = activation
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Shared layers
        self.shared_layers = nn.ModuleList()
        dims = [state_dim] + hidden_dims
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            self.shared_layers.append(nn.Linear(in_dim, out_dim))
            self.shared_layers.append(nn.LayerNorm(out_dim))

        # Mean and log_std layers
        self.mean_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_dims[-1], action_dim)

        # Initialize weights
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.constant_(m.bias, 0.0)

    def forward(self, state):
        x = state
        for layer in self.shared_layers:
            x = self.activation(layer(x))
        
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std

    def sample(self, state):
        mean, log_std = self(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # reparameterization trick
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        
        # Enforce action bounds
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob, torch.tanh(mean)