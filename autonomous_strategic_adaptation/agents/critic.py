# agents/critic.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdvancedCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[400, 300], activation=F.relu):
        super(AdvancedCritic, self).__init__()
        self.activation = activation
        
        # Q1 architecture
        self.q1_layers = nn.ModuleList()
        dims = [state_dim + action_dim] + hidden_dims
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            self.q1_layers.append(nn.Linear(in_dim, out_dim))
            self.q1_layers.append(nn.LayerNorm(out_dim))
        self.q1_output = nn.Linear(hidden_dims[-1], 1)

        # Q2 architecture
        self.q2_layers = nn.ModuleList()
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            self.q2_layers.append(nn.Linear(in_dim, out_dim))
            self.q2_layers.append(nn.LayerNorm(out_dim))
        self.q2_output = nn.Linear(hidden_dims[-1], 1)

        # Initialize weights
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.constant_(m.bias, 0.0)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        
        # Q1 forward pass
        q1 = x
        for layer in self.q1_layers:
            q1 = self.activation(layer(q1))
        q1 = self.q1_output(q1)

        # Q2 forward pass
        q2 = x
        for layer in self.q2_layers:
            q2 = self.activation(layer(q2))
        q2 = self.q2_output(q2)

        return q1, q2

    def Q1(self, state, action):
        x = torch.cat([state, action], dim=1)
        q1 = x
        for layer in self.q1_layers:
            q1 = self.activation(layer(q1))
        q1 = self.q1_output(q1)
        return q1