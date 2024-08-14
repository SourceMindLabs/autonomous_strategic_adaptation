# agents/agent.py

import numpy as np
import torch
import torch.nn.functional as F
from .actor import AdvancedActor
from .critic import AdvancedCritic
from ..utils.momentum import HeavyBallMomentum

class OptimizedAdvancedAgent:
    def __init__(self, state_dim, action_dim, hidden_dims=[400, 300], actor_lr=3e-4, critic_lr=3e-4,
                 gamma=0.99, tau=0.005, alpha=0.2, momentum_beta=0.9):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = AdvancedActor(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic = AdvancedCritic(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_actor = AdvancedActor(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_critic = AdvancedCritic(state_dim, action_dim, hidden_dims).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_momentum = HeavyBallMomentum(beta=momentum_beta)
        self.critic_momentum = HeavyBallMomentum(beta=momentum_beta)

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        # Pre-allocate tensors for efficiency
        self.state_tensor = torch.FloatTensor(1, state_dim).to(self.device)
        self.action_tensor = torch.FloatTensor(1, action_dim).to(self.device)
        self.reward_tensor = torch.FloatTensor(1, 1).to(self.device)
        self.done_tensor = torch.FloatTensor(1, 1).to(self.device)

    @torch.no_grad()
    def select_action(self, state, evaluate=False):
        self.state_tensor[0] = torch.from_numpy(state)
        action, _, _ = self.actor.sample(self.state_tensor)
        return action.squeeze(0).cpu().numpy()

    def update(self, state, action, reward, next_state, done):
        # Prepare input tensors
        self.state_tensor[0] = torch.from_numpy(state)
        self.action_tensor[0] = torch.from_numpy(action)
        self.reward_tensor[0, 0] = reward
        self.state_tensor[0] = torch.from_numpy(next_state)
        self.done_tensor[0, 0] = float(done)

        with torch.no_grad():
            next_action, next_log_pi, _ = self.actor.sample(self.state_tensor)
            target_q1, target_q2 = self.target_critic(self.state_tensor, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_pi
            target_q = self.reward_tensor + (1 - self.done_tensor) * self.gamma * target_q

        current_q1, current_q2 = self.critic(self.state_tensor, self.action_tensor)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_momentum.apply(self.critic)
        self.critic_optimizer.step()

        new_action, log_pi, _ = self.actor.sample(self.state_tensor)
        q1, q2 = self.critic(self.state_tensor, new_action)
        q = torch.min(q1, q2)
        actor_loss = (self.alpha * log_pi - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_momentum.apply(self.actor)
        self.actor_optimizer.step()

        self.soft_update(self.target_actor, self.actor)
        self.soft_update(self.target_critic, self.critic)

        return actor_loss.item(), critic_loss.item()

    def soft_update(self, target, source):
        with torch.no_grad():
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def save(self, filename):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())