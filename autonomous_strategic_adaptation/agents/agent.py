# agents/agent.py

import numpy as np
import torch
import torch.nn.functional as F
from .actor import AdvancedActor
from .critic import AdvancedCritic

class AdvancedAgent:
    def __init__(self, state_dim, action_dim, hidden_dims=[400, 300], actor_lr=3e-4, critic_lr=3e-4,
                 gamma=0.99, tau=0.005, alpha=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = AdvancedActor(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic = AdvancedCritic(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_actor = AdvancedActor(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_critic = AdvancedCritic(state_dim, action_dim, hidden_dims).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if evaluate:
                _, _, action = self.actor.sample(state)
            else:
                action, _, _ = self.actor.sample(state)
        return action.squeeze(0).cpu().numpy()

    def update(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        reward = torch.FloatTensor([reward]).unsqueeze(0).to(self.device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        done = torch.FloatTensor([float(done)]).unsqueeze(0).to(self.device)

        with torch.no_grad():
            next_action, next_log_pi, _ = self.actor.sample(next_state)
            target_q1, target_q2 = self.target_critic(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_pi
            target_q = reward + (1 - done) * self.gamma * target_q

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        new_action, log_pi, _ = self.actor.sample(state)
        q1, q2 = self.critic(state, new_action)
        q = torch.min(q1, q2)
        actor_loss = (self.alpha * log_pi - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.target_actor, self.actor)
        self.soft_update(self.target_critic, self.critic)

        return actor_loss.item(), critic_loss.item()

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)