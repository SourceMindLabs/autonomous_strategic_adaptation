# utils/adaptive_learning.py

import numpy as np

class AdaptiveLearningRate:
    def __init__(self, init_lr, min_lr=1e-6, max_lr=1.0, adaptation_speed=0.01, patience=10, cooldown=20):
        self.lr = init_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.adaptation_speed = adaptation_speed
        self.patience = patience
        self.cooldown = cooldown
        self.best_loss = float('inf')
        self.wait = 0
        self.cooldown_counter = 0
        self.history = []

    def step(self, loss):
        self.history.append(loss)
        
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return self.lr
        
        if loss < self.best_loss:
            self.best_loss = loss
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            self.lr *= (1 - self.adaptation_speed)
            self.wait = 0
            self.cooldown_counter = self.cooldown
        else:
            self.lr *= (1 + self.adaptation_speed * 0.5)  # Slower increase
        
        self.lr = max(min(self.lr, self.max_lr), self.min_lr)
        return self.lr
    
    def reset(self):
        self.lr = self.init_lr
        self.best_loss = float('inf')
        self.wait = 0
        self.cooldown_counter = 0
    
    def get_lr(self):
        return self.lr
    
    def get_history(self):
        return self.history

class CyclicLearningRate:
    def __init__(self, base_lr, max_lr, step_size, mode='triangular'):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.cycle = 0
        self.step_in_cycle = 0
        
    def step(self):
        cycle = 1 + self.step_in_cycle // (2 * self.step_size)
        x = abs(self.step_in_cycle / self.step_size - 2 * cycle + 1)
        
        if self.mode == 'triangular':
            lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x))
        elif self.mode == 'triangular2':
            lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x)) / (2 ** (cycle - 1))
        elif self.mode == 'exp_range':
            lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x)) * (0.99999 ** self.step_in_cycle)
        
        self.step_in_cycle += 1
        if self.step_in_cycle >= 2 * self.step_size:
            self.step_in_cycle = 0
            self.cycle += 1
        
        return lr

class CosineAnnealingLR:
    def __init__(self, init_lr, min_lr, total_steps):
        self.init_lr = init_lr
        self.min_lr = min_lr
        self.total_steps = total_steps
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        cosine = np.cos(np.pi * self.current_step / self.total_steps)
        lr = self.min_lr + (self.init_lr - self.min_lr) * (1 + cosine) / 2
        return lr