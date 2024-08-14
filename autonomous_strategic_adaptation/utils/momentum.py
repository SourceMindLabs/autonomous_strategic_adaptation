# utils/momentum.py

import torch
import numpy as np

class HeavyBallMomentum:
    def __init__(self, beta=0.9, dampening=0):
        self.beta = beta
        self.dampening = dampening
        self.velocity = {}
    
    def apply(self, model):
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            
            if name not in self.velocity:
                self.velocity[name] = torch.zeros_like(param.data)
            
            d_p = param.grad.data
            
            if self.dampening != 0:
                self.velocity[name] = self.beta * self.velocity[name] + (1 - self.dampening) * d_p
            else:
                self.velocity[name] = self.beta * self.velocity[name] + d_p
            
            param.data.add_(-self.velocity[name])

class NesterovMomentum:
    def __init__(self, beta=0.9):
        self.beta = beta
        self.velocity = {}
    
    def apply(self, model):
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            
            if name not in self.velocity:
                self.velocity[name] = torch.zeros_like(param.data)
            
            d_p = param.grad.data
            self.velocity[name] = self.beta * self.velocity[name] + d_p
            
            param.data.add_(-self.velocity[name])
            param.data.add_(self.beta * self.velocity[name])

class AdamMomentum:
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0
    
    def apply(self, model, learning_rate):
        self.t += 1
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            
            if name not in self.m:
                self.m[name] = torch.zeros_like(param.data)
                self.v[name] = torch.zeros_like(param.data)
            
            grad = param.grad.data
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * grad.pow(2)
            
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)
            
            param.data.addcdiv_(-learning_rate, m_hat, v_hat.sqrt() + self.epsilon)

class RMSpropMomentum:
    def __init__(self, alpha=0.99, epsilon=1e-8):
        self.alpha = alpha
        self.epsilon = epsilon
        self.square_avg = {}
    
    def apply(self, model, learning_rate):
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            
            if name not in self.square_avg:
                self.square_avg[name] = torch.zeros_like(param.data)
            
            grad = param.grad.data
            self.square_avg[name] = self.alpha * self.square_avg[name] + (1 - self.alpha) * grad.pow(2)
            avg = self.square_avg[name].sqrt().add_(self.epsilon)
            param.data.addcdiv_(-learning_rate, grad, avg)