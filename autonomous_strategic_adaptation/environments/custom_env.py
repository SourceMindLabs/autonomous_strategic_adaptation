# environments/custom_env.py

import gym
import numpy as np
from gym import spaces

class AdvancedEnvironment(gym.Env):
    def __init__(self):
        super(AdvancedEnvironment, self).__init__()
        
        # Define action and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        
        # Environment parameters
        self.max_steps = 1000
        self.current_step = 0
        self.state = None
        
        # Dynamic difficulty adjustment
        self.difficulty = 1.0
        self.success_threshold = 0.7
        self.failure_threshold = 0.3
        
    def reset(self):
        self.current_step = 0
        self.state = np.random.uniform(low=-0.1, high=0.1, size=(10,))
        return self.state
    
    def step(self, action):
        self.current_step += 1
        
        # Apply action and calculate next state
        self.state += action * 0.1 * self.difficulty
        self.state = np.clip(self.state, -1, 1)
        
        # Calculate reward
        distance = np.linalg.norm(self.state)
        reward = -distance * self.difficulty
        
        # Check if episode is done
        done = self.current_step >= self.max_steps or distance < 0.1
        
        # Dynamic difficulty adjustment
        if done:
            if distance < 0.1:
                self.adjust_difficulty(success=True)
            else:
                self.adjust_difficulty(success=False)
        
        info = {"difficulty": self.difficulty, "distance": distance}
        
        return self.state, reward, done, info
    
    def adjust_difficulty(self, success):
        if success and self.difficulty < 2.0:
            self.difficulty *= 1.1
        elif not success and self.difficulty > 0.5:
            self.difficulty *= 0.9
        self.difficulty = np.clip(self.difficulty, 0.5, 2.0)