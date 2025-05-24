import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm 
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()

        self.block1 = nn.Sequential(
            nn.Linear(n_observations, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.block3 = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.block4 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.block5 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.output_layer = nn.Linear(32, n_actions)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return self.output_layer(x)


class Environment:
    def __init__(self, states, action_space):
        self.states = states
        self.reset_states = states
        self.action_space = action_space

    def reset(self):
        self.states = self.reset_states
    

    def step(self, action):
        return 0
    
    def make_rl_data(self, price_series, window=10):
        X = []
        for i in range(len(price_series) - window - 1):
            state = price_series[i:i+window].values.astype(np.float32)
            next_state = price_series[i+1:i+1+window].values.astype(np.float32)
            reward = float(price_series[i+window+1] - price_series[i+window])
            done = i + window + 1 >= len(price_series) - 1
            X.append((state, 1 if reward > 0 else 0, reward, next_state, done))
        return X



batch_size = 64
γ = 0.99
ε_START = 0.9
ε_END = 0.05
ε_DECAY = 1000
τ = 0.005
η = 1e-4
num_episodes = 10

# State: [date, price] action: [buy, sell, hold] reward
π_net = DQN(2, 3).to(device)
b_net = DQN(2, 3).to(device)
b_net.load_state_dict(π_net.state_dict())

optimizer = optim.Adam(π_net.parameters(), lr=η)
memory = ReplayMemory(10000)
env = Environment(3, 3)
def optimize_model():
    if len(memory) < batch_size:
        return
    
    # retrieve all memeory
    transition = memory.sample(batch_size)
    batch = Transition(*zip(*transition))

    # find none terminal next state
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    # organize retrieved all memeory
    s_batch = torch.cat(batch.state)
    a_batch = torch.cat(batch.action)
    r_batch = torch.cat(batch.reward)

    state_action_values = π_net(s_batch).gather(1, a_batch.unsqueeze(1)).squeeze(1)
    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = b_net(non_final_next_states).max(1).values

    expected_state_action_values = (next_state_values * γ) + r_batch

    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values)
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(π_net.parameters(), 100)
    optimizer.step()


save_folder = r'C:\Users\wzhou\Desktop\Model\model_folder'
for i_episode in range(num_episodes):
    s = env.reset()
    s = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
    
    for t in count():
        # guess action
        a = select_action(s)  # action selection
        ŝ, R, T, truncated, __ = env.step(a)  # reward function design
        R = torch.tensor([R], device=device)
        done = T or truncated

        if done:
            s = None
        else:
            s = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)

        # remeber    
        memory.push(s, a, ŝ, R)
        s = ŝ

        # learn
        optimize_model()

        # Update the target network
        b_net_dict = b_net.state_dict()
        π_net_dict = π_net.state_dict()

        for key in π_net_dict:
            b_net_dict[key] = τ * π_net_dict[key] + (1 - τ) * b_net_dict[key]
        b_net.load_state_dict(b_net_dict)

        if done:
            if i_episode % 10 == 0:
                torch.save(π_net.state_dict(), os.path.join(save_folder, f"π_net_ep{i_episode}.pt"))
            print(f"Episode {i_episode} finished after {t + 1} time steps")
            break