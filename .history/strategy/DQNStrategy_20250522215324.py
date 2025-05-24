import numpy as np
from strategy.BaseStrategy import BaseStrategy

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, lr=1e-3, batch_size=32, memory_size=10000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        q_values = self.model(states).gather(1, actions)
        next_q_values = self.model(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones.float()) * self.gamma * next_q_values

        loss = self.criterion(q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class DQNTrainer:
    def __init__(self, state_size, action_size, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.agent = DQNAgent(state_size, action_size)
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay

    def train(self, env_data, episodes=100):
        for e in range(episodes):
            total_reward = 0
            for state, action, reward, next_state, done in env_data:
                action_taken = self.agent.act(state, self.epsilon)
                self.agent.remember(state, action_taken, reward, next_state, done)
                self.agent.replay()
                total_reward += reward
                if done:
                    break
            print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.3f}")
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def get_trained_agent(self):
        return self.agent


# Helper: Convert vectorbt data into RL-friendly environment-like format
def make_rl_data(price_series, window=10):
    X = []
    for i in range(len(price_series) - window - 1):
        state = price_series[i:i+window].values
        next_state = price_series[i+1:i+1+window].values
        reward = price_series[i+window+1] - price_series[i+window]  # price diff
        done = i + window + 1 >= len(price_series) - 1
        X.append((state, 1 if reward > 0 else 0, reward, next_state, done))
    return X


class DQNStrategy(BaseStrategy):
    def __init__(self, price, window=10, episodes=50):
        super().__init__(price)
        self.window = window
        self.episodes = episodes

    def run(self):
        # Prepare RL environment data
        env_data = make_rl_data(self.price, window=self.window)
        state_size = self.window
        action_size = 2  # 0 = hold, 1 = buy

        # Train the model
        trainer = DQNTrainer(state_size, action_size)
        trainer.train(env_data, episodes=self.episodes)
        agent = trainer.get_trained_agent()

        # Generate signals
        entries = np.zeros(len(self.price), dtype=bool)
        exits = np.zeros(len(self.price), dtype=bool)

        for i in range(self.window, len(self.price) - 1):
            state = self.price[i - self.window:i].values
            action = agent.act(state, epsilon=0)  # deterministic during inference
            if action == 1:
                entries[i] = True
            else:
                exits[i] = True  # simple inverse logic, improve as needed

        self.entries = entries
        self.exits = exits
