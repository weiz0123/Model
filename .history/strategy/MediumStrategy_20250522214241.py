import numpy as np
from strategies.base import BaseStrategy
from dqn_vectorbt_integration import DQNTrainer, make_rl_data

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
