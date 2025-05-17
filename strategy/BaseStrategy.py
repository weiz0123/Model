# strategies/base.py

import vectorbt as vbt
import pandas as pd

class BaseStrategy:
    def __init__(self, price: pd.Series):
        self.price = price
        self.entries = None
        self.exits = None
        self.portfolio = None

    def run(self):
        """To be implemented in child class"""
        raise NotImplementedError

    def backtest(self):
        self.portfolio = vbt.Portfolio.from_signals(
            self.price,
            self.entries,
            self.exits
        )

    def stats(self):
        if self.portfolio is None:
            raise ValueError("Backtest not yet run.")
        return self.portfolio.stats()

    def plot(self):
        if self.portfolio is None:
            raise ValueError("Backtest not yet run.")
        return self.portfolio.plot()
