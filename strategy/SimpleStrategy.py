# strategies/sma_crossover.py
from strategy.BaseStrategy import BaseStrategy
import vectorbt as vbt
import pandas as pd

class SMACrossoverStrategy(BaseStrategy):
    def __init__(self, price, fast_window=9, slow_window=20):
        self.fast_window = fast_window
        self.slow_window = slow_window
        super().__init__(price)

    def run(self):
        sma_fast = self.price.rolling(self.fast_window).mean()
        sma_slow = self.price.rolling(self.slow_window).mean()
        self.entries = sma_fast > sma_slow
        self.exits = sma_fast < sma_slow
        self.backtest()


class RSIStrategy(BaseStrategy):
    def __init__(self, price, rsi_window=14, entry_level=30, exit_level=70):
        self.rsi_window = rsi_window
        self.entry_level = entry_level
        self.exit_level = exit_level
        super().__init__(price)

    def run(self):
        rsi = vbt.RSI.run(self.price, window=self.rsi_window)
        self.entries = rsi.rsi < self.entry_level
        self.exits = rsi.rsi > self.exit_level
        self.backtest()




class BollingerStrategy(BaseStrategy):
    def __init__(self, price, window=20, std_mult=2.0):
        self.window = window
        self.std_mult = std_mult
        super().__init__(price)

    def run(self):
        bb = vbt.BBANDS.run(
            close=self.price,     # ✅ explicitly name it `close`
            window=self.window,
            std=self.std_mult
        )
        self.entries = self.price < bb.lower
        self.exits = self.price > bb.upper
        self.backtest()

class MomentumStrategy(BaseStrategy):
    def __init__(self, price, lag=5):
        self.lag = lag
        super().__init__(price)

    def run(self):
        momentum = self.price - self.price.shift(self.lag)
        self.entries = momentum > 0
        self.exits = momentum < 0
        self.backtest()
