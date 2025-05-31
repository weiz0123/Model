from strategy.BaseStrategy import BaseStrategy
import pandas as pd
import numpy as np



class FiPyramidStrategy(BaseStrategy):
    def __init__(self, price: pd.Series, 
                initial_drop_pct: float = 0.03, 
                fib_ratios: list = None,
                buy_pcts: list = None,
                sell_rise_pct: float = 0.05):
        """
        :param price: Price series (pd.Series)
        :param initial_drop_pct: Base drop from peak to start buying (e.g. 0.03 = 3%)
        :param fib_ratios: Fibonacci ratios to define drawdown levels (e.g. [1.0, 2.0, 3.33] = 3%, 6%, 10%)
        :param buy_pcts: Capital allocation per level (e.g. [0.02, 0.05, 0.1])
        :param sell_rise_pct: % rise from entry price to trigger a sell (e.g. 0.05 = 5%)
        """
        super().__init__(price)
        
        self.initial_drop_pct = initial_drop_pct
        self.fib_ratios = fib_ratios or [1.0, 1.618, 2.618]
        
        # Default buy_pcts increases with level (e.g., 2%, 4%, 6%)
        if buy_pcts is None:
            self.buy_pcts = [0.02 * (i + 1) for i in range(len(self.fib_ratios))]
        else:
            self.buy_pcts = buy_pcts

        if len(self.fib_ratios) != len(self.buy_pcts):
            raise ValueError("fib_ratios and buy_pcts must be of equal length.")

        self.sell_rise_pct = sell_rise_pct


    def run(self):
        peak = self.price.cummax()
        drawdown = (peak - self.price) / peak
        
        entries = pd.Series(False, index=self.price.index)
        exits = pd.Series(False, index=self.price.index)

        # Define drop levels (thresholds) and matching buy sizes
        fib_levels = self.fib_ratios  # e.g. [0.236, 0.382, 0.618, 1.0]
        drop_thresholds = [self.initial_drop_pct * r for r in fib_levels]
        buy_sizes = self.buy_pcts  # e.g. [0.02, 0.05, 0.1, 0.2]

        active_entries = []

        for level_drop, buy_size in zip(drop_thresholds, buy_sizes):
            entry_signal = (drawdown >= level_drop) & (~entries)
            new_entries = self.price[entry_signal]

            for entry_time, entry_price in new_entries.items():
                active_entries.append({
                    'time': entry_time,
                    'price': entry_price,
                    'buy_size': buy_size
                })
                entries[entry_time] = True

        # Determine exits
        for entry in active_entries:
            entry_time = entry['time']
            entry_price = entry['price']
            future_prices = self.price[self.price.index > entry_time]
            target_price = entry_price * (1 + self.sell_rise_pct)

            exit_hit = future_prices[future_prices >= target_price]
            if not exit_hit.empty:
                first_exit_time = exit_hit.index[0]
                exits[first_exit_time] = True

        self.entries = entries
        self.exits = exits
        self.backtest()
