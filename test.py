import vectorbt as vbt
import pandas as pd
import numpy as np

print("vectorbt version:", vbt.__version__)

# Create dummy Series
price = pd.Series(np.random.rand(100), name='Close')
print(type(price))

# Try vectorbt rolling
fast_sma = price.rolling(window=5).mean()
print(fast_sma.head())
