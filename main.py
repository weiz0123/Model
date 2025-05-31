import vectorbt as vbt
print("vectorbt version:", vbt.__version__)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

from utils.eda_utils import dataset_info_summary, system_info
from data.Data_PipLine import data_from_csv
import strategy.SimpleStrategy as ss
import strategy.PyramidFamilyStrategy as pf
import numpy as np

QQU =r"C:\Users\wzhou\Desktop\Model\data\qqu\QQU ETF Stock Price History.csv"
QQU_df = data_from_csv(QQU)
QQU_price = QQU_df['Close']

SOXL =r"C:\Users\wzhou\Desktop\Model\data\SOXL\SOXL ETF Stock Price History (1).csv"
SOXL_df = data_from_csv(SOXL)
SOXL_price = SOXL_df['Close']


TQQQ =r"C:\Users\wzhou\Desktop\Model\data\tqqq\TQQQ ETF Stock Price History (3).csv"
TQQQ_df= data_from_csv(TQQQ)
TQQQ_price = TQQQ_df['Close']


SP500 =r"C:\Users\wzhou\Desktop\Model\data\sp500\S&P 500 Historical Data (1).csv"
SP500_df= data_from_csv(SP500)
SP500_price = SP500_df['Close']

def compare_strategies(**strategies):
    portfolios = []
    legends = []

    for name, strategy in strategies.items():
        try:
            strategy.run()
            portfolios.append(strategy.portfolio)
            legends.append(name)
        except Exception as e:
            print(f"Error running strategy '{name}': {e}")

    if not portfolios:
        raise ValueError("No valid portfolios to compare.")

    fig = go.Figure()

    for pf, label in zip(portfolios, legends):
            fig.add_trace(go.Scatter(
                x=pf.value().index,
                y=pf.value(),
                mode='lines',
                name=label
            ))

    fig.update_layout(
        title="📈 Strategy Equity Curve Comparison",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        template="plotly_dark",
        height=800
    )

    # 📋 Optional: Stats comparison
    stats_df = pd.concat(
        [pf.stats().rename(lambda name: f"{legends[i]}: {name}") for i, pf in enumerate(portfolios)],
        axis=1
    )
    # print(stats_df)

    return fig


def run_plot(**strategies):
    for name, stradegy in  strategies.items():
        try:
            stradegy.run()
            stradegy.plot().show()
        except :
            print(f"Error running strategy '{name}': {e}") # type: ignore
            continue


def simple_strategy_exe(symbol_price):
    sma = ss.SMACrossoverStrategy(symbol_price)
    rsi = ss.RSIStrategy(symbol_price)
    mom = ss.MomentumStrategy(symbol_price)

    run_plot(  
        SMA=sma,
        RSI=rsi,
        Momentum=mom)

    compare_strategies(
        SMA=sma,
        RSI=rsi,
        Momentum=mom
    ).show()
def pyramid_strategy_exe(symbol_price):
    fipy = pf.FiPyramidStrategy(  
            symbol_price,
            initial_drop_pct=0.03,
            fib_ratios=[1.0, 2.0, 3.33],      # corresponds to 3%, 6%, 10% drop
            buy_pcts=[0.02, 0.05, 0.10],      # corresponding buy sizes
            sell_rise_pct=0.05                # fixed sell target)
        )
    run_plot(fipy=fipy)

# simple_strategy_exe(QQU_price)
# simple_strategy_exe(SOXL_price)
# simple_strategy_exe(SP500_price)

pyramid_strategy_exe(QQU_price)