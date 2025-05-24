import vectorbt as vbt
print("vectorbt version:", vbt.__version__)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.eda_utils import dataset_info_summary, system_info
from data.Data_PipLine import data_from_csv
import strategy.SimpleStrategy as ss
import plotly.graph_objects as go
QQU = r"C:\Users\wzhou\Desktop\Model\data\qqu\QQU ETF Stock Price History.csv"
df = data_from_csv(QQU)
price = df['Close']


def compare_strategies(**strategies):
    portfolios = []
    legends = []

    for name, strategy in strategies.items():
        try:
            strategy.run()
            portfolios.append(strategy.portfolio)
            legends.append(name)
        except Exception as e:
            print(f"❌ Error running strategy '{name}': {e}")

    if not portfolios:
        raise ValueError("No valid portfolios to compare.")

    # ✅ Create a Plotly figure
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

    fig.show()

    # 📋 Optional: Stats comparison
    stats_df = pd.concat(
        [pf.stats().rename(lambda name: f"{legends[i]}: {name}") for i, pf in enumerate(portfolios)],
        axis=1
    )
    print(stats_df)

    return fig
def compare_strategies_list(export_csv: str = None, **strategies):
    portfolios = []
    legends = []

    for name, strat in strategies.items():
        try:
            strat.run()
            portfolios.append(strat.portfolio)
            legends.append(name)
        except Exception as e:
            print(f"⚠️ Error running strategy '{name}': {e}")

    if not portfolios:
        raise ValueError("No valid portfolios to compare.")

    # 📊 Plot comparison
    fig = vbt.Portfolio.plot_comparison(
        portfolios,
        labels=legends,
        title="📈 Strategy Comparison",
        template="plotly_dark",
        height=800
    )
    fig.show()

    # 📋 Stats comparison
    stats_df = pd.concat(
        [pf.stats().rename(lambda col: f"{legends[i]}: {col}") for i, pf in enumerate(portfolios)],
        axis=1
    )

    if export_csv:
        stats_df.to_csv(export_csv)
        print(f"📁 Stats exported to: {export_csv}")

    return stats_df
sma = ss.SMACrossoverStrategy(price)
rsi = ss.RSIStrategy(price)
mom = ss.MomentumStrategy(price)

compare_strategies(
    SMA=sma,
    RSI=rsi,
    Momentum=mom
).show()