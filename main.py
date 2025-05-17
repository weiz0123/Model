import vectorbt as vbt
print("vectorbt version:", vbt.__version__)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def data_from_csv(file_path: str):

    df = pd.read_csv(file_path)

    # Convert 'Date' to datetime and set as index
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    df.set_index('Date', inplace=True)

    # Remove commas and convert columns
    df['Vol.'] = df['Vol.'].str.replace('M', 'e6').str.replace('K', 'e3').str.replace(',', '')
    df['Vol.'] = df['Vol.'].astype(float)
    df['Change %'] = df['Change %'].str.replace('%', '').astype(float) / 100

    # Convert remaining numeric columns
    cols_to_float = ['Price', 'Open', 'High', 'Low']
    df[cols_to_float] = df[cols_to_float].astype(float)

    # Optional: rename columns for vectorbt convention
    df.rename(columns={
        'Price': 'Close',
        'Vol.': 'Volume'
    }, inplace=True)

    # Sort by date (oldest to newest)
    df = df.sort_index()

    return df

QQU = r"C:\Users\wzhou\Desktop\Model\data\qqu\QQU ETF Stock Price History.csv"

df = data_from_csv(QQU)


price = df['Close']

fast_sma = price.rolling(window=9).mean()
slow_sma = price.rolling(window=20).mean()


# print(df.head())          # First 5 rows
# print(df.tail())          # Last 5 rows
# print(price.head())       # Close price series preview
print(df.dtypes)
print(df.info())          # Column types, non-null counts
print(df.describe())      # Summary stats (mean, std, min, max, etc.)
print(df.columns)
print(df.isnull().sum())
print(df.corr())



sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(price, label='Close Price', linewidth=1)
plt.plot(fast_sma, label='9-day SMA', linestyle='--')
plt.plot(slow_sma, label='20-day SMA', linestyle='--')
plt.title('QQU ETF Price and SMA Crossover')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# entries = fast_sma > slow_sma
# exits = fast_sma < slow_sma

# portfolio = vbt.Portfolio.from_signals(price, entries, exits)
# portfolio.plot().show()