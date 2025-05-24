import vectorbt as vbt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def data_from_csv(file_path: str):

    df = pd.read_csv(file_path)

    # Convert 'Date' to datetime and set as index
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    df.set_index('Date', inplace=True)

    # Remove commas and convert columns
    if df['Vol.'].notna().any():
        df['Vol.'] = df['Vol.'].str.replace('M', 'e6').str.replace('K', 'e3').str.replace(',', '')
        df['Vol.'] = df['Vol.'].astype(float)
    else:
        df['Vol.'] = 0.0
    df['Change %'] = df['Change %'].str.replace('%', '', regex=False).str.replace(',', '').astype(float) / 100


    # Convert remaining numeric columns
    cols_to_float = ['Price', 'Open', 'High', 'Low']
    df[cols_to_float] = df[cols_to_float].replace(',', '', regex=True).astype(float)

    # Optional: rename columns for vectorbt convention
    df.rename(columns={
        'Price': 'Close',
        'Vol.': 'Volume'
    }, inplace=True)

    # Sort by date (oldest to newest)
    df = df.sort_index()

    return df