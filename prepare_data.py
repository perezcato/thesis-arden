import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

def prepare_data():
    file_paths = [
        'data_daily/2015.csv',
        'data_daily/2016.csv',
        'data_daily/2017.csv',
        'data_daily/2018.csv',
        'data_daily/2019.csv',
        'data_daily/2020.csv',
        'data_daily/2021.csv',
        'data_daily/2022.csv',
        'data_daily/2023.csv',
        'data_daily/2024.csv',
        'data_daily/2025.csv'
    ]

    btc_data_frames = [pd.read_csv(file) for file in file_paths]
    btc_prices = pd.concat(btc_data_frames, ignore_index=True)

    # Data cleaning and preprocessing explicitly stated
    btc_prices['time'] = pd.to_datetime(btc_prices['time'])
    btc_prices.sort_values(by='time', inplace=True)

    # Keep necessary columns explicitly
    btc_prices = btc_prices[['time', 'price_open', 'price_high', 'price_low', 'price_close']]
    btc_prices.rename(columns={
        'time': 'DateTime',
        'price_open': 'Open',
        'price_high': 'High',
        'price_low': 'Low',
        'price_close': 'Close'
    }, inplace=True)

    # Explicitly handle missing values
    btc_prices.dropna(subset=['Close'], inplace=True)

    # Explicit computation and handling of log returns with outlier removal
    btc_prices['log_return'] = np.log(btc_prices['Close'] / btc_prices['Close'].shift(1))

    # Explicit outlier removal (extreme returns >30%)
    btc_prices = btc_prices[btc_prices['log_return'].abs() <= 0.3]

    # Drop remaining NaN values after filtering
    btc_prices.dropna(inplace=True)

    return btc_prices

def get_descriptive_stats():
    # Load prepared data
    btc_prices = prepare_data()

    # Ensure your returns are correctly scaled (log returns multiplied by 100 if applicable)
    returns = btc_prices['log_return']

    # Compute Descriptive Statistics explicitly
    mean_return = returns.mean()
    median_return = returns.median()
    std_return = returns.std()
    skewness_return = skew(returns)
    kurtosis_return = kurtosis(returns, fisher=True)  # Fisher's kurtosis (excess kurtosis)

    # Summarize in a clear table
    descriptive_stats = pd.DataFrame({
        'Statistic': ['Mean', 'Median', 'Standard Deviation', 'Skewness', 'Excess Kurtosis'],
        'Value': [mean_return, median_return, std_return, skewness_return, kurtosis_return]
    })

    print(descriptive_stats)
