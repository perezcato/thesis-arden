import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt

from rolling_window_volatility import rolling_window_volatility


def prepare_data():
    file_paths = sorted(glob.glob('data_daily/*.csv'))
    btc_data_frames = [pd.read_csv(file) for file in file_paths]
    btc_prices = pd.concat(btc_data_frames, ignore_index=True)

    btc_prices['DateTime'] = pd.to_datetime(btc_prices['time'])
    btc_prices.sort_values(by='DateTime', inplace=True)
    btc_prices = btc_prices[['DateTime', 'price_open', 'price_high', 'price_low', 'price_close']]

    btc_prices.rename(columns={
        'price_open': 'Open',
        'price_high': 'High',
        'price_low': 'Low',
        'price_close': 'Close'
    }, inplace=True)

    btc_prices['log_return'] = np.log(btc_prices['Close']).diff()
    btc_prices.dropna(inplace=True)
    btc_prices.reset_index(drop=True, inplace=True)

    return btc_prices

def performance_metrics(portfolio_returns):
    sharpe = portfolio_returns.mean() / portfolio_returns.std()
    sortino = portfolio_returns.mean() / portfolio_returns[portfolio_returns < 0].std()
    cumulative = (1 + portfolio_returns / 100).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = cumulative / rolling_max - 1
    max_drawdown = drawdown.min()
    return sharpe, sortino, max_drawdown

def portfolio_optimization_and_evaluation():
    rolling_window_volatility()

    # Load your prepared Bitcoin daily price data
    btc_prices = prepare_data()

    # Load your daily rolling window volatility forecasts from CSV
    forecast_df = pd.read_csv('volatility_forecast.csv', parse_dates=['Date'], index_col='Date')

    # Align forecast data with returns data
    returns_df = btc_prices.set_index('DateTime').loc[forecast_df.index]
    returns_df['log_return'] *= 100  # Daily returns scaled (%)

    # Define a realistic daily target volatility (e.g., 5-7%)
    target_vol = 5.0  # Consider experimenting between 5% - 10%

    # Calculate dynamic weights based on daily volatility forecasts
    forecast_df['GARCH_Weight_BTC'] = (target_vol / forecast_df['GARCH_Volatility']).clip(upper=1.0)
    forecast_df['EGARCH_Weight_BTC'] = (target_vol / forecast_df['EGARCH_Volatility']).clip(upper=1.0)

    # Define daily risk-free return (assume 0%) and calculate cash weight
    risk_free_return = 0.0
    forecast_df['GARCH_Weight_Cash'] = 1 - forecast_df['GARCH_Weight_BTC']
    forecast_df['EGARCH_Weight_Cash'] = 1 - forecast_df['EGARCH_Weight_BTC']

    # Set appropriate daily stop-loss threshold (e.g., -4% daily)
    stop_loss_threshold = -4.0  # Adjusted to reflect daily volatility realistically

    # Calculate portfolio returns with daily stop-loss and diversification
    returns_df['GARCH_Portfolio_Return'] = np.where(
        returns_df['log_return'] <= stop_loss_threshold,
        risk_free_return,
        returns_df['log_return'] * forecast_df['GARCH_Weight_BTC'] +
        risk_free_return * forecast_df['GARCH_Weight_Cash']
    )

    returns_df['EGARCH_Portfolio_Return'] = np.where(
        returns_df['log_return'] <= stop_loss_threshold,
        risk_free_return,
        returns_df['log_return'] * forecast_df['EGARCH_Weight_BTC'] +
        risk_free_return * forecast_df['EGARCH_Weight_Cash']
    )

    # Calculate cumulative returns
    returns_df['GARCH_CumReturn'] = (1 + returns_df['GARCH_Portfolio_Return'] / 100).cumprod()
    returns_df['EGARCH_CumReturn'] = (1 + returns_df['EGARCH_Portfolio_Return'] / 100).cumprod()

    # Evaluate Portfolio Performance

    garch_sharpe, garch_sortino, garch_mdd = performance_metrics(returns_df['GARCH_Portfolio_Return'])
    egarch_sharpe, egarch_sortino, egarch_mdd = performance_metrics(returns_df['EGARCH_Portfolio_Return'])

    # Print daily results
    print("Daily GARCH Model Performance:")
    print(f"Sharpe Ratio: {garch_sharpe:.4f}")
    print(f"Sortino Ratio: {garch_sortino:.4f}")
    print(f"Maximum Drawdown: {garch_mdd:.4%}")

    print("\nDaily EGARCH Model Performance:")
    print(f"Sharpe Ratio: {egarch_sharpe:.4f}")
    print(f"Sortino Ratio: {egarch_sortino:.4f}")
    print(f"Maximum Drawdown: {egarch_mdd:.4%}")

    # Visualization of Daily Cumulative Returns
    plt.figure(figsize=(12, 6))
    returns_df['GARCH_CumReturn'].plot(label='GARCH Portfolio')
    returns_df['EGARCH_CumReturn'].plot(label='EGARCH Portfolio')
    plt.title('Daily Cumulative Portfolio Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.show()
