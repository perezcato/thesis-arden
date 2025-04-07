import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
import matplotlib.pyplot as plt

from model import model_fit
from prepare_data import prepare_data


def generate_forecast_metrics():
    # Load the latest forecast data after epsilon adjustment
    forecast_df_latest = pd.read_csv('volatility_forecast.csv', index_col='Date')

    # Recalculate aggregated metrics explicitly
    mean_mse_garch_latest = forecast_df_latest['MSE_GARCH'].mean()
    mean_mse_egarch_latest = forecast_df_latest['MSE_EGARCH'].mean()

    mean_rmse_garch_latest = np.sqrt(mean_mse_garch_latest)
    mean_rmse_egarch_latest = np.sqrt(mean_mse_egarch_latest)

    mean_mae_garch_latest = forecast_df_latest['MAE_GARCH'].mean()
    mean_mae_egarch_latest = forecast_df_latest['MAE_EGARCH'].mean()

    mean_qlike_garch_latest = forecast_df_latest['QLIKE_GARCH'].mean()
    mean_qlike_egarch_latest = forecast_df_latest['QLIKE_EGARCH'].mean()

    # Create and explicitly present the final summary table
    summary_table_final_eps = pd.DataFrame({
        'Metric': ['Mean MSE', 'Mean RMSE', 'Mean MAE', 'Mean QLIKE'],
        'GARCH(1,1)': [mean_mse_garch_latest, mean_rmse_garch_latest, mean_mae_garch_latest, mean_qlike_garch_latest],
        'EGARCH(1,1)': [mean_mse_egarch_latest, mean_rmse_egarch_latest, mean_mae_egarch_latest,
                        mean_qlike_egarch_latest]
    })

    print(summary_table_final_eps)

def generate_volatility_cluster_vrs_forecasts():
    # Load forecast data explicitly
    forecast_df = pd.read_csv('volatility_forecast.csv', parse_dates=['Date'], index_col='Date')

    # Load your returns data explicitly
    btc_prices = prepare_data()
    btc_prices.set_index('DateTime', inplace=True)

    # Ensure alignment of data
    returns = btc_prices.loc[forecast_df.index, 'log_return']

    # Realized volatility (absolute returns as a volatility proxy)
    realized_volatility = returns.abs()

    realized_volatility.to_csv('realized_volatility.csv', encoding='utf-8')

    plt.figure(figsize=(14, 7))

    # Plot realized volatility
    plt.plot(realized_volatility.index, realized_volatility, label='Realized Volatility (|Returns|)', alpha=0.7)

    # Plot GARCH forecasts explicitly
    plt.plot(forecast_df.index, forecast_df['GARCH_Volatility'], label='GARCH Forecasted Volatility', alpha=0.8)

    # Plot EGARCH forecasts explicitly
    plt.plot(forecast_df.index, forecast_df['EGARCH_Volatility'], label='EGARCH Forecasted Volatility', alpha=0.8)

    plt.title('Figure 4.4: Volatility Clusters and Forecasted vs Actual Volatility')
    plt.xlabel('Date')
    plt.ylabel('Volatility (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()




def rolling_window_volatility():
    btc_prices = prepare_data()

    # Prepare your data (daily returns scaled by 100)
    returns = btc_prices['log_return'] * 100
    dates = btc_prices['DateTime']

    # Adjusted rolling-window parameters for daily data
    window_size = 500  # good choice for daily data (approx. 2 years)
    forecast_horizon = 1  # forecast 1 day ahead

    # Containers to hold forecasts
    garch_forecasts, egarch_forecasts, forecast_dates = [], [], []
    mse_garch, mse_egarch, mae_garch, mae_egarch, qlike_garch, qlike_egarch = [], [], [], [], [], []

    # Perform rolling-window forecasting
    for i in range(window_size, len(returns)):
        # Select daily returns directly for efficiency
        train_data = btc_prices.iloc[i - window_size:i]
        test_return = returns.iloc[i:i + forecast_horizon].values ** 2

        garch_results, egarch_results = model_fit(train_data)

        # Forecast volatility
        garch_forecast = garch_results.forecast(horizon=forecast_horizon).variance.iloc[-1, 0]
        egarch_forecast = egarch_results.forecast(horizon=forecast_horizon).variance.iloc[-1, 0]

        garch_forecasts.append(np.sqrt(garch_forecast))
        egarch_forecasts.append(np.sqrt(egarch_forecast))

        # Record forecast date
        forecast_dates.append(dates.iloc[i])

        # Accuracy metrics
        mse_garch.append(mean_squared_error(test_return, [garch_forecast]))
        mse_egarch.append(mean_squared_error(test_return, [egarch_forecast]))
        mae_garch.append(mean_absolute_error(test_return, [garch_forecast]))
        mae_egarch.append(mean_absolute_error(test_return, [egarch_forecast]))

        epsilon = 1e-5  # Recommended stable epsilon
        qlike_garch.append(np.log(garch_forecast + epsilon) + test_return[0] / (garch_forecast + epsilon))
        qlike_egarch.append(np.log(egarch_forecast + epsilon) + test_return[0] / (egarch_forecast + epsilon))

        # Diagnostics
        garch_resid = garch_results.std_resid.dropna()
        egarch_resid = egarch_results.std_resid.dropna()

        ljungbox_garch = acorr_ljungbox(garch_resid, lags=[10], return_df=True)['lb_pvalue'].iloc[0]
        ljungbox_egarch = acorr_ljungbox(egarch_resid, lags=[10], return_df=True)['lb_pvalue'].iloc[0]

        arch_test_garch = het_arch(garch_resid)[1]
        arch_test_egarch = het_arch(egarch_resid)[1]

    # Create DataFrame with forecasts
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'GARCH_Volatility': garch_forecasts,
        'EGARCH_Volatility': egarch_forecasts,
        'MSE_GARCH': mse_garch,
        'MSE_EGARCH': mse_egarch,
        'MAE_GARCH': mae_garch,
        'MAE_EGARCH': mae_egarch,
        'QLIKE_GARCH': qlike_garch,
        'QLIKE_EGARCH': qlike_egarch
    }).set_index('Date')

    # Save forecasts
    forecast_df.to_csv('volatility_forecast.csv', encoding='utf-8')

    # Plot forecasted volatility
    plt.figure(figsize=(12, 6))
    forecast_df['GARCH_Volatility'].plot(label='GARCH(1,1) Forecast')
    plt.title('GARCH Rolling-Window Volatility Forecasts (Daily Data)')
    plt.xlabel('Date')
    plt.ylabel('Forecasted Volatility (%)')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    forecast_df['EGARCH_Volatility'].plot(label='EGARCH(1,1) Forecast')
    plt.title('EGARCH Rolling-Window Volatility Forecasts (Daily Data)')
    plt.xlabel('Date')
    plt.ylabel('Forecasted Volatility (%)')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    forecast_df[['GARCH_Volatility', 'EGARCH_Volatility']].plot()
    plt.title('GARCH vs EGARCH Rolling-Window Volatility Forecasts (Daily Data)')
    plt.xlabel('Date')
    plt.ylabel('Forecasted Volatility (%)')
    plt.legend(['GARCH(1,1)', 'EGARCH(1,1)'])
    plt.show()

    generate_forecast_metrics()
    generate_volatility_cluster_vrs_forecasts()

