from model import model_fit
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt
from scipy.stats import jarque_bera


def diagnostic_tests():
    garch_results, egarch_results = model_fit()
    # Standardized residuals for GARCH
    garch_std_resid = garch_results.resid / garch_results.conditional_volatility

    # Standardized residuals for EGARCH
    egarch_std_resid = egarch_results.resid / egarch_results.conditional_volatility

    # Ljung-Box Test for GARCH
    garch_lb_test = acorr_ljungbox(garch_std_resid, lags=[10, 20], return_df=True)
    print("GARCH Ljung-Box test:")
    print(garch_lb_test)

    # Ljung-Box Test for EGARCH
    egarch_lb_test = acorr_ljungbox(egarch_std_resid, lags=[10, 20], return_df=True)
    print("\nEGARCH Ljung-Box test:")
    print(egarch_lb_test)

    # Ljung-Box Test for squared residuals (GARCH)
    garch_lb_sq = acorr_ljungbox(garch_std_resid ** 2, lags=[10, 20], return_df=True)
    print("\nGARCH Ljung-Box test (squared residuals):")
    print(garch_lb_sq)

    # Ljung-Box Test for squared residuals (EGARCH)
    egarch_lb_sq = acorr_ljungbox(egarch_std_resid ** 2, lags=[10, 20], return_df=True)
    print("\nEGARCH Ljung-Box test (squared residuals):")
    print(egarch_lb_sq)

    # Jarque-Bera Test (GARCH)
    jb_garch = jarque_bera(garch_std_resid)
    print("\nGARCH Jarque-Bera test statistic and p-value:", jb_garch)

    # Jarque-Bera Test (EGARCH)
    jb_egarch = jarque_bera(egarch_std_resid)
    print("\nEGARCH Jarque-Bera test statistic and p-value:", jb_egarch)

    # GARCH parameters
    print("\nGARCH Parameters:")
    print(garch_results.params)

    # EGARCH parameters
    print("\nEGARCH Parameters:")
    print(egarch_results.params)

    # Plot standardized residuals for GARCH
    plt.figure(figsize=(10, 5))
    plt.plot(garch_std_resid, label='GARCH Residuals', color='blue')
    plt.title('GARCH Standardized Residuals')
    plt.legend()
    plt.show()

    # Plot standardized residuals for EGARCH
    plt.figure(figsize=(10, 5))
    plt.plot(egarch_std_resid, label='EGARCH Residuals', color='green')
    plt.title('EGARCH Standardized Residuals')
    plt.legend()
    plt.show()