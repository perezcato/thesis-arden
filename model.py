from prepare_data import prepare_data
from arch import arch_model


def prepare_initial_variables():
    # Load the entire dataset
    btc_prices = prepare_data()

    # Select initial 3-year period (2015-2017) to estimate initial parameters
    initial_period_data = btc_prices[
        (btc_prices['DateTime'] >= '2015-01-01') &
        (btc_prices['DateTime'] <= '2017-12-31')
        ]

    returns_initial = initial_period_data['log_return'] * 100

    # Fit GARCH(1,1) model on the initial period
    garch_initial_model = arch_model(
        returns_initial, mean='Zero', vol='GARCH', p=1, q=1, dist='normal'
    )
    garch_initial_results = garch_initial_model.fit(disp='off')

    # Extract GARCH parameters
    garch_starting_values = garch_initial_results.params

    # Fit EGARCH(1,1) model on the initial period
    egarch_initial_model = arch_model(
        returns_initial, mean='Zero', vol='EGARCH', p=1, q=1, dist='t'
    )
    egarch_initial_results = egarch_initial_model.fit(disp='off')

    # Extract EGARCH parameters
    egarch_starting_values = egarch_initial_results.params

    return garch_starting_values, egarch_starting_values


def model_fit(data=None):
    btc_prices = prepare_data() if data is None else data

    # Explicitly scale returns for numerical stability
    returns = btc_prices['log_return'] * 100

    # Optimized GARCH(1,1)
    garch_model = arch_model(
        returns, mean='Zero', vol='GARCH', p=1, q=1, dist='normal'
    )
    garch_results = garch_model.fit(
        disp='off', options={'maxiter': 5000, 'ftol': 1e-6}
    )

    # Optimized EGARCH(1,1)
    egarch_model = arch_model(
        returns, mean='Zero', vol='EGARCH', p=1, q=1, dist='normal'
    )
    egarch_results = egarch_model.fit(
        disp='off', options={'maxiter': 5000, 'ftol': 1e-6}
    )

    return garch_results, egarch_results

