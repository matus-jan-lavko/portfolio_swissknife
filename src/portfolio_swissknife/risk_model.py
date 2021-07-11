import numpy as np
import pandas as pd
from .estimation import linear_factor_model
from .plotting import plot_rolling_beta
from .portfolio import Engine, Portfolio

class RiskModel(Engine):
    def __init__(self, portfolio: Portfolio, factors: list):
        #factors in the risk_model
        self.factors = factors
        self.portfolio = portfolio

        #Portfolio derived attributes
        self.dates = portfolio.dates[(portfolio.estimation_period - 1):] #offset for returns calculation

        self.set_period(self.portfolio.period)

        #Engine securities pointer
        #Engine securities pointer
        self.securities = self.factors

    def __call__(self):
        raise NotImplementedError

    def get_prices(self, frequency = 'daily'):
        super().get_prices(frequency = frequency)
        # todo fix dates if mismatched (TEST)
        if self.portfolio.custom_prices:
            port_dates = set(self.portfolio.dates)
            rm_dates = set(self.dates)
            if (len(port_dates) - len(rm_dates)) != 0:
                self.dates = port_dates.intersection(rm_dates)
                self.portfolio.prices = self.portfolio.prices.loc[self.dates]
                self.prices = self.prices.loc[self.dates]

                #calculate returns
                self.portfolio.returns = self.portfolio.prices.pct_change().dropna().to_numpy()
                self.returns = self.prices.pct_change().dropna().to_numpy()

    def rolling_backtest(self, method = 'linear', estimation_period = 252, window = 22, *args, **kwargs):
        #estimation_window
        self.estimation_period = estimation_period
        self.risk_backtest = {}

        if method == 'linear':
            for mod in self.portfolio.backtest.keys():
                self.risk_estimates = {'alpha': [],
                                       'beta': [],
                                       'residuals': [],
                                       'estimation_dates': []}

                for est in range(self.estimation_period, self.returns.shape[0], window):
                    Y_t = self.portfolio.backtest[mod]['returns'][est - self.estimation_period : est]
                    X_t = self.returns[est - self.estimation_period : est]
                    alpha_t, beta_t, residuals_t = linear_factor_model(Y_t, X_t, *args, **kwargs)

                    #append results
                    self.risk_estimates['alpha'].append(alpha_t)
                    self.risk_estimates['beta'].append(beta_t)
                    self.risk_estimates['residuals'].append(residuals_t)
                    self.risk_estimates['estimation_dates'].append(self.portfolio.dates[est])

                self.risk_estimates['alpha'] = np.vstack(self.risk_estimates['alpha'])
                self.risk_estimates['beta'] = np.vstack(self.risk_estimates['beta'])
                self.risk_estimates['residuals'] = np.array(self.risk_estimates['residuals'])
                self.risk_backtest[mod] = self.risk_estimates

        elif method == 'PCA':
            raise NotImplementedError


    def get_risk_report(self, model: str):
        #prepare the df
        df_t = {}
        #rolling beta
        for mod in self.risk_backtest.keys():
            df_t[mod] = self.risk_backtest[mod]['beta']
        df_t = pd.DataFrame(df_t[model], columns=self.factors,
                          index=self.risk_backtest[model]['estimation_dates'])
        #all models end exposure
        df_end = pd.DataFrame({mod: self.risk_backtest[mod]['beta'][-1] for mod in self.risk_backtest.keys()},
                              index=self.factors)

        df_end.plot(kind='barh', figsize =(15,10), title = f'Current Exposure')
        plot_rolling_beta(df_t)

    def _estimate_panel(self, method = 'linear', *args, **kwargs):
        #some checks
        assert self.returns is not None
        assert self.returns.shape[0] == self.portfolio.returns.shape[0]

        if method == 'linear':
            Y_i_t = self.portfolio.returns
            X_i_t = self.returns

            alpha_t, beta_t, residuals_t = linear_factor_model(Y_i_t, X_i_t, *args, **kwargs)

        elif method == 'PCA':
            raise NotImplementedError

        return alpha_t, beta_t, residuals_t












