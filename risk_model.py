import numpy as np
from estimation import linear_factor_model

from portfolio import Engine, Portfolio

class RiskModel(Engine):
    def __init__(self, portfolio: Portfolio, factors: list):
        #factors in the risk_model
        self.factors = factors
        self.portfolio = portfolio

        #Portfolio derived attributes
        self.dates = portfolio.dates[(portfolio.estimation_period - 1):] #offset for returns calculation

        self.set_period((self.dates[0].strftime('%Y-%m-%d'), self.dates[-1].strftime('%Y-%m-%d')))

        #Engine securities pointer
        self.securities = self.factors

    def __call__(self):
        raise NotImplementedError

    def rolling_backtest(self, method = 'linear', estimation_period = 252, window = 22, *args, **kwargs):
        #estimation_window
        self.estimation_period = estimation_period
        self.risk_backtest = {}
        #todo implement rolling factor mod estimation
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

        else:
            raise NotImplementedError




