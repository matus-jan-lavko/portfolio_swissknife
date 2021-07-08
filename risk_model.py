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

    def estimate(self, method: str, estimation_period = 252):
        #estimation_window
        self.estimation_period = estimation_period
        #todo implement rolling factor mod estimation
        raise NotImplementedError


        


