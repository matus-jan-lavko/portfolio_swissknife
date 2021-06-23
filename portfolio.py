import numpy as np
import yfinance as yf

class portfolio:
    def __init__(self, securities: list, weights = None):
        self.securities = securities
        self.size = int(len(self.securities))
        self.weights = None
        self.prices = None
        self.period = None

        if self.weights is not None:
            self.weights = weights
        else:
            #initialize to equal weights
            self.weights = np.empty(self.size, dtype=float)
            self.weights.fill(1/self.size)

    def __add__(self, other):
        raise NotImplementedError

    def __sub__(self, other):
        raise NotImplementedError

    def set_period(self, period: tuple):
        self.period = period

    #class methods
    def get_prices(self, frequency):
        try:
            assert (self.period is not None)
            self.prices = yf.download(self.securities,
                                      start = self.period[0],
                                      end = self.period[1])
            self.prices = self.prices.loc[:, ('Adj Close', slice(None))]
        except AssertionError:
            print('You need to provide start and end dates!')