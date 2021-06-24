import numpy as np
import yfinance as yf

class Engine:
    def __init__(self, securities: list):
        self.securities = securities
        self.size = int(len(self.securities))
        self.prices = None
        self.period = None

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
            self.returns = self.prices.pct_change()
        except AssertionError:
            print('You need to provide start and end dates!')

    def _get_state(self, t_0, t_1):
        #slicing the engine data structure
        assert t_0 <= t_1
        return self.returns.iloc[t_0 : t_1]

class Portfolio(Engine):
    def __init__(self, securities : list, start_weights = None):
        super().__init__(securities)

        self.start_weights = start_weights

        if self.start_weights is not None:
            self.start_weights = start_weights
        else:
            #initialize to equal weights
            self.weights = np.empty(self.size, dtype=float)
            self.weights.fill(1/self.size)

    def set_transaction_cost(self, transaction_cost = '0.005'):
        self.transaction_cost = transaction_cost






