import numpy as np
import yfinance as yf
import estimation as est
import optimization as opt

class Engine:
    def __init__(self, securities: list):
        self.securities = securities
        self.size = int(len(self.securities))
        self.prices = None
        self.returns = None
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
            self.returns = self.prices.pct_change().dropna().to_numpy()
        except AssertionError:
            print('You need to provide start and end dates!')

    def _get_state(self, t_0, t_1):
        #slicing the engine data structure
        assert t_0 <= t_1
        return self.returns[t_0 : t_1, :]

class Portfolio(Engine):
    def __init__(self, securities : list, start_weights = None):
        super().__init__(securities)

        self.start_weights = start_weights

        if self.start_weights is not None:
            self.start_weights = start_weights
        else:
            #initialize to equal weights
            self.start_weights = np.empty(self.size, dtype=float)
            self.start_weights.fill(1/self.size)

    def __call__(self):
        return f'This is a Portfolio spanning from {self.period[0]} to {self.period[1]}.' \
               f' It consists of {self.size} securities.'

    def __len__(self):
        raise NotImplementedError

    def set_transaction_cost(self, transaction_cost = '0.005'):
        self.transaction_cost = transaction_cost

    def backtest(self, models = ['EW'], frequency = 22, estimation_period = 252):
        self.backtest = {}
        for model in models:
            model_results = {'weights': [],
                             'returns': [],
                             'opt time': 0.}
            #todo implement timer!!
            num_rebalance = 0
            #estimation warmup

            for trade in range(estimation_period, self.returns.shape[0], frequency):
                #get current prices
                p_t = self._get_state(trade, trade + frequency)
                #solve optimization
                w_t = self._rebalance(model)
                #todo implementing transaction costs
                r_t = np.dot(p_t, w_t)

                model_results['returns'].append(r_t)
                model_results['weights'].append(w_t)
                num_rebalance += 1
            model_results['returns'] = np.vstack(model_results['returns'])
            model_results['weights'] = np.vstack(model_results['weights'])
            self.backtest[model] = model_results



    def _rebalance(self, opt_problem: str, **kwargs):
        if opt_problem == 'EW':
            w_opt = np.full((self.size, 1), 1/self.size)
        if opt_problem == 'GMV':
            w_opt = opt.global_minimum_variance()

        else:
            raise ValueError

        return w_opt

    def _estimate(self, moment):
        raise NotImplementedError








