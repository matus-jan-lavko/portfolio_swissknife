import numpy as np
import yfinance as yf
import time
import cvxpy as cp

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
            self.dates = self.prices.index[1:]
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
        #initialize estimation methods to simple returns
        self.estimation_method = [est.mean_return_historic, est.sample_cov]

        if self.start_weights is not None:
            self.start_weights = start_weights
        else:
            #initialize to equal weights
            self.start_weights = np.empty(self.size, dtype=float)
            self.start_weights.fill(1/self.size)
        self.start_weights = self.start_weights.reshape(self.size, 1)

    def __call__(self):
        return f'This is a Portfolio spanning from {self.period[0]} to {self.period[1]}.' \
               f' It consists of {self.size} securities.'

    def __len__(self):
        raise NotImplementedError

    def set_transaction_cost(self, transaction_cost = '0.005'):
        self.transaction_cost = transaction_cost

    def set_estimation_method(self, moment: int, function):
        self.estimation_method[moment] = function

    #todo implement constraints (long-only, leverage, weight etc.)
    def set_constraints(self, constraint_dict: dict, default = True):
        if default:
            self.constraints = {'long_only': True,
                                'leverage': 1,
                                'normalizing': True}
        else:
            self.constraints = constraint_dict

    def historical_backtest(self, models = ['EW', 'GMV'], frequency = 22,
                 estimation_period = 252, *args, **kwargs):
        self.backtest = {}
        self.estimates = {'exp_value' : [],
                          'cov_matrix' : []}

        #estimation
        for trade in range(estimation_period, self.returns.shape[0], frequency):
            # estimate necessary params
            p_est = self._get_state(trade - estimation_period, trade)

            self.estimates['exp_value'].append(self._estimate(self.estimation_method[0],
                                                              p_est , *args, **kwargs))

            self.estimates['cov_matrix'].append(self._estimate(self.estimation_method[1],
                                                               p_est , *args, **kwargs))

        #backtest logic
        for model in models:
            model_results = {'weights': [],
                             'returns': [],
                             'opt_time': 0.}
            tic = time.perf_counter()
            num_rebalance = 0

            for trade in range(estimation_period, self.returns.shape[0], frequency):
                #solve optimization starting with start_weights
                mu = self.estimates['exp_value'][num_rebalance]
                sigma = self.estimates['cov_matrix'][num_rebalance]

                if num_rebalance == 0:
                    w_t = self.start_weights
                else:
                    w_t = self._rebalance(mu, sigma, model)
                #todo implement transaction costs
                #get current prices and compute returns
                p_t = self._get_state(trade, trade + frequency)
                r_t = np.dot(p_t, w_t)

                model_results['returns'].append(r_t)
                model_results['weights'].append(w_t)
                num_rebalance += 1

            model_results['returns'] = np.vstack(model_results['returns'])
            toc = time.perf_counter()
            model_results['opt_time'] = toc - tic
            self.backtest[model] = model_results

    def _rebalance(self, mu, sigma,
                   opt_problem: str):
        if opt_problem == 'EW':
            w_opt = np.full((self.size, 1), 1/self.size)
        if opt_problem == 'GMV':
            w_opt = opt.global_minimum_variance(sigma, self.constraints, self.size)
        w_opt = w_opt.reshape(self.size, 1)
        return w_opt

    def _estimate(self, estimator, p_est, *args, **kwargs):
        moment = estimator(p_est, *args, **kwargs)
        return moment










