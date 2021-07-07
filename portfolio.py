import numpy as np
import pandas as pd
import yfinance as yf
import time

import estimation as est
import optimization as opt
import plotting
from metrics import portfolio_summary
from metrics import information_ratio, var, max_drawdown

class Engine:
    '''
    Initializes the Engine superclass that supersedes both the Portfolio and Risk Model classes. Defines the general
    data structure that fetches and stores data and retrieves states. Also sets the period for analysis that is
    encapsulated within the class and a new class has to be instantiated in order to carry out analysis in different
    time frames.
    '''
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
    def get_prices(self, frequency = 'daily'):
        try:
            assert (self.period is not None)
            self.prices = yf.download(self.securities,
                                      start = self.period[0],
                                      end = self.period[1],
                                      frequency = frequency)
            self.prices = self.prices.loc[:, ('Adj Close', slice(None))]
            self.returns = self.prices.pct_change().dropna().to_numpy()
            self.dates = self.prices.index[1:]

            if frequency == 'daily':
                self.trading_days = 252
            elif frequency == 'monthly':
                self.trading_days = 12

        except AssertionError:
            print('You need to provide start and end dates!')

    def set_custom_prices(self, frequency = 'daily'):
        raise NotImplementedError

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
        self.benchmark = None

    def __call__(self):
        return f'This is a Portfolio spanning from {self.period[0]} to {self.period[1]}.' \
               f' It consists of {self.size} securities.'

    def __len__(self):
        raise NotImplementedError

    def set_benchmark(self, benchmark: str):
        self.benchmark = yf.download(benchmark,
                                     start = self.period[0],
                                     end = self.period[1])
        self.benchmark = self.benchmark.loc[:,'Adj Close']
        self.benchmark = self.benchmark.pct_change().dropna().to_numpy()

    def set_discount(self, discount: str):
        self.discount = yf.download(discount,
                                    start = self.period[0],
                                    end = self.period[1])
        self.discount = self.discount.loc[:, 'Adj Close'].reindex(index=self.dates).fillna(method='ffill')
        self.discount = self.discount.to_numpy()
        #
        self.discount /= 100

    def set_transaction_cost(self, transaction_cost = '0.005'):
        self.transaction_cost = transaction_cost

    def set_estimation_method(self, moment: int, function):
        self.estimation_method[moment] = function

    #todo implement constraints (long-only, leverage, weight etc.)
    def set_constraints(self, constraint_dict = None, default = True):
        if default:
            self.constraints = {'long_only': True,
                                'leverage': 1,
                                'normalizing': True}
        else:
            self.constraints = constraint_dict

    def historical_backtest(self, models = ['EW', 'GMV', 'RP'], frequency = 22,
                 estimation_period = 252, *args, **kwargs):
        #caching backtest attributes
        self.weighting_models = models
        self.estimation_period = estimation_period

        self.backtest = {}
        self.estimates = {'exp_value' : [],
                          'cov_matrix' : []}

        #estimation
        for trade in range(estimation_period, self.returns.shape[0], frequency):
            # estimate necessary params
            r_est = self._get_state(trade - estimation_period, trade)

            self.estimates['exp_value'].append(self._estimate(self.estimation_method[0],
                                                              r_est , *args, **kwargs))

            self.estimates['cov_matrix'].append(self._estimate(self.estimation_method[1],
                                                               r_est , *args, **kwargs))

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
                r_est = self._get_state(trade - estimation_period, trade)

                if num_rebalance == 0:
                    w_t = self.start_weights
                    w_prev = w_t

                #mean-variance specifications
                elif model == 'MSR':
                    ir_kwargs = {'r_f' : self.discount[trade - self.estimation_period: trade],
                                 'num_periods' : self.trading_days,
                                 'ratio_type': 'sharpe'}
                    w_t = self._rebalance(mu , sigma, w_prev, model, r_est = r_est, maximum = True,
                                          function = information_ratio, function_kwargs = ir_kwargs)
                elif model == 'MES':
                    var_kwargs = {'alpha' : 0.05,
                                  'exp_shortfall': True,
                                  'dist': 't'}
                    w_t = self._rebalance(mu , sigma, w_prev, model, r_est = r_est, maximum = False,
                                          function = var, function_kwargs = var_kwargs)
                elif model == 'MDD':
                    w_t = self._rebalance(mu, sigma, w_prev, model, r_est = r_est, maximum = False,
                                          function = max_drawdown, function_kwargs=None)

                else:
                    w_t = self._rebalance(mu, sigma, w_prev, model)
                    #cache
                w_prev = w_t

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

    def get_backtest_report(self, *args, **kwargs):
        #construct the dataframe
        bt_rets =  pd.DataFrame({mod : self.backtest[mod]['returns'].flatten()
                                 for mod in self.weighting_models},
                                index = self.dates[self.estimation_period:])
        bt_rets_cum = (1+bt_rets).cumprod()
        bmark_rets_cum = (1+self.benchmark[self.estimation_period:]).cumprod()

        bt_weights = {}
        for mod in self.weighting_models:
            #exclude ew
            if mod != 'EW':
                bt_weights[mod] = pd.DataFrame(np.concatenate(self.backtest[mod]['weights'],axis=1).T,
                                               columns = self.securities)

        #plot the returns
        plotting.plot_returns(bt_rets_cum, bmark_rets_cum)
        stats = portfolio_summary(bt_rets, self.discount[self.estimation_period:],
                          self.benchmark[self.estimation_period:], self.trading_days)
        display(stats)
        #plot the weights
        plotting.plot_weights(bt_weights, self.weighting_models, *args, **kwargs)

    def _rebalance(self, mu, sigma, w_prev,
                   opt_problem: str, *args, **kwargs):

        #solve efficient frontier

        if opt_problem == 'MSR' or opt_problem == 'cVAR' or opt_problem == 'MDD':
            self.efficient_frontier = opt._quadratic_risk_utility(mu, sigma, self.constraints,
                                                                  self.size, 100)
        #solve problems
        if opt_problem == 'EW':
            w_opt = np.full((self.size, 1), 1/self.size)
        if opt_problem == 'GMV':
            w_opt = opt.global_minimum_variance(sigma, self.constraints, self.size)
        if opt_problem == 'RP':
            w_opt = opt.risk_parity(sigma,self.constraints, self.size)
        if opt_problem == 'MDR':
            w_opt = opt.max_diversification_ratio(sigma, w_prev, self.constraints)
        if opt_problem == 'MSR':
            w_opt = opt.greedy_optimization(self.efficient_frontier, *args, **kwargs)
        if opt_problem == 'MES':
            w_opt = opt.greedy_optimization(self.efficient_frontier, *args, **kwargs)
        if opt_problem == 'MDD':
            w_opt = opt.greedy_optimization(self.efficient_frontier, *args, **kwargs)


        w_opt = w_opt.reshape(self.size, 1)
        return w_opt

    def _estimate(self, estimator, p_est, *args, **kwargs):
        moment = estimator(p_est, *args, **kwargs)
        return moment










