import numpy as np
import pandas as pd
import yfinance as yf
import time
from .metrics import *
from .optimization import *
from .plotting import *
from .estimation import *

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
        '''
        Sets the period that the user wants to analyse bound to the swissknife object

        :param period: tuple of dates with tuple[0] being the start and tuple[1] being the end of the period: tuple
        :return: None
        '''

        self.period = period

    #class methods
    def get_prices(self, frequency = 'daily'):
        '''
        Pulls prices from yfinance at the requested frequency

        :param frequency: granularity of prices such 'daily', 'monthly' or other : str
        :return: None
        '''

        try:
            assert (self.period is not None)
            self.prices = yf.download(self.securities,
                                      start = self.period[0],
                                      end = self.period[1],
                                      frequency = frequency)
            self.prices = self.prices.loc[:, ('Adj Close', slice(None))]
            self.returns = self.prices.pct_change().dropna().to_numpy()
            self.dates = self.prices.index[1:]
            self.custom_prices = False

            if frequency == 'daily':
                self.trading_days = 252
            elif frequency == 'monthly':
                self.trading_days = 12

        #todo implement checker - no missing data!

        except AssertionError:
            print('You need to provide start and end dates!')

    def set_custom_prices(self, df, frequency = 'daily'):
        '''
        Sets custom prices for the swissknife object from a locally loaded source

        :param df: table of prices to be loaded: pd.DataFrame
        :param frequency: granularity of prices such as 'daily': str
        :return: None
        '''
        self.prices = df
        self.period = (df.index[0].strftime('%Y-%m-%d'), df.index[-1].strftime('%Y-%m-%d'))
        self.returns = self.prices.pct_change().dropna().to_numpy()
        self.dates = self.prices.index[1:]
        self.estimation_period = 0 #initialize to 0 until a method sets it to x > 0
        self.custom_prices = True


        if frequency == 'daily':
            self.trading_days = 252
        elif frequency == 'monthly':
            self.trading_days = 12
        else:
            raise ValueError('This frequency is not supported yet!')

    def _get_state(self, t_0, t_1):
        '''
        Returns a slice of returns from the attribute self.returns

        :param t_0: first index: int
        :param t_1: second index: int
        :return: sliced array of returns: np.array
        '''

        #slicing the engine data structure
        assert t_0 <= t_1
        return self.returns[t_0 : t_1, :]

class Portfolio(Engine):
    '''
    Portfolio class that encapsulates all the data needed for an analysis of a user-defined portfolio. Provides a way
    of examining different weighing strategies, visualisation tools and backtest of a portfolio that features fixed
    set of securities.
    '''
    def __init__(self, securities : list, start_weights = None):
        '''

        :param securities: tickers to be included in the portfolio: list of str
        :param start_weights: optional starting weights initialized to equal weights: np array
        '''
        super().__init__(securities)

        #initialize estimation methods to simple returns
        self.estimation_method = [mean_return_historic, sample_cov]

        if start_weights is not None:
            assert len(start_weights) == self.size
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

    #todo create functions for custom benchmarks and discounts
    def set_benchmark(self, benchmark: str):
        '''
        Sets a benchmark that the portfolio is compared to such as the SPY.

        :param benchmark: ticker for the benchmark to be pulled from yfinance
        :return: None
        '''

        self.benchmark = yf.download(benchmark,
                                     start = self.period[0],
                                     end = self.period[1])
        self.benchmark = self.benchmark.loc[:,'Adj Close']
        self.benchmark = self.benchmark.iloc[(-len(self.dates)-1):]
        self.benchmark = self.benchmark.pct_change().dropna().to_numpy()

    def set_discount(self, discount: str):
        '''
        Sets the discount rate that defines the opportunity set of the investor. The discount is used in all evaluation
        calculations as the main financing rate.

        :param discount: ticker of discount to be pulled from yfinance
        :return: None
        '''

        self.discount = yf.download(discount,
                                    start = self.period[0],
                                    end = self.period[1])
        self.discount = self.discount.loc[:, 'Adj Close'].reindex(index=self.dates).fillna(method='ffill')
        self.discount = self.discount.to_numpy()
        #
        self.discount /= 100

    #todo implement transaction cost model
    def set_transaction_cost(self, transaction_cost = '0.005'):
        '''
        Sets the transaction cost that will be used in the fixed transaction cost model

        :param transaction_cost: fixed transaction cost: int
        :return: None
        '''
        self.transaction_cost = transaction_cost

    def set_estimation_method(self, function, moment: int):
        '''
        Sets the estimation method for the selected statistical moment to be estimated.

        :param function: function that estimates the moment: function
        :param moment: selected moment in non-pythonic indexing: int
        :return: None
        '''
        self.estimation_method[moment - 1] = function

    def set_constraints(self, constraint_dict = None, default = True):
        '''
        Sets convex constraints for the optimizers.

        :param constraint_dict: dictionary of constraints
        :param default: flag for if to use default long-only normalized portfolio: bool
        :return: None
        '''
        if default:
            self.constraints = {'long_only': True,
                                'leverage': 1,
                                'normalizing': True}
        else:
            self.constraints = constraint_dict

    def historical_backtest(self, models = ['EW', 'GMV', 'RP'], frequency = 22,
                 estimation_period = 252, *args, **kwargs):
        '''
        Conducts a historical backtest in order to see an ex-post performance of a set of securities based on
        pre-selected weighing schemes. A tool to validate investment ideas in a non-experimental manner. This should
        not serve as any promise of future performance but rather a sanity check for the investor to help them
        understand how would a portfolio perform if projected on the past prices.

        :param models: list of models that can be used. Currently supported models are:
             - Equal Weights (EW)
             - Global Minimum Variance (GMV)
             - Equal Risk Contribution (RP)
             - Maximum Diversification Ratio (MDR)
             - Maximum Sharpe Ratio (MSR)
             - Minimum Expected Shortfall (MES)
             - Minimum Maximum Drawdown (MDD)

            To be added:
             - Minimum Skew (MS)
             - Hierarchical Risk Parity (HRP)


        :param frequency: window used for rebalancing the portfolio in number of trading days: int
        :param estimation_period: number of trading days used for estimating parameters in the models: int
        :return: None
        '''

        #caching backtest attributes
        self.weighting_models = models
        self.estimation_period = estimation_period
        self.efficient_frontier = None

        self.backtest = {}
        self.estimates = {'exp_value' : [],
                          'cov_matrix' : []}

        self.estimates = self._rolling_estimate(self.estimates, self.estimation_period, frequency)

        #backtest logic
        for model in models:
            model_results = {'weights': [],
                             'returns': [],
                             'trade_dates':[],
                             'w_change': [],
                             'gamma': [],
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
                    gamma = 1
                #mean-variance specifications
                elif model == 'MSR':
                    ir_kwargs = {'r_f' : self.discount[trade - self.estimation_period: trade],
                                 'num_periods' : self.trading_days,
                                 'ratio_type': 'sharpe'}
                    w_t, gamma = self._rebalance(mu , sigma, w_prev, model, r_est = r_est, maximum = True,
                                                 function = information_ratio, function_kwargs = ir_kwargs)
                elif model == 'MES':
                    var_kwargs = {'alpha' : 0.05,
                                  'exp_shortfall': True,
                                  'dist': 't'}
                    w_t, gamma = self._rebalance(mu , sigma, w_prev, model, r_est = r_est, maximum = False,
                                                 function = var, function_kwargs = var_kwargs)
                elif model == 'MDD':
                    w_t, gamma = self._rebalance(mu, sigma, w_prev, model, r_est = r_est, maximum = False,
                                                 function = max_drawdown, function_kwargs=None)

                else:
                    w_t, gamma = self._rebalance(mu, sigma, w_prev, model)

                #cache
                w_prev = w_t

                #todo implement transaction costs
                #get current returns and compute out of sample portfolio returns
                r_t = self._get_state(trade, trade + frequency)
                r_p = np.dot(r_t, w_t)
                model_results['returns'].append(r_p)
                model_results['weights'].append(w_t)

                #flatten hotfix
                w_delta = np.multiply(w_t.flatten(), np.cumprod(1 + r_t, axis = 1)[-1]).flatten()
                if len(model_results['weights']) > 1:
                    w_chg = w_delta - w_prev.flatten()
                else:
                    w_chg = w_delta

                model_results['w_change'].append(w_chg)
                model_results['trade_dates'].append(self.dates[trade])
                model_results['gamma'].append(gamma)
                num_rebalance += 1

            model_results['returns'] = np.vstack(model_results['returns'])
            toc = time.perf_counter()
            model_results['opt_time'] = toc - tic
            self.backtest[model] = model_results

    def get_backtest_report(self, display_weights = True, *args, **kwargs):
        '''
        Displays the basic risk and performance measures of the backtest together with weights.

        :param display_weights: flag for displaying weights plots: bool
        :return: None
        '''
        # adjust benchmark and discount
        if hasattr(self, 'benchmark'):
            self.benchmark = self.benchmark[-self.backtest['EW']['returns'].shape[0]:]
            self.discount = self.discount[-self.backtest['EW']['returns'].shape[0]:]

        #construct the dataframes
        bt_rets =  pd.DataFrame({mod : self.backtest[mod]['returns'].flatten()
                                 for mod in self.weighting_models},
                                index = self.dates[self.estimation_period:])
        bt_gamma = pd.DataFrame({mod: self.backtest[mod]['gamma']
                                 for mod in self.weighting_models})
        bt_gamma_mean = bt_gamma.mean()
        bt_rets_cum = (1+bt_rets).cumprod()
        bmark_rets_cum = (1+self.benchmark).cumprod()

        #prepare weights
        w_change_all = {mod : self.backtest[mod]['w_change'] for mod in self.weighting_models}

        plot_returns(bt_rets_cum, bmark_rets_cum, *args, **kwargs)

        stats = portfolio_summary(bt_rets, self.discount,
                                  self.benchmark, w_change = w_change_all,
                                  num_periods = self.trading_days, gamma = bt_gamma_mean)
        display(stats)

        #plot the weights
        if display_weights:

            bt_weights = {}
            for mod in self.weighting_models:
                #exclude ew
                if mod != 'EW':
                    bt_weights[mod] = pd.DataFrame(np.concatenate(self.backtest[mod]['weights'],axis=1).T,
                                                   columns = self.securities, index = self.backtest[mod]['trade_dates'])

            #plot the returns
            plot_weights(bt_weights, self.weighting_models, *args, **kwargs)



    def _rebalance(self, mu, sigma, w_prev,
                   opt_problem: str, *args, **kwargs):
        '''
        Helper function that acts in one specific point in time in order to rebalance the portfolio.

        :param mu: estimate of the first moment: np.array
        :param sigma: estimate of the second moment: np.array
        :param w_prev: cached weights from the previous window: np.array
        :param opt_problem: token for the optimization problem, see historical_backtest : str
        :return: optimized weights: np.array
        '''
        #solve efficient frontier


        if opt_problem == 'MSR' or opt_problem == 'cVAR' or opt_problem == 'MDD' or opt_problem == 'MES':
            self.efficient_frontier = quadratic_risk_utility(mu, sigma, self.constraints,
                                                                  self.size, 100)

        #solve problems
        if opt_problem == 'EW':
            w_opt = np.full((self.size, 1), 1/self.size)
            gamma = 1
        if opt_problem == 'GMV':
            gamma = 1
            w_opt = global_minimum_variance(sigma, self.constraints, self.size)
        if opt_problem == 'RP':
            gamma = 1
            w_opt = risk_parity(sigma,self.constraints, self.size)
        if opt_problem == 'MDR':
            w_opt = max_diversification_ratio(sigma, w_prev, self.constraints)
            gamma = 1
        if opt_problem == 'MSR':
            w_opt, gamma = greedy_optimization(self.efficient_frontier, *args, **kwargs)
        if opt_problem == 'MES':
            w_opt, gamma = greedy_optimization(self.efficient_frontier, *args, **kwargs)
        if opt_problem == 'MDD':
            w_opt, gamma = greedy_optimization(self.efficient_frontier, *args, **kwargs)


        w_opt = w_opt.reshape(self.size, 1)
        return w_opt, gamma

    def _rolling_estimate(self, estimates_dict, estimation_period, frequency, *args, **kwargs):
        '''
        Helper function for estimating the necessary parameters for the optimization models

        :param estimates_dict: dictionary of estimates: dict
        :param estimation_period: period for the estimates to be calculated on: int
        :param frequency: window used for re-estimation of the parameters: int
        :return: estimates of all requested moments: dict
        '''
        for trade in range(estimation_period, self.returns.shape[0], frequency):
            # estimate necessary params
            r_est = super()._get_state(trade - estimation_period, trade)

            for idx, moment in enumerate(estimates_dict.keys()):
                estimates_dict[moment].append(self._estimate(self.estimation_method[idx],
                                                             r_est, *args, **kwargs))
        return estimates_dict

    def _estimate(self, estimator, r_est, *args, **kwargs):
        moment = estimator(r_est, *args, **kwargs)
        return moment

class FactorPortfolio(Portfolio):
    '''
    A class extending the main Portfolio class that is used for the analysis of a factor that is defined in the
    RiskModel class. Depends on a defined investment universe that is defined as an instance of a Portfolio object
    and a RiskModel instance. Features a dynamic selection backtest with changing sets of securities each rebalance
    window.

    '''
    def __init__(self, universe: Portfolio, risk_model, factor: str, start_weights = None):
        '''

        :param universe: portfolio defining the investment universe: Portfolio
        :param risk_model: risk model applied on the investment universe: RiskModel
        :param factor: selected factor to be examined and backtested: str
        :param start_weights: optional starting weights: np array
        '''

        self.universe = universe
        self.risk_model = risk_model
        self.returns = self.universe.returns
        self.estimation_method = [mean_return_historic, sample_cov]
        self.dates = self.risk_model.dates
        self.trading_days = self.universe.trading_days
        self.period = self.risk_model.period

        if factor in risk_model.factors:
            self.factor_idx = self.risk_model.factors.index(factor)
        else:
            raise ValueError('Factor not specified in your model!')

        self.size =  self.size = len(self.risk_model.asset_selection['top_idx'][0][:, self.factor_idx])

        if start_weights:
            self.start_weights = start_weights
        else:
            self.start_weights = np.empty(self.size, dtype = float)
            self.start_weights.fill(1/self.size)

    def _get_state(self, t_0, t_1, filter):
        '''
        Returns a slice of returns from the attribute self.returns with a defined filter on the securities to
        be returned

        :param t_0: first index: int
        :param t_1: second index: int

        :param filter: indexes of securities to be returned: list
        :return:
        '''
        return super(FactorPortfolio, self)._get_state(t_0, t_1)[:, filter]

    def historical_backtest(self, models = ['EW', 'GMV', 'RP'], long_only = True,
                            long_exposure = 1, short_exposure = 0.3, frequency = 22,
                            estimation_period = 252, *args, **kwargs):
        '''
        Conducts a historical backtest as a long-only or a long-short spread of the factor portfolio.

        :param models: list of models that can be used. Currently supported models are:
         - Equal Weights (EW)
         - Global Minimum Variance (GMV)
         - Equal Risk Contribution (RP)
         - Maximum Sharpe Ratio (MSR)
         - Minimum Expected Shortfall (MES)
         - Minimum Maximum Drawdown (MDD)

        To be added:
         - Minimum Skew (MS)
         - Hierarchical Risk Parity (HRP)


        :param frequency: window used for rebalancing the portfolio in number of trading days: int
        :param estimation_period: number of trading days used for estimating parameters in the models: int
        :return: None
        '''

        self.weighting_models = models
        self.estimation_period = estimation_period

        self.backtest = {}
        self.estimates_top = {'exp_value' : [],
                              'cov_matrix' : []}
        self.estimates_bottom = {'exp_value' : [],
                                 'cov_matrix' : []}

        self.estimates_top = self._rolling_estimate(self.estimates_top, self.estimation_period, frequency,
                                                    self.risk_model.asset_selection['top_idx'])

        # HOTFIX monthly estimation
        diff = len(self.dates) - self.estimation_period
        while diff % len(self.risk_model.asset_selection['top_idx']) != 0:
            diff += 1
            self.estimation_period += 1

        # backtest logic
        for model in models:
            model_results = {'returns': [],
                             'weights': [],
                             'trade_dates': [],
                             'gamma': [],
                             'w_change': [],
                             'opt_time': 0.}
            tic = time.perf_counter()
            num_rebalance = 0

            for trade in range(self.estimation_period, self.returns.shape[0], frequency):
                # solve optimization starting with start_weights

                try:
                    idx_selected_long = self.risk_model.asset_selection['top_idx'][num_rebalance][:, self.factor_idx]
                    idx_selected_short = self.risk_model.asset_selection['bottom_idx'][num_rebalance][:, self.factor_idx]
                except (KeyError, IndexError):
                    idx_selected_long = self.risk_model.asset_selection['top_idx'][num_rebalance]
                    idx_selected_short = self.risk_model.asset_selection['bottom_idx'][num_rebalance]

                mu = self.estimates_top['exp_value'][num_rebalance]
                sigma = self.estimates_top['cov_matrix'][num_rebalance]
                r_est = self._get_state(trade - estimation_period, trade, idx_selected_long)

                # self.size = len(self.risk_model.asset_selection['top_idx'][num_rebalance][:, self.factor_idx])
                #todo finish the backtest!!!

                if num_rebalance == 0:
                    w_t = self.start_weights
                    w_prev = w_t
                    gamma = 1

                elif model == 'MDR':
                    raise ValueError('MDR is not supported for FactorPortfolio.')
                # mean-variance specifications
                elif model == 'MSR':
                    ir_kwargs = {'r_f': self.discount[trade - self.estimation_period: trade],
                                 'num_periods': self.trading_days,
                                 'ratio_type': 'sharpe'}
                    w_t, gamma = self._rebalance(mu, sigma, w_prev, model, r_est=r_est, maximum=True,
                                          function=information_ratio, function_kwargs=ir_kwargs)
                elif model == 'MES':
                    var_kwargs = {'alpha': 0.05,
                                  'exp_shortfall': True,
                                  'dist': 't'}
                    w_t, gamma = self._rebalance(mu, sigma, w_prev, model, r_est=r_est, maximum=False,
                                          function=var, function_kwargs=var_kwargs)
                elif model == 'MDD':
                    w_t, gamma = self._rebalance(mu, sigma, w_prev, model, r_est=r_est, maximum=False,
                                          function=max_drawdown, function_kwargs=None)

                else:
                    w_t, gamma = self._rebalance(mu, sigma, w_prev, model)

                # cache
                w_prev = w_t

                # todo implement transaction costs
                # get current returns and compute out of sample portfolio returns
                r_t_long = self._get_state(trade, trade + frequency, idx_selected_long)
                r_p_long = np.dot(r_t_long, w_t)

                # equal weight short side TODO: implement different weighing schemes and estimation
                if not long_only:
                    r_t_short = self._get_state(trade, trade + frequency, idx_selected_short)
                    w_t_short, gamma = self._rebalance(mu, sigma, w_prev, opt_problem = 'EW')
                    w_t_short *= -1
                    r_p_short = np.dot(r_t_short, w_t_short)
                    r_p = long_exposure*r_p_long.flatten() + short_exposure*r_p_short.flatten()
                else:
                    r_p = long_exposure*r_p_long

                # TODO fix turnover calculation to match changing securities
                w_delta = np.multiply(w_t.flatten(), np.cumprod(1 + r_t_long, axis=1)[-1]).flatten()
                if len(model_results['weights']) > 1:
                    w_chg = w_delta - w_prev.flatten()
                else:
                    w_chg = w_delta

                model_results['returns'].append(r_p.flatten())
                model_results['weights'].append(w_t)
                model_results['gamma'].append(gamma)
                model_results['w_change'].append(w_chg)
                model_results['trade_dates'].append(self.dates[trade])
                num_rebalance += 1

            model_results['returns'] = np.hstack(model_results['returns'])
            toc = time.perf_counter()
            model_results['opt_time'] = toc - tic
            self.backtest[model] = model_results

    @staticmethod
    def plot_compare_factors(portfolio_list: list):
        raise NotImplementedError

    def _rolling_estimate(self, estimates_dict, estimation_period, frequency,
                          security_filter, *args, **kwargs):
        '''
        Helper function for estimating the necessary parameters for the optimization models

        :param estimates_dict: dictionary of estimates: dict
        :param estimation_period: period for the estimates to be calculated on: int
        :param frequency: window used for re-estimation of the parameters: int
        :param security_filter: filter for the securities to be estimated: list
        :return: estimates of all requested moments: dict
        '''

        counter = 0
        for trade in range(estimation_period, self.returns.shape[0], frequency):
            # estimate necessary params
            try:
                idx_selected = security_filter[counter][:, self.factor_idx]
            except (KeyError, IndexError):
                try:
                    idx_selected = security_filter[counter]
                except IndexError:
                    #this gets thrown when there is monthly data from the model and days don't check out
                    #HOTFIX
                    pass

            r_est = self._get_state(trade - estimation_period, trade, idx_selected)

            for idx, moment in enumerate(estimates_dict.keys()):
                estimates_dict[moment].append(self._estimate(self.estimation_method[idx],
                                                             r_est, *args, **kwargs))

            counter += 1
        return estimates_dict

class CustomPortfolio(Portfolio):
    '''
    Implements a CustomPortfolio class that allows the user to analyze a custom strategy that has been backtested
    outside of the swissknife package.

    '''
    #todo implement
    def __init__(self, bt_rets: pd.DataFrame):
        '''
        :param bt_rets: backtest of the returns of the strategy: pd.DataFrame
        '''

        self.backtest = bt_rets


class MLPortfolio(FactorPortfolio):
    '''

    ML Portfolio class that takes in a prespecified model and allows the user to backtest an ML strategy within the
    portfolio framework.
    '''

    def __init__(self, universe: Portfolio, prediction_model, start_weights = None):
        self.universe = universe
        self.estimation_method = [mean_return_historic, sample_cov]
        # self.dates = self.risk_model.dates
        self.trading_days = self.universe.trading_days
        self.risk_model = prediction_model
        self.returns = self.universe.returns
        # self.returns = self.risk_model.returns.values
        self.period = self.risk_model.period

        self.dates = self.universe.dates
        self.size = len(self.risk_model.asset_selection['top_idx'][0])
        self.factor_idx = 0

        if start_weights:
            self.start_weights = start_weights
        else:
            self.start_weights = np.empty(self.size, dtype = float)
            self.start_weights.fill(1/self.size)




















