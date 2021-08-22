import numpy as np
import pandas as pd
from tqdm import notebook
from .estimation import linear_factor_model
from .plotting import plot_rolling_beta
from .portfolio import Portfolio, Engine
from .utils import DataHandler

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline


class RiskModel(Engine):
    '''
    A class that defines a risk model to be applied on a particular portfolio in order to assess risks and develop
    a pricing model.

    '''
    def __init__(self, portfolio: Portfolio, factors = None):
        '''

        :param portfolio: portfolio to analyze: Portfolio
        :param factors: tickers or names of factors to be included: list of str
        '''
        # factors in the risk_model
        if isinstance(factors, list):
            self.factors = factors
        elif not factors:
            self.factors = None
        else:
            raise ValueError('Factors need to be a list.')

        self.portfolio = portfolio

        # Portfolio derived attributes
        self.dates = portfolio.dates[(portfolio.estimation_period - 1):]  # offset for returns calculation

        self.set_period(self.portfolio.period)

        # Engine securities pointer
        self.securities = self.factors

    def __call__(self):
        raise NotImplementedError

    def get_prices(self, frequency='daily'):
        '''
        Pulls prices from yfinance and matches with the dates and indexes in the Portfolio object if different.

        :param frequency: granularity of the time series: str
        :return: None
        '''
        super().get_prices(frequency=frequency)
        self.prices = self.prices.dropna(axis=0)
        # todo fix dates if mismatched (TEST)
        if self.portfolio.custom_prices:
            port_dates = set(self.portfolio.dates)
            rm_dates = set(self.dates)
            if (len(port_dates) - len(rm_dates)) != 0:
                self.dates = port_dates.intersection(rm_dates)
                self.portfolio.prices = self.portfolio.prices.loc[self.dates]
                self.prices = self.prices.loc[self.dates]

                # calculate returns
                self.portfolio.returns = self.portfolio.prices.sort_index().pct_change().dropna().to_numpy()
                self.returns = self.prices.pct_change().sort_index().dropna().to_numpy()

                #fix dates
                self.dates = list(self.dates)
                self.dates.sort()
                self.dates = self.dates[-len(self.returns):]

    @DataHandler
    def rolling_factor_exposure(self, method='linear', estimation_period=252, window=22, *args, **kwargs):
        '''
        Calculates the rolling risk exposure to the predefined factors based on a selected model.

        :param method: model to be used such as 'linear' or 'PCA': str
        :param estimation_period: window for estimating the model looking back: int
        :param window: window for re-estimating the model: int
        :return: None
        '''
        self.estimation_period = estimation_period
        self.risk_backtest = {}

        if method == 'linear':
            for mod in self.portfolio.backtest.keys():
                self.risk_estimates = {'alpha': [],
                                       'beta': [],
                                       'residuals': [],
                                       'estimation_dates': []}

                for est in range(self.estimation_period, self.returns.shape[0], window):
                    Y_t = self.portfolio.backtest[mod]['returns'][est - self.estimation_period: est]
                    X_t = self._get_state(est - self.estimation_period, est)
                    alpha_t, beta_t, residuals_t = linear_factor_model(Y_t, X_t, *args, **kwargs)

                    # append results
                    self.risk_estimates['alpha'].append(alpha_t)
                    self.risk_estimates['beta'].append(beta_t)
                    self.risk_estimates['residuals'].append(residuals_t)
                    self.risk_estimates['estimation_dates'].append(self.portfolio.dates[est])

                self.risk_estimates['alpha'] = np.vstack(self.risk_estimates['alpha'])
                self.risk_estimates['beta'] = np.vstack(self.risk_estimates['beta'])
                self.risk_estimates['residuals'] = np.array(self.risk_estimates['residuals'], dtype=object)
                self.risk_backtest[mod] = self.risk_estimates

        elif method == 'PCA':
            for mod in self.portfolio.backtest.keys():
                self.risk_estimates = {'principal_components': [],
                                       'explained_variance_ratio': [],
                                       'singular_values': [],
                                       'alpha': [],
                                       'beta': [],
                                       'residuals': [],
                                       'estimation_dates': []}

                for est in range(self.estimation_period, self.portfolio.returns.shape[0], window):

                    #decompose
                    X_t = self.portfolio._get_state(est - self.estimation_period, est)
                    pca = Pipeline([('scaling', StandardScaler()),
                                    ('pca', PCA(*args, **kwargs))])

                    pca.fit_transform(X_t.T)
                    
                    #append results
                    self.risk_estimates['principal_components'].append(pca[1].components_.T)
                    self.risk_estimates['explained_variance_ratio'].append(pca[1].explained_variance_ratio_)
                    self.risk_estimates['singular_values'].append(pca[1].singular_values_)

                    #estimate exposures
                    Y_t = self.portfolio.backtest[mod]['returns'][est - self.estimation_period: est]
                    PCA_t = self.risk_estimates['principal_components'][-1]

                    alpha_t, beta_t, residuals_t = linear_factor_model(Y_t, PCA_t[:Y_t.shape[0],:])

                    # append results
                    self.risk_estimates['alpha'].append(alpha_t)
                    self.risk_estimates['beta'].append(beta_t)
                    self.risk_estimates['residuals'].append(residuals_t)
                    self.risk_estimates['estimation_dates'].append(self.portfolio.dates[est])

                    #break if at end
                    if not self.factors:
                        self.factors = list(np.arange(1, pca[1].n_components + 1))
                        self.factors = list(map(lambda x: 'PC ' + str(x), self.factors))

                    if Y_t.shape[0] < self.estimation_period: break

                #process
                self.risk_estimates['alpha'] = np.vstack(self.risk_estimates['alpha'])
                self.risk_estimates['beta'] = np.vstack(self.risk_estimates['beta'])
                self.risk_estimates['residuals'] = np.array(self.risk_estimates['residuals'], dtype=object)
                self.risk_backtest[mod] = self.risk_estimates
        else:
            raise NotImplementedError


    def rolling_factor_selection(self, percentile: int, method='linear',
                                 estimation_period=252, window=22, *args, **kwargs):
        '''
        Function that creates a spread strategy by selecting securities that are in the top and bottom percentile
        in the current period for all of the periods in a rolling fashion.

        :param percentile: threshold for the percentile such as 10 for decile, 5 for quintile etc.: int
        :param method: method used for estimating the risk model: str
        :param estimation_period: window for estimating the model looking back: int
        :param window: window for re-estimating the risk model: int
        :return: None
        '''

        self.estimation_period = estimation_period
        self.asset_selection = {'top_securities': [],
                               'bottom_securities': [],
                               'top_idx': [],
                               'bottom_idx': []}

        if method == 'linear':
            for est in range(self.estimation_period, self.returns.shape[0], window):
                Y_i_t = self.portfolio._get_state(est - self.estimation_period, est)
                X_i_t = self._get_state(est - self.estimation_period, est)
                _, beta, _, = self._estimate_panel(Y_i_t, X_i_t, method, *args, **kwargs)

                sort2d = np.argsort(beta, axis=0)
                pct = int(len(self.portfolio.securities) / percentile)

                top_idx = sort2d[-pct:]
                bottom_idx = sort2d[:pct]
                top_q_names = np.array(self.portfolio.securities)[top_idx]
                bottom_q_names = np.array(self.portfolio.securities)[bottom_idx]

                self.asset_selection['top_idx'].append(top_idx)
                self.asset_selection['bottom_idx'].append(bottom_idx)
                self.asset_selection['top_securities'].append(top_q_names)
                self.asset_selection['bottom_securities'].append(bottom_q_names)

        else:
            raise NotImplementedError

    def get_risk_report(self, model: str):
        '''
        Displays the general plots and information about the risk model such as rolling betas and current exposure.

        :param model: name of the model: str
        :return: None
        '''
        # prepare the df
        df_t = {}
        # rolling beta
        for mod in self.risk_backtest.keys():
            df_t[mod] = self.risk_backtest[mod]['beta']

        df_t = pd.DataFrame(df_t[model], columns=self.factors,
                            index=self.risk_backtest[model]['estimation_dates'])
        # all models end exposure
        df_end = pd.DataFrame({mod: self.risk_backtest[mod]['beta'][-1] for mod in self.risk_backtest.keys()},
                              index=self.factors)

        df_end.plot(kind='barh', figsize=(15, 10), title=f'Current Exposure')
        plot_rolling_beta(df_t)

    def _estimate_panel(self, panel, factors, method='linear', *args, **kwargs):
        '''
        Helper function for estimating the model in a Fama-Macbeth methodology by estimating each cross-section
        separately in contrast to panel methods.

        :param panel: matrix of target securities: np.array
        :param factors: matrix of factors: np.array
        :param method: method of the model: str
        :return: alpha_t: intersects of the model: np.array,
                 beta_t: betas of the model : np.array,
                 residuals_t: residuals of the model: np.array
        '''
        # some checks
        assert self.returns is not None
        assert self.returns.shape[0] == self.portfolio.returns.shape[0]

        if method == 'linear':
            Y_i_t = panel
            X_i_t = factors

            alpha_t, beta_t, residuals_t = linear_factor_model(Y_i_t, X_i_t, *args, **kwargs)

        elif method == 'PCA':
            raise NotImplementedError

        return alpha_t, beta_t, residuals_t

class PredictionModel(Engine):
    '''

    A prediction model that allows the user to construct a strategy driven by a statistical model
    '''

    #needs to be decorated by datahandler
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
        self.features = None

        #portfolio derived attributes
        self.set_period(self.portfolio.period)

    def set_prediction_model(self, function):
        if not callable(function):
            raise TypeError('Prediction model needs to be of type FunctionType!')
        else:
            self.model = function

    def set_features(self, X):
        if not self.features:
            if isinstance(X, pd.DataFrame):
                #time series features
                self.features = {sec : X for sec in self.portfolio.securities}
            elif isinstance(X, dict):
                #panel features
                self.features = {sec : X[sec] for sec in X.keys()}
        else:
            if isinstance(X, pd.DataFrame):
                for sec in self.features.keys():
                    merged = pd.merge(self.features[sec], X,
                                      left_on = self.features[sec].index,
                                      right_on = X.index,
                                      how = 'left')
                    merged.index = merged['key_0']
                    merged = merged.drop('key_0', axis=1)
                    merged = merged.fillna(method='bfill').fillna(method='ffill')
                    self.features[sec] = merged
            elif isinstance(X, dict):
                for sec in self.features.keys():
                    merged = pd.merge(self.features[sec], X[sec],
                                      left_on=self.features[sec].index,
                                      right_on=X[sec].index,
                                      how='left')
                    merged.index = merged['key_0']
                    merged = merged.drop('key_0', axis=1)
                    merged = merged.fillna(method='bfill').fillna(method='ffill')
                    self.features[sec] = merged

    def prepare_targets(self, features_to_merge):
        merged = pd.merge(self.portfolio.prices, features_to_merge,
                          left_on= self.portfolio.prices.index,
                          right_on = features_to_merge.index,
                          how = 'right')
        merged.index = merged['key_0']
        merged = merged.drop('key_0', axis=1)
        merged = merged.fillna(method='bfill').fillna(method='ffill')
        self.prices = merged[self.portfolio.securities]
        self.returns = self.prices.pct_change().shift(-1).dropna()

    def rolling_model_prediction(self, estimation_period = 60, window = 1, *args, **kwargs):
        self.estimation_period = estimation_period
        self.prediction_measure = {}

        for stock in notebook.tqdm(self.portfolio.securities, desc = f'Training ML models'):
            #get y,X
            y_i = self.returns[stock]
            X_i = self.features[stock][:-1].dropna(axis=1) #adjust for last return
            X_i = X_i.loc[:, (X_i != 0).any(axis=0)].apply(lambda x: pd.to_numeric(x,downcast = 'float'))
            X_i.replace([np.inf, -np.inf], 0, inplace = True) #fix infs
            # print(y_i.shape, X_i.shape)
            preds = []

            #inner loop trains the model for each security
            for trade in range(self.estimation_period, X_i.shape[0], window):

                y_i_t = y_i.iloc[trade - self.estimation_period : trade]
                X_i_t = X_i.iloc[trade - self.estimation_period : trade]
                pred_i_t = self.model(y_i_t, X_i_t)
                preds.append(pred_i_t)

            self.prediction_measure[stock] = preds

    def load_pretrained_model(self, data):
        if not hasattr(self, 'prediction_measure'):
            self.prediction_measure = {}

        if isinstance(data, pd.DataFrame):
            data = data.to_dict()

        if isinstance(data, list):
            for dict in data:
                self.prediction_measure = {**self.prediction_measure, **dict}
        else:
            self.prediction_measure = {**self.prediction_measure, **data}


    def rolling_spread_selection(self, percentile, window = 1):

        self.prediction_measure = pd.DataFrame(self.prediction_measure)
        self.asset_selection = {'top_securities': [],
                               'bottom_securities': [],
                               'top_idx': [],
                               'bottom_idx': []}

        for est in range(0, self.prediction_measure.shape[0], window):
            sort = np.argsort(self.prediction_measure.iloc[est], axis = 0)
            pct = int(len(self.portfolio.securities) / percentile)

            top_idx = sort[-pct:]
            bottom_idx = sort[:pct]
            top_q_names = np.array(self.portfolio.securities)[top_idx]
            bottom_q_names = np.array(self.portfolio.securities)[bottom_idx]

            self.asset_selection['top_idx'].append(top_idx.values)
            self.asset_selection['bottom_idx'].append(bottom_idx.values)
            self.asset_selection['top_securities'].append(top_q_names)
            self.asset_selection['bottom_securities'].append(bottom_q_names)









