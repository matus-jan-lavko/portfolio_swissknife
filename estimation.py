import numpy as np
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression

def mean_return_historic(r: np.array):
    '''
    Computes mean historical expected returns

    r: t x n np.array of returns

    out: 1 x n np.array of expected returns
    '''
    comp = np.prod(1 + r, axis = 0)
    nPer = r.shape[0]
    annr = comp ** (252 / nPer) - 1
    return annr


def ema_return_historic(r, window=22):
    '''
    Computes exponentially weighted historical returns

    r: t x n pd.DataFrame of returns
    window: window for calculating exponential moving average

    out: 1 x n np.array of expected returns
    '''
    ema = (1 + r.ewm(span=window).mean().iloc[-1])
    annr = ema ** 252 - 1
    return annr.values


def sample_cov(r):
    '''
    Sample empirical covariance estimator

    r: t x n pd.DataFrame of returns

    out: n x n np.array of covariances
    '''
    return np.cov(r, rowvar=False) * 252


def elton_gruber_cov(r):
    '''
    Constant correlation model of Elton and Gruber

    r: t x n pd.DataFrame of returns

    out: n x n np.array of covariances
    '''
    rho = r.corr()
    n_ = rho.shape[0]

    rho_bar = (rho.sum() - n_) / (n_ * (n_ - 1))
    ccor = np.full_like(rho, rho_bar)
    np.fill_diagonal(ccor, 1.)
    sig = r.std()
    ccov = ccor * np.outer(sig, sig)
    return ccov


def shrinkage_cov(r, delta=0.5, prior_model=elton_gruber_cov):
    '''
    Shrinks the sample covariance towards a specified model such as Elton/Gruber

    r: n x 1 returns
    delta: shrinkage parameter [0,1] #0.5 is standard
    prior_model: prior_model function that we shrink towards

    out: n x n shrunk covariance matrix
    '''

    prior = prior_model(r, **kwargs)
    sig_hat = sample_cov(r)

    # https://jpm.pm-research.com/content/30/4/110
    honey = delta * prior + (1 - delta) * sig_hat
    return honey

def linear_factor_model(Y, X, kernel = None, regularization = None):
    t_, n_ = Y.shape

    #setting weights
    if kernel is None:
        kernel = np.ones(t_) / t_

    m_Y = kernel @ Y
    m_X = kernel @ X

    Y_p = ((Y - m_Y).T * np.sqrt(kernel)).T
    X_p = ((X - m_X).T * np.sqrt(kernel)).T

    #model fit
    if regularization == 'L1':
        mod = Lasso(alpha = 0.01/(2.*t_), fit_intercept = True)
    if regularization == 'L2':
        mod = Ridge(alpha = 0.01/(2.*t_), fit_intercept = True)
    if regularization == 'net':
        mod = ElasticNet(alpha = 0.01/(2.*t_), fit_intercept = True)
    else:
        mod = LinearRegression()

    mod.fit(X_p, Y_p)

    params = {'alpha': mod.intercept_,
              'beta': mod.coef_}
    params['residuals'] = np.subtract(Y - params['alpha'], X @ np.atleast_2d(params['beta'].T))
    return params