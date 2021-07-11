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
    ema = np.mean((1 + _ewma_vectorized_2d(r, window = window, axis = 1)) ,axis=0)
    annr = ema ** 252 - 1
    return annr


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
    rho = np.corrcoef(r, rowvar= False)
    n_ = rho.shape[0]

    rho_bar = (rho.sum() - n_) / (n_ * (n_ - 1))
    ccor = np.full_like(rho, rho_bar)
    np.fill_diagonal(ccor, 1.)
    sig = r.std()
    ccov = ccor * np.outer(sig, sig)
    return ccov


def shrinkage_cov(r, delta=0.5, prior_model=elton_gruber_cov, *args, **kwargs):
    '''
    Shrinks the sample covariance towards a specified model such as Elton/Gruber

    r: n x 1 returns
    delta: shrinkage parameter [0,1] #0.5 is standard
    prior_model: prior_model function that we shrink towards

    out: n x n shrunk covariance matrix
    '''

    prior = prior_model(r, *args, **kwargs)
    sig_hat = sample_cov(r)

    # https://jpm.pm-research.com/content/30/4/110
    honey = delta * prior + (1 - delta) * sig_hat
    return honey

def linear_factor_model(Y, X, kernel = None, regularize = None):
    t_, n_ = Y.shape

    #setting weights
    if kernel is None:
        kernel = np.ones(t_) / t_

    m_Y = kernel @ Y
    m_X = kernel @ X

    Y_p = ((Y - m_Y).T * np.sqrt(kernel)).T
    X_p = ((X - m_X).T * np.sqrt(kernel)).T

    #model fit
    if regularize == 'L1':
        mod = Lasso(alpha = 0.01/(2.*t_), fit_intercept = True)
    if regularize == 'L2':
        mod = Ridge(alpha = 0.01/(2.*t_), fit_intercept = True)
    if regularize == 'net':
        mod = ElasticNet(alpha = 0.01/(2.*t_), fit_intercept = True)
    else:
        mod = LinearRegression()

    mod.fit(X_p, Y_p)

    alpha = mod.intercept_
    beta = mod.coef_
    residuals = np.subtract(Y - alpha, X @ np.atleast_2d(beta.T))

    return alpha, beta, residuals

def _ewma_vectorized(data, alpha, offset=None, dtype=None, order='C', out=None):
    """
    Calculates the exponential moving average over a vector.
    Will fail for large inputs.
    :param data: Input data
    :param alpha: scalar float in range (0,1)
        The alpha parameter for the moving average.
    :param offset: optional
        The offset for the moving average, scalar. Defaults to data[0].
    :param dtype: optional
        Data type used for calculations. Defaults to float64 unless
        data.dtype is float32, then it will use float32.
    :param order: {'C', 'F', 'A'}, optional
        Order to use when flattening the data. Defaults to 'C'.
    :param out: ndarray, or None, optional
        A location into which the result is stored. If provided, it must have
        the same shape as the input. If not provided or `None`,
        a freshly-allocated array is returned.
    """
    data = np.array(data, copy=False)

    if dtype is None:
        if data.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float64
    else:
        dtype = np.dtype(dtype)

    if data.ndim > 1:
        # flatten input
        data = data.reshape(-1, order)

    if out is None:
        out = np.empty_like(data, dtype=dtype)
    else:
        assert out.shape == data.shape
        assert out.dtype == dtype

    if data.size < 1:
        # empty input, return empty array
        return out

    if offset is None:
        offset = data[0]

    alpha = np.array(alpha, copy=False).astype(dtype, copy=False)

    # scaling_factors -> 0 as len(data) gets large
    # this leads to divide-by-zeros below
    scaling_factors = np.power(1. - alpha, np.arange(data.size + 1, dtype=dtype),
                               dtype=dtype)
    # create cumulative sum array
    np.multiply(data, (alpha * scaling_factors[-2]) / scaling_factors[:-1],
                dtype=dtype, out=out)
    np.cumsum(out, dtype=dtype, out=out)

    # cumsums / scaling
    out /= scaling_factors[-2::-1]

    if offset != 0:
        offset = np.array(offset, copy=False).astype(dtype, copy=False)
        # add offsets
        out += offset * scaling_factors[1:]

    return out

def _ewma_vectorized_2d(data, window, axis=None, offset=None, dtype=None, order='C', out=None):
    """
    Calculates the exponential moving average over a given axis.
    Credit to Jake Walden: https://stackoverflow.com/questions/42869495/
    :param data: Input data, must be 1D or 2D array.
    :param alpha: scalar float in range (0,1)
        The alpha parameter for the moving average.
    :param axis: The axis to apply the moving average on.
        If axis==None, the data is flattened.
    :param offset: optional
        The offset for the moving average. Must be scalar or a
        vector with one element for each row of data. If set to None,
        defaults to the first value of each row.
    :param dtype: optional
        Data type used for calculations. Defaults to float64 unless
        data.dtype is float32, then it will use float32.
    :param order: {'C', 'F', 'A'}, optional
        Order to use when flattening the data. Ignored if axis is not None.
    :param out: ndarray, or None, optional
        A location into which the result is stored. If provided, it must have
        the same shape as the desired output. If not provided or `None`,
        a freshly-allocated array is returned.
    """
    data = np.array(data, copy=False)

    assert data.ndim <= 2

    #calculate the window
    sum_proportion = 0.99 #proportion of the window to be summed upon
    alpha = 1 - np.exp(np.log(1 - sum_proportion) / window)

    if dtype is None:
        if data.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float64
    else:
        dtype = np.dtype(dtype)

    if out is None:
        out = np.empty_like(data, dtype=dtype)
    else:
        assert out.shape == data.shape
        assert out.dtype == dtype

    if data.size < 1:
        # empty input, return empty array
        return out

    if axis is None or data.ndim < 2:
        # use 1D version
        if isinstance(offset, np.ndarray):
            offset = offset[0]
        return _ewma_vectorized(data, alpha, offset, dtype=dtype, order=order,
                               out=out)

    assert -data.ndim <= axis < data.ndim

    # create reshaped data views
    out_view = out
    if axis < 0:
        axis = data.ndim - int(axis)

    if axis == 0:
        # transpose data views so columns are treated as rows
        data = data.T
        out_view = out_view.T

    if offset is None:
        # use the first element of each row as the offset
        offset = np.copy(data[:, 0])
    elif np.size(offset) == 1:
        offset = np.reshape(offset, (1,))

    alpha = np.array(alpha, copy=False).astype(dtype, copy=False)

    # calculate the moving average
    row_size = data.shape[1]
    row_n = data.shape[0]
    scaling_factors = np.power(1. - alpha, np.arange(row_size + 1, dtype=dtype),
                               dtype=dtype)
    # create a scaled cumulative sum array
    np.multiply(
        data,
        np.multiply(alpha * scaling_factors[-2], np.ones((row_n, 1), dtype=dtype),
                    dtype=dtype)
        / scaling_factors[np.newaxis, :-1],
        dtype=dtype, out=out_view
    )
    np.cumsum(out_view, axis=1, dtype=dtype, out=out_view)
    out_view /= scaling_factors[np.newaxis, -2::-1]

    if not (np.size(offset) == 1 and offset == 0):
        offset = offset.astype(dtype, copy=False)
        # add the offsets to the scaled cumulative sums
        out_view += offset[:, np.newaxis] * scaling_factors[np.newaxis, 1:]

    return out