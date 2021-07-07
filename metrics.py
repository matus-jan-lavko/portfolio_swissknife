import numpy as np
import pandas as pd
from scipy.stats import norm, t


def annualized_average_return(r, num_periods):
    return (np.sum(r) * (num_periods / r.shape[0])) * 100


def annualized_cagr(r, num_periods):
    return ((np.prod((1 + r))) ** (num_periods / r.shape[0]) - 1) * 100


def annualized_stdev(r, num_periods, downside=False):
    if downside:
        semistd = np.std(r[r < 0])
        return (semistd * (num_periods ** 0.5)) * 100
    else:
        return (np.sqrt(num_periods * (np.sum(r - np.mean(r) ** 2) / (r.shape[0] - 1)))) * 100


def skewness(r):
    t = r.shape[0]
    term1 = t / ((t - 1) * (t - 2))
    term2 = np.sum(((r - np.mean(r)) / np.std(r, ddof=1)) ** 3)
    return term1 * term2


def kurtosis(r):
    t = r.shape[0]
    term1 = t / ((t - 1) * (t - 2) * (t - 3))
    term2 = np.sum(((r - np.mean(r)) / np.std(r, ddof=1)) ** 4)
    term3 = (3 * (t - 1) ** 2) / ((t - 2) * (t - 3))
    return term1 * term2 - term3


def coskewness(r, r_b):
    t = r.shape[0]
    term1 = t / ((t - 1) * (t - 2))
    num = ((r - np.mean(r)) * ((r_b - np.mean(r_b) ** 2)))
    denom = np.std(r, ddof=1) * (np.std(r_b, ddof=1) ** 2)
    return term1 * np.sum(num / denom)


def cokurtosis(r, r_b):
    t = r.shape[0]
    term1 = t / ((t - 1) * (t - 2) * (t - 3))
    num = ((r - np.mean(r) * ((r_b - np.mean(r_b) ** 3))))
    denom = np.std(r, ddof=1) * (np.std(r_b, ddof=1) ** 3)
    term3 = (3 * (t - 1) ** 2) / ((t - 2) * (t - 3))
    return term1 * np.sum((num / denom)) - term3


def max_drawdown(r):
    dd = _drawdown(r)
    max_dd = np.min(dd)
    return -(max_dd) * 100


def max_drawdown_duration(r):
    dd = _drawdown(r)
    end = np.argmin(dd)
    start = np.argmax(dd[:end])
    return (end - start)


def information_ratio(r, r_f, num_periods, ratio_type='sharpe'):
    # convert the annual yield to per period
    r_f = (1 + r_f) ** (1 / num_periods) - 1
    r_exc = r - r_f

    if ratio_type == 'sharpe':
        ir = annualized_average_return(r_exc, num_periods) / annualized_stdev(r, num_periods)
    if ratio_type == 'sortino':
        ir = annualized_average_return(r_exc, num_periods) / annualized_stdev(r, num_periods, True)
    if ratio_type == 'calmar':
        max_dd = max_drawdown(r)
        ir = -annualized_average_return(r, num_periods) / max_dd
    return ir


def var(r, alpha=0.05, exp_shortfall=False, dist='t'):
    # fit the distributions
    mu_fit_norm, sig_fit_norm = norm.fit(r)
    if dist == 'normal':
        if exp_shortfall:
            var = alpha ** -1 * norm.pdf(norm.ppf(alpha)) * sig_fit_norm - mu_fit_norm
        else:
            var = norm.ppf(1 - alpha) * sig_fit_norm - mu_fit_norm
        return var
    if dist == 't':
        nu_fit, mu_fit, sig_fit = t.fit(r)
        if exp_shortfall:
            xa_nu = t.ppf(alpha, nu_fit)
            var = -1 / alpha * (1 - nu_fit) ** -1 * (nu_fit - 2 + xa_nu ** 2) * t.pdf(xa_nu,
                                                                                      nu_fit) * sig_fit_norm - mu_fit_norm
        else:
            var = np.sqrt((nu_fit - 2) / nu_fit) * t.ppf(1 - alpha, nu_fit) * sig_fit_norm - mu_fit_norm
        return var


def reward_to_var(r, num_periods, *args, **kwargs):
    var_val = var(r, *args, **kwargs)
    reward = (annualized_average_return(r, num_periods) / (num_periods * var_val))
    return reward

def pain_ratio(r, num_periods):
    t = r.shape[0]
    dd = _drawdown(r)
    pain_index = -np.sum(dd / t)
    pain_ratio = annualized_average_return(r, num_periods) / pain_index
    return pain_ratio


def portfolio_summary(r, r_f, r_b, num_periods):
    # r = np.multiply(r, 100)
    # r_f = np.multiply(r_f, 100)
    # r_b = np.multiply(r_b, 100)
    stats = pd.DataFrame({
        'Average Returns': r.aggregate(annualized_average_return,
                                       num_periods=num_periods).map('{:,.2f}%'.format),
        'CAGR': r.aggregate(annualized_cagr, num_periods=num_periods).map('{:,.2f}%'.format),
        'Volatility': r.aggregate(annualized_stdev, num_periods=num_periods).map('{:,.2f}%'.format),
        'Max DD': r.aggregate(max_drawdown).map('{:,.2f}%'.format),
        # 'Max DD Duration': r.aggregate(max_drawdown_duration).map('{:,.0f}'.format),
        'Skewness': r.aggregate(skewness).map('{:,.2f}'.format),
        'Kurtosis': r.aggregate(kurtosis).map('{:,.2f}'.format),
        'Sharpe Ratio': r.aggregate(information_ratio, r_f=r_f, ratio_type='sharpe',
                                    num_periods=num_periods).map('{:,.2f}'.format),
        'Sortino Ratio': r.aggregate(information_ratio, r_f=r_f, ratio_type='sortino',
                                     num_periods=num_periods).map('{:,.2f}'.format),
        'Calmar Ratio': r.aggregate(information_ratio, r_f=r_f, ratio_type='calmar',
                                    num_periods=num_periods).map('{:,.2f}'.format),
        'Pain Ratio': r.aggregate(pain_ratio, num_periods=num_periods).map('{:,.2f}'.format),
        'Reward to 95% Var': r.aggregate(reward_to_var, num_periods=num_periods,
                                         alpha=0.05, dist='t').map('{:,.2f}'.format),
        'Reward to 95% cVar': r.aggregate(reward_to_var, num_periods=num_periods,
                                          alpha=0.05, exp_shortfall=True, dist='t').map('{:,.2f}'.format),
        'CoSkew with Benchmark': r.aggregate(coskewness, r_b=r_b).map('{:,.2f}'.format),
        'CoKurt within Benchmark': r.aggregate(cokurtosis, r_b=r_b).map('{:,.2f}'.format)

    })
    return stats.T


def _drawdown(r):
    index = 1000 * np.cumprod(1 + r)
    hwm = np.maximum.accumulate(index)
    dd = (index - hwm) / hwm
    return dd
