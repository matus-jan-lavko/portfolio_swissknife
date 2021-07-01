import numpy as np

def annualized_average_return(r, num_periods):
    return np.sum(r) * (num_periods/r.shape[0])

def annualized_cagr(r, num_periods):
    return (np.prod((1+r))) ** (num_periods/r.shape[0]) - 1

def annualized_stdev(r, num_periods):
    return np.sqrt(num_periods * (np.sum(r - np.mean(r) ** 2) / (r.shape[0] - 1)))

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
    return term1*np.sum(num/denom)

def cokurtosis(r, r_b):
    t = r.shape[0]
    term1 = t / ((t - 1) * (t - 2) * (t - 3))
    num = ((r - np.mean(r) * ((r_b - np.mean(r_b) ** 3))))
    denom = np.std(r, ddof=1) * (np.std(r_b, ddof=1) ** 3)
    term3 = (3 * (t - 1) ** 2) / ((t - 2) * (t - 3))
    return term1*np.sum((num/denom)) - term3

def max_drawdown(r):
    index = 1000 * np.cumprod(1 + r)
    hwm = np.maximum.accumulate(index)
    dd = (index - hwm) / hwm
    max_dd = np.min(dd)
    return -max_dd

def max_drawdown_duration(r):
    index = 1000 * np.cumprod(1 + r)
    hwm = np.maximum.accumulate(index)
    dd = (index - hwm) / hwm
    end = np.argmin(dd)
    start = np.argmax(index[:end])
    return (end-start)

def pain_ratio(r):
    raise NotImplementedError




