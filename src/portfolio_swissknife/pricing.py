import numpy as np
from collections import defaultdict
import scipy.stats as st

def BS_pricer(portfolio, tenor_periods = 24, K_std_range = 2):
    #full chain
    option_chain = defaultdict(dict)

    #separate
    option_chain_call = defaultdict(dict)
    option_chain_put = defaultdict(dict)
    delta_call = defaultdict(dict)
    delta_put = defaultdict(dict)
    gamma = defaultdict(dict)
    vega = defaultdict(dict)
    K_est_dict = defaultdict(dict)

    #parameters - fixed
    assert portfolio.discount is not None
    sigma = np.std(portfolio.returns[:, -portfolio.estimation_period:],0)*np.sqrt(252)
    s_0 = portfolio.prices.iloc[-1,:].values
    #range for priced strikes 2std apart
    K_range = (np.round(s_0 - np.multiply(K_std_range, (sigma * s_0))),
               np.round(s_0 + np.multiply(K_std_range, (sigma * s_0))))

    for idx, stock in enumerate(portfolio.securities):
        K_est = list(np.arange(K_range[0][idx], K_range[1][idx]))
        K_est_dict[stock] = K_est

        option_chain_call[stock] = np.zeros([tenor_periods+1, len(K_est)])
        option_chain_put[stock] = np.zeros([tenor_periods+1, len(K_est)])
        delta_call[stock] = np.zeros([tenor_periods+1, len(K_est)])
        delta_put[stock] = np.zeros([tenor_periods+1, len(K_est)])
        gamma[stock] = np.zeros([tenor_periods+1, len(K_est)])
        vega[stock] = np.zeros([tenor_periods+1, len(K_est)])

        for tenor in range(1,tenor_periods+1):

            pricer = lambda op_type, K: BS_Option_Price(op_type, s_0[idx], K, sigma[idx],
                                                        0, tenor * 22, portfolio.discount[-1]).flatten()
            delta = lambda op_type, K: BS_Delta(op_type, s_0[idx], K, sigma[idx],
                                                0, tenor * 22, portfolio.discount[-1]).flatten()

            # price options
            option_chain_call[stock][tenor, :] = pricer('call', K_est)
            option_chain_put[stock][tenor, :] = pricer('put', K_est)
            delta_call[stock][tenor, :] = delta('call', K_est)
            delta_put[stock][tenor, :] = delta('put', K_est)
            gamma[stock][tenor, :] = BS_Gamma(s_0[idx], K_est, sigma[idx], 0, tenor * 22, portfolio.discount[-1]).flatten()
            vega[stock][tenor, :] = BS_Vega(s_0[idx], K_est, sigma[idx], 0, tenor * 22, portfolio.discount[-1]).flatten()

    #update option chain
    for stock in portfolio.securities:
        option_chain[stock]['call'] = option_chain_call[stock].T
        option_chain[stock]['put'] = option_chain_put[stock].T
        option_chain[stock]['delta_call'] = delta_call[stock].T
        option_chain[stock]['delta_put'] = delta_put[stock].T
        option_chain[stock]['gamma'] = gamma[stock].T
        option_chain[stock]['vega'] = vega[stock].T
        option_chain[stock]['K_range'] = K_est_dict[stock]

    return option_chain

def BS_Option_Price(CP,S_0,K,sigma,t,T,r):
    K = np.array(K).reshape([len(K),1])
    d1 = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma,2.0))
          * (T-t)) / (sigma * np.sqrt(T-t))
    d2 = d1 - sigma * np.sqrt(T-t)
    if CP == 'call':
        value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-r * (T-t))

    elif CP == 'put':
        value = st.norm.cdf(-d2) * K * np.exp(-r * (T-t)) - st.norm.cdf(-d1)*S_0

    return value

def BS_Delta(CP,S_0,K,sigma,t,T,r):
    K = np.array(K).reshape([len(K),1])
    d1 = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma,2.0)) * \
             (T-t)) / (sigma * np.sqrt(T-t))
    if CP == 'call':
        value = st.norm.cdf(d1)
    elif CP == 'put':
        value = st.norm.cdf(d1)-1
    return value

def BS_Gamma(S_0,K,sigma,t,T,r):
    K = np.array(K).reshape([len(K),1])
    d1 = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma,2.0)) * \
             (T-t)) / (sigma * np.sqrt(T-t))
    return st.norm.pdf(d1) / (S_0 * sigma * np.sqrt(T-t))

def BS_Vega(S_0,K,sigma,t,T,r):
    d1 = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma,2.0)) * \
             (T-t)) / (sigma * np.sqrt(T-t))
    return S_0*st.norm.pdf(d1)*np.sqrt(T-t)