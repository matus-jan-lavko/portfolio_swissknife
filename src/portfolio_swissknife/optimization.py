import numpy as np
import cvxpy as cp
from scipy.optimize import minimize

def global_minimum_variance(sigma, constraints: dict, n_):
    '''
    Calculates the GMV portfolio.

    :param sigma: estimates of the cov matrix: np.array
    :param constraints: dictionary of constraints: dict
    :param n_: number of assets: int
    :return: optimal weights: np.array
    '''
    w = cp.Variable(n_)
    risk = cp.quad_form(w, sigma)

    constraint_list = _build_constraints_cvxpy(w, constraints)
    objective = cp.Minimize(risk * 1000)
    problem = cp.Problem(objective, constraint_list)
    problem.solve()

    w_opt = w.value
    return w_opt


def risk_parity(sigma , constraints: dict, n_):
    '''
    Calculates the Equal Risk Parity portfolio.

    :param sigma: estimates of the cov matrix: np.array
    :param constraints: dictionary of constraints: dict
    :param n_: number of assets: int
    :return: optimal weights: np.array
    '''

    def risk_contribution(w, sigma):
        '''
        Calculates asset contribution to total risk

        :param w: weights: np.array
        :param sigma: estimated cov matrix: np.array
        :return: risk contribution: np.array
        '''

        w = np.matrix(w)
        sigma = np.sqrt(_portfolio_variance(w, sigma))
        # Marginal Risk Contribution
        mrc = sigma * w.T
        # Total asset risk contribution
        rc = np.multiply(mrc, w.T) / sigma
        return rc

    def risk_objective(x, params):
        '''
        Calculates portfolio risk and provides an objective for the scipy.minimize function

        :param x: dummy portfolio weights: np.array
        :param params: parameters estimated within the outer function: list
        :return: objective function: np.array
        '''
        # calculate portfolio risk
        covar = params[0]  # covariance table
        b = params[1]  # risk target in percent of portfolio risk
        sig_p = np.sqrt(_portfolio_variance(x, covar)) # portfolio sigma
        risk_target = np.asmatrix(np.multiply(sig_p, b))
        asset_rc = risk_contribution(x, covar)
        J = sum(np.square(asset_rc - risk_target.T))[0, 0]  # sum of squared error
        return J

    b = np.ones((1, n_))
    b = b / n_  # your risk budget percent of total portfolio risk (equal risk)

    constraint_tup = _build_constraints_scipy(constraints)
    problem = minimize(risk_objective, b, args=[sigma, b], method='SLSQP',
                       constraints=constraint_tup, options={'disp': False})

    w_opt = np.array(problem.x)
    return w_opt

def max_diversification_ratio(sigma, w_prev, constraints: dict):
    '''
    Calculates the Maximum Diversification Ratio portfolio

    :param sigma: cov matrix estimates: np.array
    :param w_prev: cached weights from previous window: np.array
    :param constraints: optimizing constraints: dict
    :return: optimal weights: np.array
    '''

    def diversification_ratio(w, sigma):
        '''
        Defines and calculates the diversfication ratio

        :param w: weight dummy: np.array
        :param sigma: estimates of cov matrix: np.array
        :return: diversfication ratio: np.array
        '''
        #average weighted volatility
        w_vol = np.dot(np.sqrt(np.diag(sigma)), w.T)
        #port vol
        port_vol = np.sqrt(_portfolio_variance(w, sigma))
        div_ratio = w_vol/port_vol
        return (-div_ratio)

    constraint_tup = _build_constraints_scipy(constraints)
    problem = minimize(diversification_ratio, w_prev, args = sigma, method = 'SLSQP',
                       constraints = constraint_tup, options={'disp': False})

    w_opt = np.array(problem.x)
    return w_opt

def greedy_optimization(efficient_frontier: dict, r_est, maximum, function, function_kwargs):
    '''
    Greedy optimization of a portfolio based on a efficient frontier

    :param efficient_frontier: mean variance frontier of optimized portfolios: np.array
    :param r_est: series of returns to estimate the objective on: np.array
    :param maximum: flag for maximum: bool
    :param function: function that calculates the objective: function
    :param function_kwargs: arg
    :return: optimal weights: np.array
    '''
    grid_vals = np.zeros(len(efficient_frontier['portfolios']))
    for idx, solu in enumerate(efficient_frontier['portfolios']):
        r_p = np.dot(solu, r_est.T)
        if function_kwargs:
            grid_vals[idx] = function(r_p, **function_kwargs)
        else:
            grid_vals[idx] = function(r_p)

    if maximum:
        opt_id = np.argmax(grid_vals)
    else:
        opt_id = np.argmin(grid_vals)

    w_opt = efficient_frontier['portfolios'][opt_id]
    gamma = efficient_frontier['gamma'][opt_id]
    return w_opt, gamma

def hierarchical_risk_parity():
    #todo implement
    raise NotImplementedError

def quadratic_risk_utility(mu, sigma, constraints: dict, n_, grid_size = 100):
    '''
    Calculates the mean-variance efficient frontier based on the estimates of the first and second moments

    :param mu: first moment estimates: np.array
    :param sigma: second moment estimates: np.array
    :param constraints: constraints dictionary: np.array
    :param n_: number of assets: int
    :param grid_size: number of efficient portfolios on the grid: int
    :return: efficient frontier: list
    '''
    w = cp.Variable(n_)
    gamma = cp.Parameter(nonneg=True)
    port_ret = mu.T @ w
    risk = cp.quad_form(w, sigma)
    constraint_list = _build_constraints_cvxpy(w, constraints)

    objective = cp.Minimize(gamma*risk - port_ret)
    problem = cp.Problem(objective, constraint_list)

    #solve the set of efficient portfolios
    gamma_grid = np.linspace(0,10, grid_size)

    #todo rewrite so that you can back out the risk-aversion parameter

    efficient_frontier = {'portfolios':[],
                          'gamma': []}
    for i in range(grid_size):
        gamma.value = gamma_grid[i]
        problem.solve()
        efficient_frontier['portfolios'].append(w.value)
        efficient_frontier['gamma'].append(gamma_grid[i])

    return efficient_frontier

def _portfolio_variance(w, sigma):
    '''
    Helper function for calculating portfolio variance

    :param w: weights of the portfolio: np.array
    :param sigma: estimated cov matrix: np.array
    :return: float
    '''
    w = np.matrix(w)
    sig_p = w * sigma * w.T
    return sig_p[0,0]

def _build_constraints_cvxpy(w_obj, constraints: dict):
    '''
    Functions that builds constraints for an optimization program written on the cvxpy package.

    :param w_obj: weights object: cvxpy.variable
    :param constraints: dictionary of constraints mapped to the cvxpy constraints
    :return: list of cvxpy constraints to be input to the optimizer: list
    '''
    constraint_list = []
    for k, v in constraints.items():
        if k == 'long_only' and v == True:
            constraint_list.append(w_obj * 1000 >= 0)
        if k == 'leverage':
            constraint_list.append(w_obj * 1000 <= v * 1000)
        if k == 'normalizing' and v == True:
            constraint_list.append(cp.sum(w_obj) * 1000 == 1 * 1000)
    return constraint_list

def _build_constraints_scipy(constraints: dict):
    '''
    Functions that builds constraints for an optimization program written on the scipy package.

    :param constraints: dictionary of constraints mapped to the scipy constraints
    :return: scipy constraints to be input to the optimizer: tuple
    '''

    for k, v in constraints.items():
        if k == 'long_only' and v == True:
            long_only = {'type': 'ineq', 'fun' : lambda x: x}
        if k == 'normalizing' and v == True:
            normalizing = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
    constraint_tup = (long_only, normalizing)
    return constraint_tup