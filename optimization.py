import numpy as np
import cvxpy as cp

def global_minimum_variance(sigma, constraints: dict , n_):
    w = cp.Variable(n_)
    risk = cp.quad_form(w, sigma)

    constraint_list = _build_constraints(w,constraints)
    problem = cp.Problem(cp.Minimize(risk), constraint_list)
    problem.solve()
    w_opt = w.value

    return w_opt

def risk_parity(mu, sigma, constraints: dict, n_):
    raise NotImplementedError


def _build_constraints(w_obj, constraints: dict):
    constraint_list = []
    for k,v in constraints.items():
        if k == 'long_only' and v == True:
            constraint_list.append(w_obj >= 0)
        if k == 'leverage':
            constraint_list.append(w_obj <= v)
        if k == 'normalizing' and v == True:
            constraint_list.append(cp.sum(w_obj) == 1)
    return constraint_list