import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
get_ipython().run_line_magic("load_ext", " autoreload")
get_ipython().run_line_magic("autoreload", " 2")
plt.style.use('bmh')


#dev on the fly
os.chdir('C:\\Users\matus\pysource\portfolio_swissknife')


# get_ipython().getoutput("pip install .")


from portfolio_swissknife import portfolio as ps
from portfolio_swissknife import risk_model as rm


#loading ext. data -- close prices of SPX
universe = pd.read_csv('ext_data/00_db_SPX__PX_LAST.csv', index_col = 0, parse_dates = True)
universe = universe[::-1].loc[:,universe.notna().all(axis=0)]
securities = [universe.columns[i].split(' ')[0] for i, _ in enumerate(universe.columns)]
universe = universe.loc['2014':]


port_universe = ps.Portfolio(securities)
port_universe.set_custom_prices(universe, 'daily')


factors = ['SPY', 'VLUE', 'SIZE', 'QUAL', 'MTUM', 'USMV']
rm_universe = rm.RiskModel(port_universe, factors)
rm_universe.get_prices('daily')


rm_universe.rolling_factor_selection(10, 'linear', int(2*252), 22)


factor_pf_vlue = ps.FactorPortfolio(port_universe, rm_universe, 'VLUE')
factor_pf_vlue.set_constraints(default = True)
factor_pf_vlue.set_benchmark('SPY')
factor_pf_vlue.set_discount('^TNX')


factor_pf_vlue.historical_backtest(models=['EW','GMV','RP','MES'],frequency=22, estimation_period=int(2*252))


factor_pf_vlue.get_backtest_report(display_weights = False, title = 'VLUE long only top decile')


factor_pf_mtum = ps.FactorPortfolio(port_universe, rm_universe, 'MTUM')
factor_pf_mtum.set_constraints(default = True)
factor_pf_mtum.set_benchmark('SPY')
factor_pf_mtum.set_discount('^TNX')


factor_pf_mtum.historical_backtest(models=['EW','GMV','RP','MES'],frequency=22, estimation_period=int(2*252))


factor_pf_mtum.get_backtest_report(display_weights = False, title = 'MTUM long only top decile')
