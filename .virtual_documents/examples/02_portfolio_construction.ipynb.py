import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
get_ipython().run_line_magic("load_ext", " autoreload")
get_ipython().run_line_magic("autoreload", " 2")
plt.style.use('bmh')


from portfolio_swissknife import portfolio as ps
from portfolio_swissknife import risk_model as rm


#loading ext. data -- close prices of SPX
universe = pd.read_csv('../ext_data/00_db_SPX__PX_LAST.csv', index_col = 0, parse_dates = True)
universe = universe[::-1].loc[:,universe.notna().all(axis=0)]
securities = [universe.columns[i].split(' ')[0] for i, _ in enumerate(universe.columns)]
universe = universe.loc['2014':]


port_universe = ps.Portfolio(securities)
port_universe.set_custom_prices(universe, 'daily')


factors = ['SPY', 'VLUE', 'SIZE', 'QUAL', 'MTUM', 'USMV']
rm_universe = rm.RiskModel(port_universe, factors)
rm_universe.get_prices('daily')


a, b, c = rm_universe._estimate_panel(method='linear')


b.shape


sort = np.argsort(b[:,0])


plt.bar(height=b[:,0][sort],x=np.array(rm_universe.portfolio.securities)[sort])


np.array(rm_universe.portfolio.securities)[sort]



