import matplotlib.pyplot as plt
import os
import pandas as pd
get_ipython().run_line_magic("load_ext", " autoreload")
get_ipython().run_line_magic("autoreload", " 2")
plt.style.use('bmh')


from portfolio_swissknife import portfolio as ps
from portfolio_swissknife import risk_model as rm


sectors = ['XLK', 'XLY', 'XLB', 'XLC', 'XLE', 'XLU', 'XLF', 'XLV', 'XOP']
random_stocks = ['JCI', 'TGT', 'CMCSA', 'CPB', 'MO', 'APA', 'MMC', 'JPM',
                 'ZION', 'PSA', 'BAX', 'BMY', 'LUV', 'PCAR', 'TXT', 'TMO',]

pf1 = ps.Portfolio(random_stocks)
pf1.set_period(('2012-01-01', '2021-05-28'))
pf1.get_prices('daily')


pf1.set_benchmark('SPY')
pf1.set_discount('^TNX')


#TODO FIX SHRINKAGE
# pf1.set_estimation_method(est.ema_return_historic, moment = 1)
# pf1.set_estimation_method(est.elton_gruber_cov, moment = 2)


pf1.set_constraints(default=True) #defaults to long_only fully invested portfolio with no leverage
pf1.historical_backtest(models=['EW','RP', 'GMV', 'MDD', 'MDR'], frequency=22, estimation_period = int(252*1.5))


#runtimes
pd.DataFrame(pf1.backtest).loc['opt_time'].plot(kind='bar', title='Optimization runtime')


pf1.backtest['EW']['returns'].shape


pf1.get_backtest_report(num_rows=2)


factors = ['SPY', 'VLUE', 'SIZE', 'QUAL', 'MTUM', 'USMV']
rm1 = rm.RiskModel(pf1, factors)
rm1.get_prices('daily')


rm1.rolling_factor_exposure(method='linear')


rm1.get_risk_report(model = 'MDD')



