from portfolio import Portfolio
import pandas as pd
import matplotlib.pyplot as plt

def main():
    securities = ['SPY', 'BND', 'GLD', 'NDX']
    test = Portfolio(securities)
    test.set_period(('2015-01-01', '2021-05-28'))
    test.get_prices('daily')

    (1+test._get_state(0,50)).cumprod().plot()
    plt.show()
if __name__ == '__main__':
    main()

