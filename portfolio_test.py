from portfolio import portfolio
import pandas as pd
import matplotlib.pyplot as plt

def main():
    securities = ['SPY', 'BND', 'GLD', 'NDX']
    test = portfolio(securities)
    test.set_period(('2015-01-01', '2021-05-28'))
    test.get_prices('daily')
    test.prices.plot()
    plt.show()

if __name__ == '__main__':
    main()

