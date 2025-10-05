import datetime as dt
import os
import numpy as np
import pandas as pd
from util import get_data, plot_data
import matplotlib.pyplot as plt
# import TheoreticallyOptimalStrategy as tos


def author():
    return 'zhe343'


def compute_portvals(orders, symbol='JPM', start_val=100000, commission=0, impact=0):
    # 1. use for loop to creat a df of prices: trade symbols and cash, first and last trade date. Make a copy
    start_date = orders.index[0]
    end_date = orders.index[-1]
    dates = pd.date_range(start_date, end_date)
    prices = get_data([symbol], dates, colname='Adj Close')
    prices.drop(['SPY'], axis=1, inplace=True)
    prices['CASH'] = 1.0

    # 2. creat a df of trades: quantity...
    trades = prices.replace(prices, 0)
    for index, row in orders.iterrows():
        if row['Shares'] > 0:
            trades.ix[index, symbol] += row['Shares']
            trades.ix[index, 'CASH'] -= row['Shares'] * prices.ix[index, symbol] * (1 + impact) + commission
        elif row['Shares'] < 0:
            trades.ix[index, symbol] -= row['Shares'] * -1
            trades.ix[index, 'CASH'] += row['Shares'] * -1 * prices.ix[index, symbol] * (1 - impact) - commission

    # 3. df of Holdings
    holdings = prices.replace(prices, 0)
    holdings.ix[0] = trades.ix[0]
    holdings.ix[0, 'CASH'] += start_val
    for i in range(1, len(holdings)):
        holdings.ix[i] = holdings.ix[i - 1] + trades.ix[i]

    # 4. df of values = prices * holdings
    values = prices * holdings
    total = values.sum(axis=1)
    return total

# if __name__ == "__main__":
#     pd.set_option('display.max_rows', None)
#     df_trades = tos.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
#     portvals = compute_portvals(df_trades, symbol='JPM', start_val=100000, commission=0, impact=0)
#     print(portvals)
