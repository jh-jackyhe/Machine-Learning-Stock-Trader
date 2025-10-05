import datetime as dt
import os
import numpy as np
import pandas as pd
from util import get_data, plot_data
import matplotlib.pyplot as plt


def author():
    return 'zhe343'


def testPolicy(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
    dates = pd.date_range(sd, ed)
    prices = get_data([symbol], dates, colname='Adj Close')
    trades = prices.replace(prices, 0)
    # trades['Symbol'] = symbol
    trades['Shares'] = 0
    trades.drop(['SPY', symbol], axis=1, inplace=True)
    max_holding = 1000
    holding = 0
    for i in range(1, len(prices)):
        if prices.ix[i - 1, symbol] <= prices.ix[i, symbol]:
            trades.ix[i - 1, 'Shares'] = max_holding - holding
            holding += trades.ix[i - 1, 'Shares']
        elif prices.ix[i - 1, symbol] > prices.ix[i, symbol]:
            trades.ix[i - 1, 'Shares'] = (max_holding + holding) * -1
            holding += trades.ix[i - 1, 'Shares']
    return trades


def benchmark(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
    dates = pd.date_range(sd, ed)
    prices = get_data([symbol], dates, colname='Adj Close')
    trades = prices.replace(prices, 0)
    # trades['Symbol'] = symbol
    trades['Shares'] = 0
    trades.drop(['SPY', symbol], axis=1, inplace=True)
    trades.ix[0, 'Shares'] = 1000
    return trades


# if __name__ == "__main__":
#     df_trades = testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
#     print(df_trades)
