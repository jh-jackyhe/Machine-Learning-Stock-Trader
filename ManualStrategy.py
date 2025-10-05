"""
ManualStrategy.py
Code implementing a ManualStrategy object (your Manual Strategy) in the strategy_evaluation/ directory. 
It should implement testPolicy() which returns a trades data frame (see below). 
The main part of this code should call marketsimcode as necessary to generate the plots used in the report.
The in-sample period is January 1, 2008 to December 31, 2009. 
The out-of-sample/testing period is January 1, 2010 to December 31, 2011.  
"""

import datetime as dt
import os
import numpy as np
import pandas as pd
from util import get_data, plot_data
import matplotlib.pyplot as plt
import marketsimcode as msim
import indicators as ind


def author():
    return 'zhe343'  # replace tb34 with your Georgia Tech username.


def testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
    bbp = ind.bbp(symbol, sd, ed, lookback=14)
    ema_ratio = ind.ema_ratio(symbol, sd, ed, lookback=14)
    macd = ind.macd(symbol, sd, ed, lookback=14)
    trades = bbp.replace(bbp, 0)
    trades['Shares'] = 0
    trades.drop([symbol], axis=1, inplace=True)
    indicators = trades.replace(trades, 0)
    max_holding = 1000
    holding = 0

    # 29.69% return
    for i in range(len(trades)):
        if bbp.ix[i, symbol] <= 0.3:
            indicators.ix[i, 'Shares'] += 1
        elif bbp.ix[i, symbol] >= 0.7:
            indicators.ix[i, 'Shares'] -= 1

        if ema_ratio.ix[i, symbol] <= 0.8:
            indicators.ix[i, 'Shares'] += 1
        elif ema_ratio.ix[i, symbol] >= 1.2:
            indicators.ix[i, 'Shares'] -= 1

        if macd.ix[i, symbol] > 0:
            indicators.ix[i, 'Shares'] += 1
        elif macd.ix[i, symbol] < 0:
            indicators.ix[i, 'Shares'] -= 1

    for i in range(len(indicators)):
        if indicators.ix[i, 'Shares'] >= 2:
            trades.ix[i, 'Shares'] = max_holding - holding
            holding += trades.ix[i, 'Shares']
        elif indicators.ix[i, 'Shares'] <= -2:
            trades.ix[i, 'Shares'] = (max_holding + holding) * -1
            holding += trades.ix[i, 'Shares']

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


if __name__ == "__main__":
    df_trades = testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    print(df_trades)
    # msim.compute_portvals(df_trades, symbol='JPM', start_val=100000, commission=9.95, impact=0.005)

