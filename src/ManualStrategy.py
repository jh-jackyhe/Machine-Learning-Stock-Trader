"""
ManualStrategy.py
Code implementing a ManualStrategy object (your Manual Strategy) in the strategy_evaluation/ directory. 
It should implement testPolicy() which returns a trades data frame (see below). 
The main part of this code should call marketsimcode as necessary to generate the plots.
The in-sample period is January 1, 2008 to December 31, 2009. 
The out-of-sample/testing period is January 1, 2010 to December 31, 2011.  
"""

import datetime as dt
import pandas as pd
import indicators as ind
from util import get_data


def testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
    bbp = ind.bbp(symbol, sd, ed, lookback=14)
    ema_ratio = ind.ema_ratio(symbol, sd, ed, lookback=14)
    macd = ind.macd(symbol, sd, ed, lookback=14)
    trades = bbp * 0  # Create a DataFrame with same structure but all zeros
    trades['Shares'] = 0
    trades.drop([symbol], axis=1, inplace=True)
    indicators = trades * 0  # Create a DataFrame with same structure but all zeros
    max_holding = 1000
    holding = 0

    # 29.69% return
    for i in range(len(trades)):
        if bbp.iloc[i][symbol] <= 0.3:
            indicators.iloc[i, indicators.columns.get_loc('Shares')] += 1
        elif bbp.iloc[i][symbol] >= 0.7:
            indicators.iloc[i, indicators.columns.get_loc('Shares')] -= 1

        if ema_ratio.iloc[i][symbol] <= 0.8:
            indicators.iloc[i, indicators.columns.get_loc('Shares')] += 1
        elif ema_ratio.iloc[i][symbol] >= 1.2:
            indicators.iloc[i, indicators.columns.get_loc('Shares')] -= 1

        if macd.iloc[i][symbol] > 0:
            indicators.iloc[i, indicators.columns.get_loc('Shares')] += 1
        elif macd.iloc[i][symbol] < 0:
            indicators.iloc[i, indicators.columns.get_loc('Shares')] -= 1

    for i in range(len(indicators)):
        if indicators.iloc[i]['Shares'] >= 2:
            trades.iloc[i, trades.columns.get_loc('Shares')] = max_holding - holding
            holding += trades.iloc[i]['Shares']
        elif indicators.iloc[i]['Shares'] <= -2:
            trades.iloc[i, trades.columns.get_loc('Shares')] = (max_holding + holding) * -1
            holding += trades.iloc[i]['Shares']

    return trades


def benchmark(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
    dates = pd.date_range(sd, ed)
    prices = get_data([symbol], dates, colname='Adj Close')
    trades = prices * 0  # Create a DataFrame with same structure but all zeros
    trades['Shares'] = 0
    trades.drop(['SPY', symbol], axis=1, inplace=True)
    trades.iloc[0, trades.columns.get_loc('Shares')] = 1000
    return trades
