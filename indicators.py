import datetime as dt
import os
import numpy as np
import pandas as pd
from util import get_data, plot_data
import matplotlib.pyplot as plt


def author():
    return 'zhe343'


def bbp(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), lookback=14):
    dates = pd.date_range(sd, ed)
    prices = get_data([symbol], dates, colname='Adj Close')
    prices.drop(['SPY'], axis=1, inplace=True)
    # bb%
    sma = prices.rolling(window=lookback, min_periods=lookback).mean()
    rolling_std = prices.rolling(window=lookback, min_periods=lookback).std()
    top_band = sma + 2 * rolling_std
    bottom_band = sma - 2 * rolling_std
    bbp = (prices - bottom_band) / (top_band - bottom_band)
    bbp[:lookback - 1] = 0.5
    return bbp


def ema_ratio(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), lookback=14):
    dates = pd.date_range(sd, ed)
    prices = get_data([symbol], dates, colname='Adj Close')
    prices.drop(['SPY'], axis=1, inplace=True)
    # EMA
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema_ratio = prices / ema12
    return ema_ratio


def macd(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), lookback=14):
    dates = pd.date_range(sd, ed)
    prices = get_data([symbol], dates, colname='Adj Close')
    prices.drop(['SPY'], axis=1, inplace=True)
    # MACD and signal
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    diff = macd - signal
    return diff
