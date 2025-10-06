import pandas as pd
from util import get_data


def compute_portvals(orders, symbol='JPM', start_val=100000, commission=0, impact=0):
    # 1. use for loop to creat a df of prices: trade symbols and cash, first and last trade date. Make a copy
    start_date = orders.index[0]
    end_date = orders.index[-1]
    dates = pd.date_range(start_date, end_date)
    prices = get_data([symbol], dates, colname='Adj Close')
    prices.drop(['SPY'], axis=1, inplace=True)
    prices['CASH'] = 1.0

    # 2. creat a df of trades: quantity...
    trades = prices * 0
    for index, row in orders.iterrows():
        if row['Shares'] > 0:
            trades.loc[index, symbol] += row['Shares']
            trades.loc[index, 'CASH'] -= row['Shares'] * prices.loc[index, symbol] * (1 + impact) + commission
        elif row['Shares'] < 0:
            trades.loc[index, symbol] -= row['Shares'] * -1
            trades.loc[index, 'CASH'] += row['Shares'] * -1 * prices.loc[index, symbol] * (1 - impact) - commission

    # 3. df of Holdings
    holdings = prices * 0
    holdings.iloc[0] = trades.iloc[0]
    holdings.loc[holdings.index[0], 'CASH'] += start_val
    for i in range(1, len(holdings)):
        holdings.iloc[i] = holdings.iloc[i - 1] + trades.iloc[i]

    # 4. df of values = prices * holdings
    values = prices * holdings
    total = values.sum(axis=1)
    return total
