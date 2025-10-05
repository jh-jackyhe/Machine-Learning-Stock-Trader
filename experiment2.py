"""
Create a chart that shows:
Value of the ManualStrategy portfolio (normalized to 1.0 at the start)
Value of the StrategyLearner portfolio (normalized to 1.0 at the start)
Value of the Benchmark portfolio (normalized to 1.0 at the start)
"""

import datetime as dt
import os
import sys
import numpy as np
import pandas as pd
from util import get_data, plot_data
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import marketsimcode as msim
import ManualStrategy as ms
import StrategyLearner as sl


def author():
    return 'zhe343'


def e2():
    # ms_trades = ms.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    # benchmark_trades = ms.benchmark(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)

    sl_learner1 = sl.StrategyLearner(verbose=False, impact=0.005, commission=0)  # constructor
    sl_learner1.add_evidence(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)  # training phase
    sl_trades1 = sl_learner1.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)  # testing phase

    sl_learner2 = sl.StrategyLearner(verbose=False, impact=0.01, commission=0)  # constructor
    sl_learner2.add_evidence(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)  # training phase
    sl_trades2 = sl_learner2.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)  # testing phase

    sl_learner3 = sl.StrategyLearner(verbose=False, impact=0.02, commission=0)  # constructor
    sl_learner3.add_evidence(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)  # training phase
    sl_trades3 = sl_learner3.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)  # testing phase


    volume1 = (sl_trades1.abs()).sum()
    volume2 = (sl_trades2.abs()).sum()
    volume3 = (sl_trades3.abs()).sum()

    stdoutOrigin = sys.stdout
    sys.stdout = open("t3.txt", 'w')
    print(f"Total shares traded with 0.005 impact: {volume1}")
    print(f"Total shares traded with 0.01 impact: {volume2}")
    print(f"Total shares traded with 0.02 impact: {volume3}")
    f = open("t3.txt", 'w')


    # ms_portvals = msim.compute_portvals(ms_trades, symbol='JPM', start_val=100000, commission=9.95, impact=0.005)
    # benchmark = msim.compute_portvals(benchmark_trades, symbol='JPM', start_val=100000, commission=9.95, impact=0.005)
    sl_portvals1 = msim.compute_portvals(sl_trades1, symbol='JPM', start_val=100000, commission=9.95, impact=0.005)
    sl_portvals2 = msim.compute_portvals(sl_trades2, symbol='JPM', start_val=100000, commission=9.95, impact=0.01)
    sl_portvals3 = msim.compute_portvals(sl_trades3, symbol='JPM', start_val=100000, commission=9.95, impact=0.02)

    sl_portvals1_norm = sl_portvals1 / sl_portvals1[0]
    sl_portvals2_norm = sl_portvals2 / sl_portvals2[0]
    sl_portvals3_norm = sl_portvals3 / sl_portvals3[0]
    plt.plot(sl_portvals1_norm, label="0.005 impact", color="green")
    plt.plot(sl_portvals2_norm, label="0.01 impact", color="red")
    plt.plot(sl_portvals3_norm, label="0.02 impact", color="yellow")

    plt.title("Impact Comparison of Strategy Learner")
    plt.xlabel("Date")
    plt.ylabel("Normalized Value")
    plt.legend()
    # plt.show()
    plt.savefig('fig4.png')
    plt.clf()

    # Get portfolio stats
    # cum_ret = (portvals[-1] / portvals[0]) - 1
    # cum_ret_benchmark = (benchmark[-1] / benchmark[0]) - 1
    # daily_ret = (portvals[1:] / portvals[:-1].values) - 1
    # daily_ret_benchmark = (benchmark[1:] / benchmark[:-1].values) - 1
    # avg_daily_ret = daily_ret.mean()
    # avg_daily_ret_benchmark = daily_ret_benchmark.mean()
    # std_daily_ret = daily_ret.std()
    # std_daily_ret_benchmark = daily_ret_benchmark.std()

    # stdoutOrigin = sys.stdout
    # sys.stdout = open("p6_results.txt", 'w')
    # print("{:.4%}".format(cum_ret))
    # print("{:.4%}".format(cum_ret_benchmark))
    # print("{:.4%}".format(avg_daily_ret))
    # print("{:.4%}".format(avg_daily_ret_benchmark))
    # print("{:.4%}".format(std_daily_ret))
    # print("{:.4%}".format(std_daily_ret_benchmark))
    # f = open("p6_results.txt", 'w')


if __name__ == "__main__":
    register_matplotlib_converters()
    e2()
    # id.indicators()
