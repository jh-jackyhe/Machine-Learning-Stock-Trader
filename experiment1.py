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


def e1():
    ms_trades = ms.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    benchmark_trades = ms.benchmark(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)

    sl_learner = sl.StrategyLearner(verbose=False, impact=0.005, commission=9.95)  # constructor
    sl_learner.add_evidence(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)  # training phase
    sl_trades = sl_learner.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)  # testing phase

    ms_portvals = msim.compute_portvals(ms_trades, symbol='JPM', start_val=100000, commission=9.95, impact=0.005)
    benchmark = msim.compute_portvals(benchmark_trades, symbol='JPM', start_val=100000, commission=9.95, impact=0.005)
    sl_portvals = msim.compute_portvals(sl_trades, symbol='JPM', start_val=100000, commission=9.95, impact=0.005)


    ms_portvals_norm = ms_portvals / ms_portvals[0]
    sl_portvals_norm = sl_portvals / sl_portvals[0]
    benchmark_norm = benchmark / benchmark[0]
    plt.plot(benchmark_norm, label="Benchmark", color="green")
    plt.plot(ms_portvals_norm, label="Manual Strategy", color="red")
    plt.plot(sl_portvals_norm, label="Random Forest", color="yellow")


    plt.title("Strategy Comparison")
    plt.xlabel("Date")
    plt.ylabel("Normalized Value")
    plt.legend()
    # plt.show()
    plt.savefig('fig3.png')
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
    e1()
    # id.indicators()
