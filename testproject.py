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
import experiment1 as e1
import experiment2 as e2


def author():
    return 'zhe343'

def mstrat():
    ms_trades1 = ms.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    benchmark_trades1 = ms.benchmark(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)

    ms_portvals1 = msim.compute_portvals(ms_trades1, symbol='JPM', start_val=100000, commission=9.95, impact=0.005)
    benchmark1 = msim.compute_portvals(benchmark_trades1, symbol='JPM', start_val=100000, commission=9.95, impact=0.005)

    ms_trades2 = ms.testPolicy(symbol="JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000)
    benchmark_trades2 = ms.benchmark(symbol="JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000)

    ms_portvals2 = msim.compute_portvals(ms_trades2, symbol='JPM', start_val=100000, commission=9.95, impact=0.005)
    benchmark2 = msim.compute_portvals(benchmark_trades2, symbol='JPM', start_val=100000, commission=9.95, impact=0.005)

    ms_portvals_norm1 = ms_portvals1 / ms_portvals1[0]
    benchmark_norm1 = benchmark1 / benchmark1[0]
    plt.plot(benchmark_norm1, label="Benchmark", color="green")
    plt.plot(ms_portvals_norm1, label="Manual Strategy", color="red")
    plt.title("In-sample Manual Strategy")
    plt.xlabel("Date")
    plt.ylabel("Normalized Value")
    plt.legend()
    # plt.show()
    plt.savefig('fig1.png')
    plt.clf()

    ms_portvals_norm2 = ms_portvals2 / ms_portvals2[0]
    benchmark_norm2 = benchmark2 / benchmark2[0]
    plt.plot(benchmark_norm2, label="Benchmark", color="green")
    plt.plot(ms_portvals_norm2, label="Manual Strategy", color="red")
    plt.title("Out-sample Manual Strategy")
    plt.xlabel("Date")
    plt.ylabel("Normalized Value")
    plt.legend()
    # plt.show()
    plt.savefig('fig2.png')
    plt.clf()



    # Get portfolio stats
    cum_ret1 = (ms_portvals_norm1[-1] / ms_portvals_norm1[0]) - 1
    cum_ret_benchmark1 = (benchmark1[-1] / benchmark1[0]) - 1
    daily_ret1 = (ms_portvals_norm1[1:] / ms_portvals_norm1[:-1].values) - 1
    daily_ret_benchmark1 = (benchmark1[1:] / benchmark1[:-1].values) - 1
    avg_daily_ret1 = daily_ret1.mean()
    avg_daily_ret_benchmark1 = daily_ret_benchmark1.mean()
    std_daily_ret1 = daily_ret1.std()
    std_daily_ret_benchmark1 = daily_ret_benchmark1.std()

    #out sample
    cum_ret2 = (ms_portvals_norm2[-1] / ms_portvals_norm2[0]) - 1
    cum_ret_benchmark2 = (benchmark2[-1] / benchmark2[0]) - 1
    daily_ret2 = (ms_portvals_norm2[1:] / ms_portvals_norm2[:-1].values) - 1
    daily_ret_benchmark2 = (benchmark2[1:] / benchmark2[:-1].values) - 1
    avg_daily_ret2 = daily_ret2.mean()
    avg_daily_ret_benchmark2 = daily_ret_benchmark2.mean()
    std_daily_ret2 = daily_ret2.std()
    std_daily_ret_benchmark2 = daily_ret_benchmark2.std()

    stdoutOrigin = sys.stdout
    sys.stdout = open("t2.txt", 'w')
    print("{:.4%}".format(cum_ret1))
    print("{:.4%}".format(cum_ret_benchmark1))
    print("{:.4%}".format(avg_daily_ret1))
    print("{:.4%}".format(avg_daily_ret_benchmark1))
    print("{:.4%}".format(std_daily_ret1))
    print("{:.4%}".format(std_daily_ret_benchmark1))
    print("{:.4%}".format(cum_ret2))
    print("{:.4%}".format(cum_ret_benchmark2))
    print("{:.4%}".format(avg_daily_ret2))
    print("{:.4%}".format(avg_daily_ret_benchmark2))
    print("{:.4%}".format(std_daily_ret2))
    print("{:.4%}".format(std_daily_ret_benchmark2))
    f = open("t2.txt", 'w')



def test():
    e1.e1()
    e2.e2()


if __name__ == "__main__":
    register_matplotlib_converters()
    mstrat()
    test()
    # id.indicators()
