"""
Create a chart that shows:
Value of the ManualStrategy portfolio (normalized to 1.0 at the start)
Value of the StrategyLearner portfolio (normalized to 1.0 at the start)
Value of the Benchmark portfolio (normalized to 1.0 at the start)
"""

import datetime as dt
import os
import traceback

import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import register_matplotlib_converters

import ManualStrategy as ms
import experiment1 as e1
import experiment2 as e2
import marketsimcode as msim


def pct(x): return f"{x:.4%}"


def mstrat():
    # Create results directory if it doesn't exist
    results_dir = '../results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    ms_trades1 = ms.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    benchmark_trades1 = ms.benchmark(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)

    ms_portvals1 = msim.compute_portvals(ms_trades1, symbol='JPM', start_val=100000, commission=9.95, impact=0.005)
    benchmark1 = msim.compute_portvals(benchmark_trades1, symbol='JPM', start_val=100000, commission=9.95, impact=0.005)

    ms_trades2 = ms.testPolicy(symbol="JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000)
    benchmark_trades2 = ms.benchmark(symbol="JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000)

    ms_portvals2 = msim.compute_portvals(ms_trades2, symbol='JPM', start_val=100000, commission=9.95, impact=0.005)
    benchmark2 = msim.compute_portvals(benchmark_trades2, symbol='JPM', start_val=100000, commission=9.95, impact=0.005)

    ms_portvals_norm1 = ms_portvals1 / ms_portvals1.iloc[0]
    benchmark_norm1 = benchmark1 / benchmark1.iloc[0]
    plt.plot(benchmark_norm1, label="Benchmark", color="green")
    plt.plot(ms_portvals_norm1, label="Manual Strategy", color="red")
    plt.title("In-sample Manual Strategy")
    plt.xlabel("Date")
    plt.ylabel("Normalized Value")
    plt.legend()
    # plt.show()
    plt.savefig('../results/fig1.png')
    plt.clf()

    ms_portvals_norm2 = ms_portvals2 / ms_portvals2.iloc[0]
    benchmark_norm2 = benchmark2 / benchmark2.iloc[0]
    plt.plot(benchmark_norm2, label="Benchmark", color="green")
    plt.plot(ms_portvals_norm2, label="Manual Strategy", color="red")
    plt.title("Out-sample Manual Strategy")
    plt.xlabel("Date")
    plt.ylabel("Normalized Value")
    plt.legend()
    # plt.show()
    plt.savefig('../results/fig2.png')
    plt.clf()

    # Get portfolio stats
    cum_ret1 = (ms_portvals_norm1.iloc[-1] / ms_portvals_norm1.iloc[0]) - 1
    cum_ret_benchmark1 = (benchmark1.iloc[-1] / benchmark1.iloc[0]) - 1
    daily_ret1 = (ms_portvals_norm1[1:] / ms_portvals_norm1[:-1].values) - 1
    daily_ret_benchmark1 = (benchmark1[1:] / benchmark1[:-1].values) - 1
    avg_daily_ret1 = daily_ret1.mean()
    avg_daily_ret_benchmark1 = daily_ret_benchmark1.mean()
    std_daily_ret1 = daily_ret1.std()
    std_daily_ret_benchmark1 = daily_ret_benchmark1.std()

    # out sample
    cum_ret2 = (ms_portvals_norm2.iloc[-1] / ms_portvals_norm2.iloc[0]) - 1
    cum_ret_benchmark2 = (benchmark2.iloc[-1] / benchmark2.iloc[0]) - 1
    daily_ret2 = (ms_portvals_norm2[1:] / ms_portvals_norm2[:-1].values) - 1
    daily_ret_benchmark2 = (benchmark2[1:] / benchmark2[:-1].values) - 1
    avg_daily_ret2 = daily_ret2.mean()
    avg_daily_ret_benchmark2 = daily_ret_benchmark2.mean()
    std_daily_ret2 = daily_ret2.std()
    std_daily_ret_benchmark2 = daily_ret_benchmark2.std()

    table = pd.DataFrame(
        {
            "Manual Strategy in-sample": [
                pct(cum_ret1),
                pct(avg_daily_ret1),
                pct(std_daily_ret1),
            ],
            "Benchmark in-sample": [
                pct(cum_ret_benchmark1),
                pct(avg_daily_ret_benchmark1),
                pct(std_daily_ret_benchmark1),
            ],
            "Manual Strategy out-sample": [
                pct(cum_ret2),
                pct(avg_daily_ret2),
                pct(std_daily_ret2),
            ],
            "Benchmark out-sample": [
                pct(cum_ret_benchmark2),
                pct(avg_daily_ret_benchmark2),
                pct(std_daily_ret_benchmark2),
            ],
        },
        index=[
            "cumulative returns",
            "mean of daily returns",
            "standard deviation of daily returns",
        ],
    )

    table.to_markdown("../results/t1.md")


def test():
    e1.e1()
    e2.e2()


def main():
    print("[testproject] starting…")
    register_matplotlib_converters()
    try:
        mstrat()   # should save charts/tables and print file paths
        test()     # should save charts/tables and print file paths
    except Exception as e:
        print("[testproject] ERROR:", e)
        traceback.print_exc()
    else:
        print("[testproject] done.")

if __name__ == "__main__":
    main()