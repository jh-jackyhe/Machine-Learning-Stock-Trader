"""
Create a chart that shows:
Value of the ManualStrategy portfolio (normalized to 1.0 at the start)
Value of the StrategyLearner portfolio (normalized to 1.0 at the start)
Value of the Benchmark portfolio (normalized to 1.0 at the start)
"""

import datetime as dt

import matplotlib.pyplot as plt

import ManualStrategy as ms, marketsimcode as msim, StrategyLearner as sl


def e1():
    ms_trades = ms.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    benchmark_trades = ms.benchmark(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)

    sl_learner = sl.StrategyLearner(verbose=False, impact=0.005, commission=9.95)  # constructor
    sl_learner.add_evidence(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),
                            sv=100000)  # training phase
    sl_trades = sl_learner.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),
                                      sv=100000)  # testing phase

    ms_portvals = msim.compute_portvals(ms_trades, symbol='JPM', start_val=100000, commission=9.95, impact=0.005)
    benchmark = msim.compute_portvals(benchmark_trades, symbol='JPM', start_val=100000, commission=9.95, impact=0.005)
    sl_portvals = msim.compute_portvals(sl_trades, symbol='JPM', start_val=100000, commission=9.95, impact=0.005)

    ms_portvals_norm = ms_portvals / ms_portvals.iloc[0]
    sl_portvals_norm = sl_portvals / sl_portvals.iloc[0]
    benchmark_norm = benchmark / benchmark.iloc[0]
    plt.plot(benchmark_norm, label="Benchmark", color="green")
    plt.plot(ms_portvals_norm, label="Manual Strategy", color="red")
    plt.plot(sl_portvals_norm, label="Random Forest", color="yellow")

    plt.title("Strategy Comparison")
    plt.xlabel("Date")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.savefig('../results/fig3.png')
    plt.clf()
