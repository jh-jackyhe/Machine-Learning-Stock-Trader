"""
Create a chart that shows:
Value of the ManualStrategy portfolio (normalized to 1.0 at the start)
Value of the StrategyLearner portfolio (normalized to 1.0 at the start)
Value of the Benchmark portfolio (normalized to 1.0 at the start)
"""

import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import marketsimcode as msim, StrategyLearner as sl


def e2():
    sl_learner1 = sl.StrategyLearner(verbose=False, impact=0.005, commission=0)  # constructor
    sl_learner1.add_evidence(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),
                             sv=100000)  # training phase
    sl_trades1 = sl_learner1.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),
                                        sv=100000)  # testing phase

    sl_learner2 = sl.StrategyLearner(verbose=False, impact=0.01, commission=0)  # constructor
    sl_learner2.add_evidence(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),
                             sv=100000)  # training phase
    sl_trades2 = sl_learner2.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),
                                        sv=100000)  # testing phase

    sl_learner3 = sl.StrategyLearner(verbose=False, impact=0.02, commission=0)  # constructor
    sl_learner3.add_evidence(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),
                             sv=100000)  # training phase
    sl_trades3 = sl_learner3.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),
                                        sv=100000)  # testing phase

    volume1 = (sl_trades1.abs()).sum()
    volume2 = (sl_trades2.abs()).sum()
    volume3 = (sl_trades3.abs()).sum()

    impacts = [0.005, 0.01, 0.02]
    volumes = [volume1, volume2, volume3]

    t3 = pd.DataFrame(
        {"Impact": impacts, "Traded shares": volumes}
    ).sort_values("Impact")

    t3_fmt = t3.copy()
    t3_fmt["Impact"] = t3_fmt["Impact"].map(lambda x: f"{x:g}")  # 0.005, 0.01, 0.02
    t3_fmt["Traded shares"] = t3_fmt["Traded shares"].map(
        lambda n: f"{int(n.iloc[0] if isinstance(n, pd.Series) else n):,}")

    t3_fmt.to_markdown("../results/t2.md", index=False)

    sl_portvals1 = msim.compute_portvals(sl_trades1, symbol='JPM', start_val=100000, commission=9.95, impact=0.005)
    sl_portvals2 = msim.compute_portvals(sl_trades2, symbol='JPM', start_val=100000, commission=9.95, impact=0.01)
    sl_portvals3 = msim.compute_portvals(sl_trades3, symbol='JPM', start_val=100000, commission=9.95, impact=0.02)

    sl_portvals1_norm = sl_portvals1 / sl_portvals1.iloc[0]
    sl_portvals2_norm = sl_portvals2 / sl_portvals2.iloc[0]
    sl_portvals3_norm = sl_portvals3 / sl_portvals3.iloc[0]
    plt.plot(sl_portvals1_norm, label="0.005 impact", color="green")
    plt.plot(sl_portvals2_norm, label="0.01 impact", color="red")
    plt.plot(sl_portvals3_norm, label="0.02 impact", color="yellow")

    plt.title("Impact Comparison of Strategy Learner")
    plt.xlabel("Date")
    plt.ylabel("Normalized Value")
    plt.legend()
    # plt.show()
    plt.savefig('../results/fig4.png')
    plt.clf()
