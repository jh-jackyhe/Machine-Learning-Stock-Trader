
# Strategy Evaluation

Implementation of my Machine Learning trading strategy evaluation.

- Indicators: Bollinger Band percentage, Exponential moving average ratio, Moving Average Convergence Divergence
- ML strategy: RandomForestClassifier → signals in {-1, 0, 1}
- Market sim: vectorized with commission & market impact; target holdings ∈ {-1000, 0, +1000}

## How it works

**Core tasks**
- Build a **Manual Strategy** that combines **3 indicators above**
- Build a **Strategy Learner** using the **same indicators** (classification-based learner).
- Provide **add_evidence(...)** (train on a symbol & time window) and **testPolicy(...)** (predict trades with learning off).
- Run **experiments** and compare to a **benchmark**.

**Data & periods (report convention)**
- Report trades for **JPM**. You may use other symbols (e.g., SPY) to inform signals.
- **In-sample**: 2008‑01‑01 → 2009‑12‑31
- **Out-of-sample**: 2010‑01‑01 → 2011‑12‑31
- Benchmark: buy **1000 shares** on the first day, hold, normalize charts to **1.0** at start.
- See results/t1.md, results/fig1.png, results/fig2.png for Benchmark and Manual Strategy performance metrics and charts.

**Experiment 1 (Manual vs Learner, in-sample)**
- Plot normalized portfolio values for **Manual**, **Learner**, and **Benchmark** on the same chart.
- Discuss why the strategies differ; include a table with **Cumulative Return**, **Mean** and **StdDev** of daily returns.
- See results/fig3.png for Manual Strategy performance metrics

**Experiment 2 (impact sensitivity)**
- Vary **market impact** (e.g., 0.0 / 0.005 / 0.01+) and analyze how trades & results change.
- State a hypothesis and report at least two metrics across ≥3 impact settings.
- See results/t2.md, results/fig4.png for Impact Comparison of Strategy Learner

**Trading conventions (used here)**
- Orders move position toward target **{-1000, 0, +1000}** (delta trades).
- Simulator applies **commission** and **market impact**: buy at _p·(1+impact)_, sell at _p·(1‑impact)_.

## Quick start
```bash
conda env create -f environment.yml
conda activate ml4t
python src/testproject.py
```

## Repo layout
```
src/
  indicators.py         # indicators (3 used by both strategies)
  StrategyLearner.py   # sklearn RF, same API (add_evidence, testPolicy)
  ManualStrategy.py    # rule-based strategy
  marketsimcode.py          # commission + impact simulator
  util.py               # get_data(...)
results/                # generated charts and tables
```

## License & provenance
- Code is MIT-licensed; indicators & strategies are my own implementations.
