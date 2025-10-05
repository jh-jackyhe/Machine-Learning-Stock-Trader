
"""
Tiny data loader with on-disk cache powered by yfinance.

Usage (Python):
    from data import get_prices
    df = get_prices(["JPM", "SPY"], "2008-01-01", "2009-12-31")

CLI:
    python data.py --symbols JPM SPY --start 2008-01-01 --end 2009-12-31

Notes:
    - Caches per-symbol CSVs in `data_cache/` by default (one file per symbol+interval).
    - Subsequent calls merge newly fetched rows into the cache.
    - Returns a DataFrame indexed by Date with one column per symbol (default field: "Adj Close").
"""
# from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union
import pandas as pd
import yfinance as yf

def _to_ts(x) -> pd.Timestamp:
    ts = pd.Timestamp(x)
    # Make index tz-naive for consistency
    return ts.tz_localize(None) if ts.tzinfo is not None else ts

def _read_cached(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path, parse_dates=['Date']).set_index('Date')
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df.sort_index()
    return pd.DataFrame()

def _write_cache(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.reset_index().rename(columns={'index': 'Date'})
    out.to_csv(path, index=False)

def _fetch(symbol: str, start: pd.Timestamp, end: pd.Timestamp, interval: str='1d') -> pd.DataFrame:
    df = yf.download(
        symbol,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    # Normalize columns in case a MultiIndex shows up
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join([c for c in col if c]).strip() for col in df.columns]
    df.index.name = 'Date'
    df = df[~df.index.duplicated(keep='last')]
    # Make tz-naive
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    return df.sort_index()

def ensure_cached(
    symbol: str,
    start: Union[str, pd.Timestamp],
    end: Union[str, pd.Timestamp],
    interval: str='1d',
    cache_dir: str='data_cache'
) -> pd.DataFrame:
    start = _to_ts(start)
    end = _to_ts(end)
    cache_path = Path(cache_dir) / f"{symbol}_{interval}.csv"
    cached = _read_cached(cache_path)

    # If cache already spans the requested window, return it
    if not cached.empty:
        if cached.index.min() <= start and cached.index.max() >= (end - pd.Timedelta(days=1)):
            return cached

    # Otherwise fetch the requested window and merge into cache
    fetched = _fetch(symbol, start, end, interval=interval)
    if fetched.empty:
        return cached

    merged = fetched if cached.empty else pd.concat([cached, fetched]).sort_index()
    merged = merged[~merged.index.duplicated(keep='last')]
    _write_cache(merged, cache_path)
    return merged

def get_prices(
    symbols: Union[str, Iterable[str]],
    start: Union[str, pd.Timestamp],
    end: Union[str, pd.Timestamp],
    field: str='Adj Close',
    interval: str='1d',
    cache_dir: str='data_cache'
) -> pd.DataFrame:
    """Return a Date-indexed DataFrame with one column per symbol (default field: 'Adj Close')."""
    if isinstance(symbols, str):
        symbols = [symbols]

    frames = []
    for sym in symbols:
        df = ensure_cached(sym, start, end, interval=interval, cache_dir=cache_dir)
        if df.empty:
            frames.append(pd.Series(name=sym, dtype='float64'))
            continue
        # Slice to requested window
        sliced = df.loc[_to_ts(start):_to_ts(end)]
        if field not in sliced.columns:
            raise KeyError(f"Field '{field}' not found for {sym}. Available columns: {list(sliced.columns)}")
        series = sliced[field].rename(sym)
        frames.append(series)

    out = pd.concat(frames, axis=1)
    out.index.name = 'Date'
    return out.sort_index()

if __name__ == "__main__":
    df = get_prices(["JPM", "SPY"], "2008-01-01", "2009-12-31")


    import argparse

    parser = argparse.ArgumentParser(description="Fetch prices via yfinance with simple CSV caching.")
    parser.add_argument("--symbols", nargs="+", required=True, help="One or more tickers, e.g., JPM SPY")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--field", default="Adj Close", help="Column to return (e.g., 'Adj Close', 'Close')")
    parser.add_argument("--interval", default="1d", help="Bar interval (1d, 1wk, 1mo)")
    parser.add_argument("--cache-dir", default="data_cache", help="Cache directory")
    args = parser.parse_args()

    df = get_prices(args.symbols, args.start, args.end, field=args.field, interval=args.interval, cache_dir=args.cache_dir)
    # Print last few rows and also write a sample csv for quick inspection
    print(df.tail())
    Path("figures").mkdir(exist_ok=True)
    df.to_csv("figures/prices_sample.csv")
    print("Saved figures/prices_sample.csv")
