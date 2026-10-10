"""MODULE A - Tactical Index Rebalancing driven by NLP sentiment signals."""
import numpy as np
import pandas as pd
from .prices import get_prices


def _clip_normalise(w, wmin, wmax):
    for _ in range(20):
        w = w.clip(wmin, wmax)
        w = w / w.sum()
        if w.min() >= wmin - 1e-9 and w.max() <= wmax + 1e-9:
            break
    return w


def rebalance(signals: pd.DataFrame, tickers, alpha=0.5, decay=0.5,
              wmin=0.02, wmax=0.15, simulate_prices=False):
    """Positive sentiment -> weight up, negative -> weight down.
    Returns (weights_history, index_df)."""
    sig = signals[signals.company.isin(tickers)].copy()
    sig["date"] = pd.to_datetime(sig.timestamp).dt.normalize()
    start, end = sig.date.min() - pd.Timedelta(days=1), sig.date.max()
    prices = get_prices(list(tickers), start, end, simulate_prices)
    # sentiment weighted by impact so big events move weights more
    sig["ws"] = sig.sentiment_score * (sig.impact_score / 10)
    daily = sig.groupby(["date", "company"]).ws.mean().unstack().reindex(prices.index).fillna(0.0)
    daily = daily.reindex(columns=tickers).fillna(0.0)

    n = len(tickers)
    w = pd.Series(1 / n, index=tickers)
    ema = pd.Series(0.0, index=tickers)
    rets = prices.pct_change().fillna(0.0)
    hist, idx_val, bench_val = [], [100.0], [100.0]
    for i, d in enumerate(prices.index):
        if i > 0:  # earn return using yesterday's weights
            idx_val.append(idx_val[-1] * (1 + (hist[-1] * rets.loc[d]).sum()))
            bench_val.append(bench_val[-1] * (1 + rets.loc[d].mean()))
        ema = decay * ema + (1 - decay) * daily.loc[d]
        w = _clip_normalise(w * (1 + alpha * ema * 5), wmin, wmax)
        hist.append(w.copy())
    weights = pd.DataFrame(hist, index=prices.index)
    index_df = pd.DataFrame({"Sentiment Index": idx_val, "Equal-Weight Benchmark": bench_val},
                            index=prices.index)
    return weights, index_df
