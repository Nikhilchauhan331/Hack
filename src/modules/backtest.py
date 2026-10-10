"""Sanity backtest: does today's sentiment rank-correlate with next-day returns?
Reports the Information Coefficient (Spearman). With simulated prices the IC is ~0 by construction;
use real prices (yfinance) to get a meaningful number."""
import pandas as pd
from risk_engine.engine import load_signals
from risk_engine.nlp import COMPANIES
from .prices import get_prices


def information_coefficient(signals, simulate=False):
    sig = signals[signals.company.isin(COMPANIES)].copy()
    sig["date"] = pd.to_datetime(sig.timestamp).dt.normalize()
    tickers = sorted(sig.company.unique())
    prices = get_prices(tickers, sig.date.min() - pd.Timedelta(days=1), sig.date.max() + pd.Timedelta(days=3), simulate)
    nxt = prices.pct_change().shift(-1)
    daily = sig.groupby(["date", "company"]).sentiment_score.mean().reset_index()
    daily["next_ret"] = [nxt.at[d, c] if d in nxt.index else float("nan") for d, c in zip(daily.date, daily.company)]
    daily = daily.dropna()
    ic = daily.sentiment_score.corr(daily.next_ret, method="spearman")
    hit = ((daily.sentiment_score > 0) == (daily.next_ret > 0)).mean()
    return {"observations": len(daily), "spearman_IC": round(float(ic), 3), "directional_hit_rate": round(float(hit), 3)}


def main(simulate=False):
    print(information_coefficient(load_signals(), simulate))
