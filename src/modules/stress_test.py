"""MODULE B - Strategic Portfolio Stress Testing triggered by high-impact events."""
import numpy as np
import pandas as pd

# Shocks at impact=10. They are scaled by impact_score/10.
SCENARIOS = {
    "Geopolitical":  dict(equity=-0.12, rate_shift=0.010, spread_shift=0.015, pd_mult=1.8, vol_up=0.30),
    "Macroeconomic": dict(equity=-0.10, rate_shift=0.020, spread_shift=0.010, pd_mult=1.5, vol_up=0.20),
    "Credit Event":  dict(equity=-0.08, rate_shift=0.005, spread_shift=0.030, pd_mult=2.5, vol_up=0.25),
    "Regulatory/Legal": dict(equity=-0.05, rate_shift=0.0, spread_shift=0.005, pd_mult=1.2, vol_up=0.10),
}
PD_BASE = {"AAA": .0002, "AA": .0005, "A": .001, "BBB": .004, "BB": .015, "B": .05}
LGD = 0.45
TRIGGER_IMPACT = 7.0


def load_portfolio(path):
    return pd.read_csv(path)


def find_triggers(signals: pd.DataFrame, threshold=TRIGGER_IMPACT):
    t = signals[(signals.impact_score > threshold) & (signals.event_type.isin(SCENARIOS))]
    return t.sort_values("impact_score", ascending=False)


def apply_stress(portfolio: pd.DataFrame, event_type: str, impact: float):
    sc = SCENARIOS[event_type]
    k = min(impact / 10.0, 1.0)
    eq, dy, ds, pdm = sc["equity"] * k, sc["rate_shift"] * k, sc["spread_shift"] * k, 1 + (sc["pd_mult"] - 1) * k
    p = portfolio.copy()
    p["value_before"] = p.notional.astype(float)
    pnl = np.zeros(len(p))
    for i, r in enumerate(p.itertuples()):
        if r.asset_type == "equity":
            pnl[i] = r.notional * eq
        elif r.asset_type == "bond":      # duration approximation: dP/P = -D*(dy + dspread)
            pnl[i] = -r.duration * (dy + ds) * r.notional
        elif r.asset_type == "loan":      # incremental expected loss from higher PD
            base = PD_BASE.get(r.rating, .01)
            pnl[i] = -(base * pdm - base) * LGD * r.notional - 0.5 * r.duration * ds * r.notional * 0.3
        elif r.asset_type == "derivative":  # delta-equivalent exposure
            pnl[i] = r.delta * r.notional * eq
    p["pnl"] = pnl
    p["value_after"] = p.value_before + p.pnl
    summary = p.groupby("asset_type")[["value_before", "value_after", "pnl"]].sum()
    return p, summary, dict(equity=eq, rate_shift=dy, spread_shift=ds, pd_multiplier=pdm)
