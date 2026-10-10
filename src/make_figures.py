"""Regenerates docs/architecture.png and docs/results_*.png from the current output/ + data/.
Run:  python src/make_figures.py [--simulate-prices]"""
import os, sys, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from risk_engine.engine import load_signals, DATA_DIR
from modules.rebalancer import rebalance
from modules.stress_test import find_triggers, apply_stress, load_portfolio

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DOCS = os.path.join(ROOT, "docs")
NAVY, TEAL, AMBER, RED, GREY = "#0B1F3A", "#0F8B8D", "#F2A541", "#C8553D", "#6B7A8F"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
simulate = "--simulate-prices" in sys.argv


def box(ax, x, y, w, h, title, sub, color, txt="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=color, ec="none"))
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", color=txt, fontsize=11, fontweight="bold")
    ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", color=txt, fontsize=8)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, color=GREY, lw=1.6))


def architecture():
    fig, ax = plt.subplots(figsize=(14, 7.2), dpi=170)
    ax.set_xlim(0, 14); ax.set_ylim(0, 7.2); ax.axis("off")
    ax.text(0.3, 6.8, "AI/NLP Financial Risk Engine - architecture & data flow", fontsize=15, fontweight="bold", color=NAVY)
    # sources
    box(ax, 0.3, 4.6, 2.5, 1.0, "Financial news", "GDELT / NewsAPI / CSV", GREY)
    box(ax, 0.3, 3.3, 2.5, 1.0, "Twitter / X posts", "Kaggle tweets CSV", GREY)
    box(ax, 0.3, 1.2, 2.5, 1.0, "Market prices", "yfinance (sim. fallback)", GREY)
    box(ax, 0.3, 0.0, 2.5, 0.9, "Transactions", "loans, bonds, derivatives", GREY)
    # ingestion
    box(ax, 3.5, 3.7, 2.2, 1.5, "Ingestion", "unify to\n[timestamp, source, text]", TEAL)
    # engine
    ax.add_patch(FancyBboxPatch((6.1, 2.85), 3.7, 3.55, boxstyle="round,pad=0.02,rounding_size=0.15", fc="#E8F3F3", ec=TEAL, lw=2))
    ax.text(7.95, 6.15, "NLP RISK ENGINE", ha="center", fontsize=11, fontweight="bold", color=NAVY)
    rows = [("Entity detection", "S&P 100 aliases + $cashtags"), ("Sentiment  (-1 to +1)", "FinBERT  |  VADER + finance lexicon"),
            ("Event classifier", "keyword rules  |  zero-shot"), ("Impact score (1-10)", "severity x intensity x source")]
    for i, (t, sub) in enumerate(rows):
        y = 5.35 - i * 0.62
        ax.add_patch(FancyBboxPatch((6.3, y), 3.3, 0.52, boxstyle="round,pad=0.01,rounding_size=0.08", fc=NAVY, ec="none"))
        ax.text(6.45, y + 0.36, t, color="white", fontsize=9, fontweight="bold", va="center")
        ax.text(6.45, y + 0.14, sub, color="#BFD3E6", fontsize=7, va="center")
    # output
    box(ax, 10.5, 3.7, 3.2, 1.5, "Structured signals", "output/signals.csv | .json\nFastAPI  /signals  /analyze", AMBER, NAVY)
    # modules
    box(ax, 7.0, 0.95, 3.0, 1.45, "Module A", "Tactical index rebalancer\nweights follow sentiment", TEAL)
    box(ax, 10.7, 0.95, 3.0, 1.45, "Module B", "Stress tester\nimpact > 7 triggers shocks", RED)
    ax.text(10.3, 2.62, "Streamlit dashboard shows both modules + live text tester", ha="center", fontsize=8.5, color=NAVY, style="italic")
    for (a, b, c, d) in [(2.8, 5.1, 3.5, 4.7), (2.8, 3.8, 3.5, 4.2), (5.7, 4.45, 6.1, 4.45), (9.8, 4.45, 10.5, 4.45),
                         (11.2, 3.7, 9.0, 2.4), (12.6, 3.7, 12.2, 2.4), (2.8, 1.7, 7.0, 1.7)]:
        arrow(ax, a, b, c, d)
    ax.plot([2.8, 12.2], [0.45, 0.45], color=GREY, lw=1.6); arrow(ax, 12.2, 0.45, 12.2, 0.95)
    fig.savefig(os.path.join(DOCS, "architecture.png"), bbox_inches="tight", facecolor="white"); plt.close(fig)


def results():
    s = load_signals()
    # --- engine overview
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=170)
    ct = s.groupby(["event_type", "source"]).size().unstack(fill_value=0)
    ct = ct.loc[ct.sum(axis=1).sort_values().index]
    ct.plot.barh(stacked=True, ax=ax[0], color=[TEAL, AMBER]); ax[0].set_title("Signals by event type", color=NAVY, fontweight="bold"); ax[0].set_ylabel("")
    ax[1].hist(s.impact_score, bins=range(1, 11), color=NAVY, edgecolor="white"); ax[1].axvline(7, color=RED, ls="--", lw=2)
    ax[1].text(7.1, ax[1].get_ylim()[1] * 0.9, "stress-test\ntrigger > 7", color=RED, fontsize=9)
    ax[1].set_title("Impact score distribution", color=NAVY, fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(DOCS, "results_engine.png")); plt.close(fig)

    # --- module A
    tk = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM", "BAC", "GS", "XOM", "JNJ", "PFE", "WMT", "KO"]
    w, idx = rebalance(s, tk, simulate_prices=simulate)
    chg = (w.iloc[-1] - w.iloc[0]).sort_values()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=170)
    colors = [RED if v < 0 else TEAL for v in chg]
    ax[0].barh(chg.index, chg.values * 100, color=colors); ax[0].set_title("Weight change, start to end (pp)", color=NAVY, fontweight="bold")
    for t in list(chg.index[:2]) + list(chg.index[-2:]):
        ax[1].plot(w.index, w[t] * 100, label=t, lw=2)
    ax[1].axhline(100 / len(tk), color=GREY, ls=":"); ax[1].set_title("Largest weight moves (% of index)", color=NAVY, fontweight="bold"); ax[1].legend(frameon=False, fontsize=8)
    plt.setp(ax[1].get_xticklabels(), rotation=30, ha="right", fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(DOCS, "results_module_a.png")); plt.close(fig)

    # --- module B
    pf = load_portfolio(os.path.join(DATA_DIR, "sample_transactions.csv"))
    trig = find_triggers(s)
    r = trig.iloc[0]
    pos, summ, sh = apply_stress(pf, r.event_type, r.impact_score)
    naive = summ.value_before * 0.9
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=170)
    x = range(len(summ)); wd = 0.27
    ax[0].bar([i - wd for i in x], summ.value_before / 1e6, wd, label="Before", color=NAVY)
    ax[0].bar(list(x), summ.value_after / 1e6, wd, label="After (event-aware)", color=RED)
    ax[0].bar([i + wd for i in x], naive / 1e6, wd, label="After (naive flat -10%)", color=GREY)
    ax[0].set_xticks(list(x)); ax[0].set_xticklabels(summ.index); ax[0].set_ylabel("Value (M)"); ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title(f"{r.event_type} event, impact {r.impact_score}", color=NAVY, fontweight="bold")
    sec = pos.groupby("sector").pnl.sum().sort_values() / 1e6
    ax[1].barh(sec.index, sec.values, color=RED); ax[1].set_title("P&L impact by sector (M)", color=NAVY, fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(DOCS, "results_module_b.png")); plt.close(fig)

    tot_b, tot_a = summ.value_before.sum(), summ.value_after.sum()
    stats = {"signals": int(len(s)), "sources": int(s.source.nunique()), "high_impact": int((s.impact_score > 7).sum()),
             "n_triggers": int(len(trig)), "event": r.event_type, "impact": float(r.impact_score),
             "portfolio_m": round(tot_b / 1e6, 1), "loss_m": round((tot_b - tot_a) / 1e6, 1), "loss_pct": round((1 - tot_a / tot_b) * 100, 2),
             "naive_loss_m": round(tot_b * 0.1 / 1e6, 1),
             "by_asset_loss_m": {k: round(-v / 1e6, 1) for k, v in summ.pnl.items()},
             "top_up": chg.index[-1], "top_up_pp": round(chg.iloc[-1] * 100, 1),
             "top_down": chg.index[0], "top_down_pp": round(chg.iloc[0] * 100, 1)}
    json.dump(stats, open(os.path.join(DOCS, "key_numbers.json"), "w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    os.makedirs(DOCS, exist_ok=True)
    architecture(); results()
