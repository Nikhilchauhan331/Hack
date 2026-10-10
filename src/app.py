"""Streamlit dashboard: Engine demo + Module A + Module B.   Run: streamlit run app.py"""
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from risk_engine.engine import run, load_signals, OUT_DIR, DATA_DIR
from risk_engine.nlp import analyze_text, COMPANIES
from modules.rebalancer import rebalance
from modules.stress_test import load_portfolio, find_triggers, apply_stress, SCENARIOS

st.set_page_config(page_title="AI/NLP Risk Engine", layout="wide")
st.title("AI/NLP Financial Risk Engine")

with st.sidebar:
    st.header("Engine")
    live = st.checkbox("Also pull live GDELT / NewsAPI", value=False)
    sim = st.checkbox("Use simulated prices (offline)", value=False)
    if st.button("Run engine now") or not os.path.exists(os.path.join(OUT_DIR, "signals.csv")):
        with st.spinner("Processing text..."):
            run(live)
        st.success("Signals refreshed")
signals = load_signals()

t0, t1, t2, t3 = st.tabs(["Engine & Signals", "Module A: Index Rebalancer", "Module B: Stress Test", "Try your own text"])

with t0:
    c = st.columns(4)
    c[0].metric("Signals", len(signals)); c[1].metric("Sources", signals.source.nunique())
    c[2].metric("Avg sentiment", f"{signals.sentiment_score.mean():.2f}")
    c[3].metric("High impact (>7)", int((signals.impact_score > 7).sum()))
    st.plotly_chart(px.histogram(signals, x="event_type", color="source", title="Events by type"), use_container_width=True)
    st.dataframe(signals.sort_values("timestamp", ascending=False), use_container_width=True)

with t1:
    st.subheader("Tactical Index Rebalancing (sentiment-driven)")
    all_t = list(COMPANIES)
    tickers = st.multiselect("Index constituents (10-20)", all_t, default=all_t[:15])
    a, d = st.columns(2)
    alpha = a.slider("Sensitivity (alpha)", 0.1, 2.0, 0.5); decay = d.slider("Sentiment memory (decay)", 0.0, 0.9, 0.5)
    if len(tickers) >= 5:
        w, idx = rebalance(signals, tickers, alpha=alpha, decay=decay, simulate_prices=sim)
        st.plotly_chart(px.area(w, title="Index weights over time", labels={"value": "weight", "index": "date"}), use_container_width=True)
        st.plotly_chart(px.line(idx, title="Index value (base 100)"), use_container_width=True)
        chg = (w.iloc[-1] - w.iloc[0]).sort_values()
        st.plotly_chart(px.bar(chg, title="Weight change (start to end)"), use_container_width=True)

with t2:
    st.subheader("Strategic Portfolio Stress Testing")
    pf = load_portfolio(os.path.join(DATA_DIR, "sample_transactions.csv"))
    st.caption(f"Synthetic portfolio: {len(pf)} positions, notional {pf.notional.sum()/1e6:,.0f}M")
    thr = st.slider("Trigger when Impact Score >", 1.0, 9.5, 7.0)
    trig = find_triggers(signals, thr)
    if trig.empty:
        st.info("No high-impact events detected; lower the threshold or add data.")
    else:
        trig = trig.reset_index(drop=True)
        pick = st.selectbox("Detected high-impact event", trig.index,
                            format_func=lambda i: f"[{trig.event_type[i]} | impact {trig.impact_score[i]}] {trig.text[i][:90]}")
        ev = trig.loc[pick]
        pos, summ, shocks = apply_stress(pf, ev.event_type, ev.impact_score)
        st.write("**Shocks applied:**", {k: round(v, 4) for k, v in shocks.items()})
        before, after = summ.value_before.sum(), summ.value_after.sum()
        m = st.columns(3); m[0].metric("Before", f"{before/1e6:,.1f}M"); m[1].metric("After", f"{after/1e6:,.1f}M", f"{(after-before)/1e6:,.2f}M"); m[2].metric("Loss %", f"{(after/before-1)*100:.2f}%")
        fig = go.Figure([go.Bar(name="Before", x=summ.index, y=summ.value_before), go.Bar(name="After", x=summ.index, y=summ.value_after)])
        fig.update_layout(barmode="group", title="Portfolio value by asset class, before vs after stress")
        st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(px.bar(pos.groupby("sector").pnl.sum().sort_values(), title="P&L impact by sector"), use_container_width=True)
        st.dataframe(pos.sort_values("pnl").head(10))

with t3:
    txt = st.text_area("Paste a headline or tweet", "Fed signals rate hike as inflation fears grow; Apple shares plunge sharply")
    src = st.radio("Source", ["news", "twitter"], horizontal=True)
    if txt:
        st.json(analyze_text(txt, src))
