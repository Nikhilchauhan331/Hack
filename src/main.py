"""Command-line entry point.
  python src/main.py            -> run the risk engine on /data and write output/signals.csv|json
  python src/main.py --live     -> additionally pull live GDELT / NewsAPI headlines
  python src/main.py --eval     -> evaluate the NLP engine on data/eval_labeled.csv
  python src/main.py --backtest -> sentiment vs next-day-return check (Information Coefficient)
"""
import argparse
from risk_engine.engine import run

ap = argparse.ArgumentParser()
ap.add_argument("--live", action="store_true")
ap.add_argument("--eval", action="store_true")
ap.add_argument("--backtest", action="store_true")
ap.add_argument("--simulate-prices", action="store_true", help="offline price simulation")
a = ap.parse_args()

if a.eval:
    from risk_engine.evaluate import main as ev
    ev()
elif a.backtest:
    from modules.backtest import main as bt
    bt(simulate=a.simulate_prices)
else:
    df = run(a.live)
    cols = ["timestamp", "source", "company", "sentiment_score", "event_type", "impact_score"]
    print(df[cols].tail(12).to_string(index=False))
    print(f"\n{len(df)} signals -> output/signals.csv, output/signals.json "
          f"(sentiment backend: {df.sentiment_backend.iloc[0]})")
