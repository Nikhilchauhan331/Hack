"""AI/NLP Risk Engine: ingest -> analyse -> structured signals (CSV + JSON)."""
import argparse
import os
import pandas as pd

from .ingest import load_news_csv, load_tweets_csv, fetch_newsapi, fetch_gdelt
from .nlp import analyze_text

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
OUT_DIR = os.path.join(BASE, "output")
DATA_DIR = os.path.join(BASE, "data")


def build_signals(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in raw.iterrows():
        for sig in analyze_text(str(r["text"]), r["source"]):
            rows.append({"timestamp": pd.to_datetime(r["timestamp"], utc=True).tz_localize(None),
                         "source": r["source"], "text": r["text"], **sig})
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def save_signals(df: pd.DataFrame, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "signals.csv"), index=False)
    df.assign(timestamp=df["timestamp"].astype(str)).to_json(
        os.path.join(out_dir, "signals.json"), orient="records", indent=2)


def load_signals(path=None) -> pd.DataFrame:
    path = path or os.path.join(OUT_DIR, "signals.csv")
    return pd.read_csv(path, parse_dates=["timestamp"])


def run(live=False) -> pd.DataFrame:
    frames = [load_news_csv(os.path.join(DATA_DIR, "sample_news.csv")),
              load_tweets_csv(os.path.join(DATA_DIR, "sample_tweets.csv"))]
    if live:
        for fn in (fetch_gdelt, fetch_newsapi):
            try:
                frames.append(fn())
            except Exception as e:
                print(f"[warn] live source {fn.__name__} skipped: {e}")
    df = build_signals(pd.concat(frames, ignore_index=True))
    save_signals(df)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="also pull GDELT/NewsAPI")
    out = run(ap.parse_args().live)
    print(out[["timestamp", "source", "company", "sentiment_score", "event_type", "impact_score"]]
          .tail(15).to_string(index=False))
    print(f"\n{len(out)} signals written to output/signals.csv and signals.json")
