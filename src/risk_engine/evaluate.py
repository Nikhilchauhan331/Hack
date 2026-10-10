"""Evaluate the NLP engine on a small hand-labelled set (indicative only, n~40)."""
import os
import pandas as pd
from .engine import DATA_DIR
from .nlp import sentiment_score, classify_event


def to_label(x, thr=0.15):
    return "positive" if x > thr else "negative" if x < -thr else "neutral"


def evaluate(path=None):
    df = pd.read_csv(path or os.path.join(DATA_DIR, "eval_labeled.csv"))
    scored = df.text.map(lambda t: sentiment_score(t)[0])
    df["pred_sent"] = scored.map(to_label)
    df["pred_event"] = df.text.map(lambda t: classify_event(t)[0])
    ev_rows = df[df.source == "news"]  # tweets rarely state an event: scored on sentiment only
    out = {
        "n": len(df),
        "sentiment_accuracy": round((df.pred_sent == df.true_sentiment).mean(), 3),
        "sentiment_direction_accuracy(pos/neg only)": round(
            (df[df.true_sentiment != "neutral"].pred_sent == df[df.true_sentiment != "neutral"].true_sentiment).mean(), 3),
        "event_accuracy(news)": round((ev_rows.pred_event == ev_rows.true_event).mean(), 3),
        "event_baseline_always_Other": round((ev_rows.true_event == "Other").mean(), 3),
        "sentiment_backend": sentiment_score("test")[1],
    }
    return df, out


def main():
    df, out = evaluate()
    for k, v in out.items():
        print(f"{k:50s} {v}")
    bad = df[(df.pred_sent != df.true_sentiment) | ((df.source == 'news') & (df.pred_event != df.true_event))]
    if len(bad):
        print("\nMisclassified:\n", bad[["text", "true_sentiment", "pred_sent", "true_event", "pred_event"]].to_string(index=False))


if __name__ == "__main__":
    main()
