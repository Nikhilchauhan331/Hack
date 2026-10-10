"""Generates sample news / tweets / transactions (offline-friendly).
Replace with Kaggle datasets (see README) for real runs."""
import os
import numpy as np, pandas as pd
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SEED = int(os.getenv("SAMPLE_DATA_SEED", "7"))
END_DATE = pd.Timestamp(
    os.getenv("SAMPLE_DATA_END_DATE", pd.Timestamp.today().strftime("%Y-%m-%d"))
).normalize()
rng = np.random.default_rng(SEED)
days = pd.bdate_range(end=END_DATE, periods=15)

news = [
 "Apple beats earnings expectations as iPhone revenue surges to a record",
 "Microsoft unveils new Azure AI product, analysts upgrade the stock",
 "Amazon faces antitrust probe; shares slump on regulatory fears",
 "Nvidia profit soars on strong demand, raises quarterly guidance",
 "Tesla recall hits 200,000 vehicles, shares plunge",
 "JPMorgan warns of rising loan losses as credit conditions weaken",
 "Goldman Sachs downgraded after weak trading revenue misses estimates",
 "Boeing faces lawsuit and investigation, stock falls sharply",
 "Pfizer receives approval for new drug, shares gain",
 "Meta Platforms to acquire AI startup in major acquisition deal",
 "Exxon profit growth strong as oil rallies amid tensions",
 "Netflix subscriber growth beats estimates, bullish outlook",
 "Intel misses revenue guidance, layoffs announced",
 "Walmart expands dividend after record quarterly profit",
 "Disney launches new streaming product; analysts optimistic",
 "Federal Reserve signals rate hike as inflation fears grow, markets selloff",
 "Geopolitical tensions escalate as new sanctions and missile conflict shock global markets",
 "Recession fears rise after weak GDP and jobs report, central bank warns",
 "Credit rating downgrade triggers bond default fears and liquidity crisis in banks",
 "Ceasefire talks ease war tensions, markets rally",
 "Bank of America reports strong gains, loan growth boosts profit",
 "Cisco unveils rollout of new security product",
 "Coca-Cola profit weak as inflation hits margins",
 "Google faces major antitrust lawsuit, fine possible",
]
tweets = [
 "Loving the new iPhone, $AAPL going to the moon! bullish",
 "Tesla is a scam, crash incoming. bearish $TSLA",
 "Nvidia absolutely crushing it, record gains again",
 "Amazon layoffs again? weak outlook imo",
 "BREAKING: war escalates, markets crash, sanctions announced",
 "Goldman Sachs fraud probe?? this is bad",
 "Microsoft Azure outage, not good for MSFT",
 "Fed rate hike fears are a big shock to the market",
 "Netflix wins again, strong growth",
 "Boeing recall is a disaster, shares plunge",
 "Disney new launch looks amazing, optimism high",
 "Intel losses mounting, bearish",
]
def mk(texts, n):
    ts = [days[rng.integers(0, len(days))] + pd.Timedelta(hours=int(rng.integers(8, 20))) for _ in range(n)]
    return pd.DataFrame({"timestamp": ts, "text": texts[:n]})
reps = news * 2; mk(reps, len(reps)).sort_values("timestamp").to_csv(os.path.join(ROOT, "data", "sample_news.csv"), index=False)
reps = tweets * 3; mk(reps, len(reps)).sort_values("timestamp").to_csv(os.path.join(ROOT, "data", "sample_tweets.csv"), index=False)

n = 80
types = rng.choice(["loan", "bond", "derivative", "equity"], n, p=[.35, .3, .15, .2])
tx = pd.DataFrame({
 "txn_id": [f"T{i:03d}" for i in range(n)],
 "counterparty": rng.choice(["Acme Corp", "Globex", "Initech", "Umbrella", "Hooli", "Stark Ind", "Wayne Ent"], n),
 "asset_type": types,
 "sector": rng.choice(["Banks", "Energy", "Tech", "Retail", "Industrials", "Utilities"], n),
 "rating": rng.choice(["AAA", "AA", "A", "BBB", "BB", "B"], n, p=[.1, .2, .25, .25, .12, .08]),
 "notional": rng.integers(1, 50, n) * 1_000_000,
 "rate": np.round(rng.uniform(0.02, 0.09, n), 4),
 "duration": np.round(rng.uniform(1, 9, n), 1),
 "delta": np.round(rng.uniform(0.3, 0.9, n), 2),
})
tx.to_csv(os.path.join(ROOT, "data", "sample_transactions.csv"), index=False)
print("sample data written")
