"""Data ingestion from >= 2 sources: financial news and Twitter/X-style posts.
Each loader returns a DataFrame[timestamp, source, text].
Live sources (NewsAPI, GDELT) are optional; CSV loaders always work offline."""
import os
import pandas as pd
import requests


def load_news_csv(path):
    df = pd.read_csv(path)
    df["source"] = "news"
    return df[["timestamp", "source", "text"]]


def load_tweets_csv(path):
    df = pd.read_csv(path)
    df["source"] = "twitter"
    return df[["timestamp", "source", "text"]]


def fetch_newsapi(query="stock market OR earnings", api_key=None, page_size=50):
    """Live headlines from newsapi.org (free key). Set NEWSAPI_KEY env var."""
    api_key = api_key or os.getenv("NEWSAPI_KEY")
    if not api_key:
        raise RuntimeError("NEWSAPI_KEY not set")
    r = requests.get("https://newsapi.org/v2/everything", timeout=20,
                     params={"q": query, "language": "en", "sortBy": "publishedAt",
                             "pageSize": page_size, "apiKey": api_key})
    r.raise_for_status()
    rows = [{"timestamp": a["publishedAt"], "source": "news",
             "text": f"{a['title']}. {a.get('description') or ''}"} for a in r.json()["articles"]]
    return pd.DataFrame(rows)


def fetch_gdelt(query="economy OR markets", maxrecords=75):
    """Live headlines from the GDELT DOC 2.0 API (no key needed)."""
    r = requests.get("https://api.gdeltproject.org/api/v2/doc/doc", timeout=30,
                     params={"query": query, "mode": "artlist", "format": "json",
                             "maxrecords": maxrecords, "sort": "datedesc"})
    r.raise_for_status()
    rows = [{"timestamp": pd.to_datetime(a["seendate"], format="%Y%m%dT%H%M%SZ"),
             "source": "news", "text": a["title"]} for a in r.json().get("articles", [])]
    return pd.DataFrame(rows)
