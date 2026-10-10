"""NLP core: entity detection, sentiment, event classification, impact scoring.

Pluggable back-ends (choose with environment variables):
  SENTIMENT_BACKEND = auto | finbert | vader | lexicon     (default: vader)
  EVENT_BACKEND     = keyword | zeroshot | hybrid           (default: keyword)
'auto' is an offline-safe alias for VADER; FinBERT is opt-in because loading it
may download a model from Hugging Face.
Every back-end returns the same schema, so the rest of the pipeline never changes.
"""
import math
import os
import re
from functools import lru_cache

SENTIMENT_BACKEND = os.getenv("SENTIMENT_BACKEND", "vader").lower()
EVENT_BACKEND = os.getenv("EVENT_BACKEND", "keyword").lower()

# ------------------------------------------------------------------ lexicon
# Finance-specific lexicon (VADER-style scale roughly -4..+4)
FIN_LEX = {
    "beats": 2.2, "beat": 2.0, "surge": 2.5, "surges": 2.5, "soar": 2.6, "soars": 2.6,
    "rally": 2.2, "record": 1.8, "upgrade": 2.0, "upgraded": 2.0, "growth": 1.6,
    "profit": 1.6, "strong": 1.8, "gains": 1.8, "gain": 1.6, "bullish": 2.4,
    "outperform": 2.0, "wins": 1.8, "approval": 1.6, "approved": 1.6, "boost": 1.8,
    "expands": 1.4, "breakthrough": 2.2, "dividend": 1.2, "optimism": 1.8, "optimistic": 1.8,
    "misses": -2.2, "miss": -2.0, "plunge": -2.8, "plunges": -2.8, "crash": -3.0,
    "slump": -2.4, "downgrade": -2.2, "downgraded": -2.2, "loss": -1.8, "losses": -2.0,
    "default": -3.0, "bankruptcy": -3.4, "lawsuit": -1.8, "probe": -1.8, "fine": -1.4,
    "fraud": -3.2, "weak": -1.8, "bearish": -2.4, "recall": -2.0, "layoffs": -2.0,
    "cuts": -1.4, "war": -2.4, "sanctions": -2.0, "tensions": -1.8, "crisis": -2.8,
    "recession": -2.6, "inflation": -1.4, "fears": -1.8, "warns": -1.8, "warning": -1.6,
    "selloff": -2.4, "volatility": -1.2, "collapse": -3.2, "missile": -2.4, "falls": -1.8,
    "scam": -3.0, "disaster": -3.0, "shock": -2.0, "conflict": -2.0, "outage": -1.8,
    "slumps": -2.4, "tumbles": -2.6, "jumps": 2.2, "climbs": 1.8, "raises": 1.4,
    "crushing": 2.0, "amazing": 2.0, "mounting": -1.4, "escalates": -2.2, "escalate": -2.2,
}
NEGATORS = {"not", "no", "never", "without", "fails", "failed"}


def _lexicon_score(text: str):
    toks = re.findall(r"[a-z']+", text.lower())
    s, drivers = 0.0, []
    for i, t in enumerate(toks):
        v = FIN_LEX.get(t)
        if v is None:
            continue
        if i > 0 and toks[i - 1] in NEGATORS:
            v = -0.74 * v
        s += v
        drivers.append(t)
    return round(s / math.sqrt(s * s + 15), 4), drivers  # VADER-style normalisation


# ------------------------------------------------------------------ back-ends
@lru_cache(maxsize=1)
def _load_vader():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        v = SentimentIntensityAnalyzer()
        v.lexicon.update(FIN_LEX)
        return v
    except Exception:
        return None


@lru_cache(maxsize=1)
def _load_finbert():
    try:
        from transformers import pipeline
        return pipeline("text-classification", model="ProsusAI/finbert", top_k=None, truncation=True)
    except Exception as e:
        print(f"[nlp] FinBERT unavailable ({type(e).__name__}); falling back")
        return None


@lru_cache(maxsize=1)
def _load_zeroshot():
    try:
        from transformers import pipeline
        return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    except Exception as e:
        print(f"[nlp] zero-shot model unavailable ({type(e).__name__}); using keywords")
        return None


@lru_cache(maxsize=4096)
def sentiment_score(text: str):
    """Return (sentiment in [-1,1], backend_name, driver_words)."""
    mode = SENTIMENT_BACKEND
    if mode == "finbert":
        fb = _load_finbert()
        if fb is not None:
            out = {d["label"].lower(): d["score"] for d in fb(text[:1500])[0]}
            return round(out.get("positive", 0) - out.get("negative", 0), 4), "finbert", []
    if mode in ("auto", "vader"):
        vd = _load_vader()
        if vd is not None:
            return round(vd.polarity_scores(text)["compound"], 4), "vader+finlex", _lexicon_score(text)[1]
    s, drv = _lexicon_score(text)
    return s, "lexicon", drv


# ------------------------------------------------------------------ entities
COMPANIES = {
    "AAPL": ["apple", "iphone", "aapl"], "MSFT": ["microsoft", "azure", "msft"],
    "AMZN": ["amazon", "aws", "amzn"], "GOOGL": ["google", "alphabet", "googl"],
    "META": ["meta platforms", "facebook", "instagram"],
    "NVDA": ["nvidia", "nvda"], "TSLA": ["tesla", "tsla"],
    "JPM": ["jpmorgan", "jp morgan", "jpm"], "BAC": ["bank of america", "bofa"],
    "GS": ["goldman sachs", "goldman"], "XOM": ["exxon", "exxonmobil"],
    "JNJ": ["johnson & johnson", "johnson and johnson", "j&j"], "PFE": ["pfizer"],
    "WMT": ["walmart"], "KO": ["coca-cola", "coca cola"], "DIS": ["disney"],
    "NFLX": ["netflix"], "INTC": ["intel"], "CSCO": ["cisco"], "BA": ["boeing"],
}
_COMPANY_RE = {t: re.compile(r"(?<![\w])(" + "|".join(re.escape(a) for a in al) + r")(?![\w])", re.I)
               for t, al in COMPANIES.items()}
_CASHTAG = re.compile(r"\$([A-Z]{1,5})\b")


def detect_companies(text: str):
    found = {t for t, rx in _COMPANY_RE.items() if rx.search(text)}
    found |= {m for m in _CASHTAG.findall(text) if m in COMPANIES}
    return sorted(found) or ["MARKET"]


# ------------------------------------------------------------------ events
EVENT_KEYWORDS = {
    "Geopolitical": ["war", "sanction", "sanctions", "conflict", "tariff", "tariffs", "missile",
                     "invasion", "embargo", "tensions", "ceasefire", "trade war", "military"],
    "Macroeconomic": ["inflation", "interest rate", "interest rates", "rate hike", "rate cut",
                      "fed", "federal reserve", "gdp", "unemployment", "recession", "cpi",
                      "central bank", "jobs report"],
    "Credit Event": ["default", "downgrade", "downgraded", "bankruptcy", "credit rating",
                     "insolvency", "loan losses", "write-off", "debt", "liquidity crisis"],
    "Merger/Acquisition": ["acquire", "acquires", "acquisition", "merger", "buyout", "takeover",
                           "to buy", "stake"],
    "Product Launch": ["launch", "launches", "unveil", "unveils", "release", "rollout", "introduces"],
    "Earnings": ["earnings", "revenue", "profit", "quarterly", "guidance", "eps"],
    "Regulatory/Legal": ["lawsuit", "antitrust", "probe", "investigation", "regulator", "sec",
                         "fined", "recall"],
}
EVENT_LABELS = list(EVENT_KEYWORDS) + ["Other"]
_EVENT_RE = {e: re.compile(r"\b(" + "|".join(re.escape(k) for k in kws) + r")\b", re.I)
             for e, kws in EVENT_KEYWORDS.items()}

BASE_IMPACT = {"Geopolitical": 6.0, "Macroeconomic": 5.5, "Credit Event": 6.5,
               "Merger/Acquisition": 5.0, "Product Launch": 3.0, "Earnings": 4.0,
               "Regulatory/Legal": 5.0, "Other": 2.0}
INTENSITY = re.compile(r"\b(plunge\w*|crash\w*|collapse\w*|soar\w*|surge\w*|massive|record|shock\w*|"
                       r"emergency|crisis|major|sharply|historic|worst|biggest|escalat\w*)\b", re.I)
SOURCE_WEIGHT = {"news": 1.0, "twitter": 0.7}


def _keyword_event(text: str):
    hits = {e: _EVENT_RE[e].findall(text) for e in _EVENT_RE}
    best = max(hits, key=lambda e: len(hits[e]))
    if not hits[best]:
        return "Other", 0.3, []
    total = sum(len(v) for v in hits.values())
    return best, round(len(hits[best]) / total, 2), sorted({h.lower() for h in hits[best]})


def classify_event(text: str):
    """Return (event_type, confidence, matched_keywords)."""
    label, conf, kws = _keyword_event(text)
    if EVENT_BACKEND == "zeroshot" or (EVENT_BACKEND == "hybrid" and conf < 0.6):
        zs = _load_zeroshot()
        if zs is not None:
            r = zs(text, candidate_labels=EVENT_LABELS)
            return r["labels"][0], round(r["scores"][0], 2), kws
    return label, conf, kws


def impact_score(event: str, sentiment: float, text: str, source: str = "news") -> float:
    """Impact = (event base severity + intensity words + 2*|sentiment|) * source reliability, clipped 1-10."""
    base = BASE_IMPACT.get(event, 2.0)
    intensity = min(len(INTENSITY.findall(text)) * 1.0, 2.5)
    raw = (base + intensity + 2.0 * abs(sentiment)) * SOURCE_WEIGHT.get(source, 0.8)
    return round(max(1.0, min(10.0, raw)), 1)


def analyze_text(text: str, source: str = "news"):
    """Return list of structured signals (one per company mentioned)."""
    s, backend, drivers = sentiment_score(text)
    ev, conf, kws = classify_event(text)
    imp = impact_score(ev, s, text, source)
    return [{"company": c, "sentiment_score": s, "event_type": ev, "impact_score": imp,
             "event_confidence": conf, "sentiment_backend": backend,
             "drivers": ", ".join(sorted(set(drivers) | set(kws)))} for c in detect_companies(text)]
