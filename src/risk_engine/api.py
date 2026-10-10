"""Simple API so downstream apps can consume signals.
Run:  uvicorn risk_engine.api:app --reload   -> http://127.0.0.1:8000/docs"""
import os
from typing import Literal
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
from .engine import load_signals, run, OUT_DIR
from .nlp import analyze_text

app = FastAPI(title="AI/NLP Risk Engine")


class Item(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)
    source: Literal["news", "twitter"] = "news"


@app.get("/signals")
def signals(
    company: str | None = None,
    min_impact: float = Query(default=0, ge=0, le=10),
):
    df = load_signals() if os.path.exists(os.path.join(OUT_DIR, "signals.csv")) else run()
    if company:
        df = df[df.company == company.upper()]
    df = df[df.impact_score >= min_impact]
    return df.assign(timestamp=df.timestamp.astype(str)).to_dict(orient="records")


@app.post("/analyze")
def analyze(item: Item):
    return analyze_text(item.text, item.source)
