import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from risk_engine.nlp import analyze_text, detect_companies, classify_event, impact_score
from modules.stress_test import apply_stress, load_portfolio, SCENARIOS
from risk_engine.engine import DATA_DIR


def test_sentiment_direction():
    pos = analyze_text("Apple beats earnings as profit surges to a record")[0]
    neg = analyze_text("Tesla plunges after fraud probe and massive recall")[0]
    assert pos["sentiment_score"] > 0.3 > -0.3 > neg["sentiment_score"]


def test_entities_and_cashtag():
    assert detect_companies("Nvidia and Goldman Sachs rally") == ["GS", "NVDA"]
    assert detect_companies("buying $AAPL today") == ["AAPL"]
    assert detect_companies("markets fall") == ["MARKET"]


def test_event_and_impact_bounds():
    assert classify_event("New sanctions as war escalates")[0] == "Geopolitical"
    assert 1.0 <= impact_score("Credit Event", -1, "crisis collapse default shock crash") <= 10.0


def test_high_impact_geopolitical_triggers_losses():
    pf = load_portfolio(os.path.join(DATA_DIR, "sample_transactions.csv"))
    _, summ, _ = apply_stress(pf, "Geopolitical", 9.0)
    assert summ.value_after.sum() < summ.value_before.sum()


def test_stress_scales_with_impact():
    pf = load_portfolio(os.path.join(DATA_DIR, "sample_transactions.csv"))
    lo = apply_stress(pf, "Geopolitical", 3.0)[1].pnl.sum()
    hi = apply_stress(pf, "Geopolitical", 9.0)[1].pnl.sum()
    assert hi < lo < 0


if __name__ == "__main__":  # run without pytest:  python tests/test_engine.py
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("PASS", n)
