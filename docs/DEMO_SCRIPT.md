# 10-minute demo video script (screen recording, YouTube *Unlisted*)

Guideline flow: Intro ~30s, Setup/Run ~30s, End-to-end 2-3 min, Results & impact 30-60s. You have 10 min, so use the extra time for depth.

| Time | Show | Say (in your own words) |
|---|---|---|
| 0:00-0:45 | Slide 1-2 | Who you are, the problem (unstructured text is too slow to read), one-line solution: one NLP engine feeding two risk modules. |
| 0:45-1:30 | Terminal | Show README Quickstart: `pip install -r requirements.txt`, `python src/main.py`, then `streamlit run src/app.py`. Show the signals table printing. |
| 1:30-3:00 | Slide 3 + code `src/risk_engine/nlp.py` | Walk through architecture; explain entity detection, sentiment, event classification, impact formula; mention swappable FinBERT/VADER and zero-shot. |
| 3:00-4:00 | Dashboard tab "Try your own text" | Paste 3 headlines (positive earnings, geopolitical shock, a tweet with $cashtag). Show JSON: sentiment, event, impact, drivers. Explain tweets have lower source weight (0.7). |
| 4:00-4:30 | Tab "Engine & Signals" | Show counts, events by type, impact distribution, the table, and `output/signals.json`. |
| 4:30-6:30 | Tab "Module A" | Explain weights rule (sentiment x impact, smoothing, 2-15% caps). Move the alpha slider and show weights change; compare index vs benchmark. Be honest that offline prices are simulated. |
| 6:30-8:30 | Tab "Module B" | Pick a high-impact Geopolitical event; show shocks applied, before/after by asset class and sector; lower the threshold to show more triggers; compare to naive flat shock. |
| 8:30-9:15 | Terminal | `python src/main.py --eval` and `python tests/test_engine.py`: show checks pass; state eval set is small. |
| 9:15-10:00 | Slides 6-7 | Domain impact (risk, PMs, treasury), limitations, next steps (FinBERT default, calibration, streaming). |

Tips: record at 1080p, close private tabs, pre-run once so models/data are cached, test the link in an incognito window.
