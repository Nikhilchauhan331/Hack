# Likely jury questions and honest answers (read before the live pitch)

1. **Why VADER + a finance lexicon instead of FinBERT?** Default is zero-heavy-dependency so anyone can run it in one command.
   FinBERT is already wired in (`SENTIMENT_BACKEND=finbert`), same schema. I did not measure FinBERT's gain on my set; I would evaluate it on a labelled public set (e.g. Financial PhraseBank).
2. **How is the impact score defined? Is it validated?** Heuristic: event base severity + intensity words + 2*|sentiment|, times source reliability. It is *not* calibrated to realised returns; the `--backtest` hook (Spearman IC vs next-day return) is the path to calibrating it.
3. **How good is the classifier?** On my 41 hand-labelled texts: ~76% sentiment accuracy with the lexicon back-end; event rules matched all news rows, but I wrote both rules and labels, so it is a smoke test, not generalisation.
4. **Why are weights capped at 2-15%?** Prevents a noisy sentiment burst from concentrating the index; reflects typical index-construction caps.
5. **How do you avoid overreacting to one tweet?** Tweets have 0.7 source weight, impact-weighted sentiment, and EMA smoothing (decay) across days.
6. **How are the stress shocks justified?** Simplified, transparent: bonds via duration approximation (dP/P = -D*(dy+dspread)), loans via PD multiplier x LGD (45%), equities via price shock, derivatives via delta-equivalent exposure; scaled by impact/10. Not a regulatory (FRTB/CCAR) model.
7. **What about duplicates, spam and bots?** Not handled yet; next step is near-duplicate detection and source credibility weights.
8. **Data?** Synthetic + public (Yahoo Finance via yfinance, optional GDELT/NewsAPI/Kaggle). No client or proprietary data.
9. **How would this scale?** Replace CSV ingestion with a stream (Kafka), batch FinBERT on GPU, serve signals via the FastAPI/queue; modules already consume only the signals schema.
10. **Did you use AI?** Yes, AI assistance for scaffolding and docs (disclosed in README); I can explain every module.
