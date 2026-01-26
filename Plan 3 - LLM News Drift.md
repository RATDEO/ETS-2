## Plan 3: Align LLM With News-Drift Method (4B Model)

Goal: align the LLM component with the paper’s methodology (headline sentiment -> short‑term drift)
and reduce the LLM’s burden of generating full 30‑day price paths.

### Step 1: Prompt + labeling alignment (paper-style)
- Update the headline prompt to the paper format (YES/NO/UNKNOWN + 1‑sentence rationale).
- Set temperature=0 by default for reproducibility.
- Keep multi‑vote majority voting for reliability.
- Output goes to `data/news/headlines_labeled.csv`.

### Step 2: Data quality filters (EU/English + dedup)
- Add a filtering step that:
  - Keeps English headlines from EU sources.
  - Removes same‑day near‑duplicates (similarity > 0.6).
  - Optionally enforces keyword presence.
- Output goes to `data/news/headlines_filtered.csv`.

### Step 3: Daily sentiment series (full dataset range)
- Aggregate labeled headlines to daily sentiment (`YES=+1, UNKNOWN=0, NO=-1`).
- Output goes to `data/news/daily_sentiment.csv`.

### Step 4: News‑drift adjustment method
- Add a new method `TSM+NEWS-DRIFT`:
  - Uses daily sentiment score to nudge only short horizons (1–5 days).
  - Applies asymmetric weight (stronger for negative news).
  - Uses a decay schedule so the effect fades quickly.
  - Caps adjustment to avoid unrealistic jumps.

### Step 5: Benchmark with 4B model
- Run baseline TSM vs. `TSM+NEWS-DRIFT`.
- Compare MSE and trend accuracy across horizons.
- Document any gains at short horizons.

### Step 6: Final scale-up (later)
- Repeat Step 5 with a larger LLM once the 4B results stabilize.

### Possible Paper-Alignment Changes (Backlog)
1. Add strict relevance filtering (paper uses Ravenpack relevance=100) via stronger keyword rules or manual lists.
2. Exclude repetitive headlines across days (event similarity > 90 days), not just same-day dedup.
3. Separate overnight vs intraday headlines using release time cutoffs (before 9am / after 4pm).
4. Evaluate with drift/return metrics and long-short portfolio tests, not only MSE on price paths.
5. Treat the LLM as the primary signal (headline -> score) instead of blending into TSM.
6. Increase headline coverage (paper scale is much larger); expand sources and time range.
7. Map sentiment to targeted instruments or entities (paper uses company-level prompts).

---
Operational sequence (for scripts):
1) `scripts/fetch_news_gdelt.py` -> `data/news/headlines_raw.csv`
2) `scripts/filter_news_headlines.py` -> `data/news/headlines_filtered.csv`
3) `scripts/label_news_sentiment.py` -> `data/news/headlines_labeled.csv`
4) `scripts/build_daily_sentiment.py` -> `data/news/daily_sentiment.csv`
5) `python src/run_experiment.py --config src/config/tuned_llm_news_drift_gemma.yaml`
