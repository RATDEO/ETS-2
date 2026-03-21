# UK SENT Transfer Plan

## Objective
- Apply the existing EU `SENT` pipeline to UK ETS, not a new sentiment family.
- Compare both prompt-side `CoT-SENT-RF` and sentiment-as-feature against the corrected UK base.

## Candidates
- `base_energy16`: Improved UK hard-regime base anchor with no sentiment.
  Expected outcome: Reference baseline.
- `sent_feature_raw`: Apply the EU Strategy-3 style sentiment-as-feature path using the UK raw daily sentiment series.
  Expected outcome: Expected impact: flat to mildly positive if UK news has any incremental statistical signal.
- `sent_feature_importance`: Sentiment-as-feature using the UK importance-weighted daily series.
  Expected outcome: Expected impact: slightly better than raw if headline relevance is noisy.
- `cot_sent_raw`: Apply the EU paper-style CoT-SENT-RF refiner to the UK improved base using the raw UK daily sentiment series.
  Expected outcome: Expected impact: likely small and regime-dependent; W0/W3 most likely to benefit if the UK corpus is usable.
- `cot_sent_importance`: Apply the EU paper-style CoT-SENT-RF refiner to the UK improved base using the importance-weighted UK daily sentiment series.
  Expected outcome: Expected impact: slightly higher upside than raw, but also more risk of over-thinning the already sparse UK news signal.
