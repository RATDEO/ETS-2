# W0-W4 Legacy Source Benchmark Plan

## Objective
- Start from the corrected UK-specific live stack and improved Huber base.
- Reintroduce legacy `/Data` source groups from the older pipeline in controlled bundles.
- Test whether any of those legacy source groups improve current `W0-W4` performance.

## Method
- Build merged data roots that keep `uk_ets/Data_auto_uk` as the base and overlay legacy directories only where needed.
- Run a broad base-only sweep first.
- Then run the current best live refiner on the best few source bundles.

## Candidates
- `baseline_energy16` (current): Current improved UK-specific base without legacy-only source groups.
  Expected outcome: Anchor candidate.
- `legacy_indices5` (legacy_indices): Add legacy carbon-market indices, including KSET, and expose them to the base model.
  Expected outcome: Possible small help on W3/W4 if global carbon beta matters.
- `legacy_vstoxx` (legacy_all): Turn VSTOXX back on as a legacy volatility proxy.
  Expected outcome: Possible small help if hard windows are volatility-regime driven.
- `legacy_icap_secondary` (current): Expose raw ICAP secondary and spread features back into the base.
  Expected outcome: Possible help in auction/microstructure-heavy windows.
- `legacy_auction_volume` (legacy_auctions): Overlay the old auction files and reintroduce auction volume/lag features.
  Expected outcome: Possible help in W4 if early auction regime is better represented by legacy files.
- `legacy_market_compact20` (legacy_all): Compact mixed legacy stack: one index, VSTOXX, richer ICAP, and auction volume on top of the improved base.
  Expected outcome: Best chance of a robust mixed-source gain without blowing up dimensionality.
- `legacy_market_stack_all` (legacy_all): Throw the main legacy market groups in at once with a wider feature budget.
  Expected outcome: Highest variance candidate. Could help if source omission is the core problem, could also dilute badly.
