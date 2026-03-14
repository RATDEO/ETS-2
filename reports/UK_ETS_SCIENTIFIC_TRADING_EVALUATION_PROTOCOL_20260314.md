# UK ETS Scientific Trading Evaluation Protocol

Generated: 2026-03-14

## Executive Summary

If the question is "how do we properly test the real-world trading performance of this system?", the answer is:

1. Freeze the strategy now.
2. Demote all previously touched historical periods to research only.
3. Build a point-in-time tradable UKA futures dataset with explicit execution assumptions.
4. Run a leakage-safe, nested, rolling historical study for calibration only.
5. Use prospective paper trading as the first confirmatory test.
6. Only after a successful paper-trading phase should the system be allowed into a small live pilot.

Anything weaker than that is still useful research, but it is not a scientific real-world trading test.

## 1. Why the Current Evaluation Is Not Yet Sufficient

The current repo has already fixed several serious leakage paths in the forecasting pipeline and has improved the trading reporting materially. That is real progress. However, the current trading evidence is still not sufficient for a scientific claim of deployable profitability.

The main reasons are:

- The current trading metrics are based on stylized non-overlapping horizon backtests and offset-averaged partitions, not on a broker- or exchange-realistic execution simulator. See [return_metrics.py](/Users/davidwilkinson/Desktop/ETS%202/src/eval/return_metrics.py) and [report_uk_financial_outcomes.py](/Users/davidwilkinson/Desktop/ETS%202/uk_ets/scripts/report_uk_financial_outcomes.py).
- The rolling-origin check improved the amount of evidence, but even there the economic edge was not robust enough to call conclusive. See [summary.md](/Users/davidwilkinson/Desktop/ETS%202/reports/uk_ets_llm_4b_rolling_origin_forecast/20260312_021028/summary.md).
- The 2025H2-2026Q1 holdout and several earlier periods have been reused heavily for model design. That historical evidence is now exploratory, not confirmatory.
- The current backtests still rely on a continuous price series and simplified transaction cost assumptions, whereas real trading would occur in specific ICE UKA futures contracts with specific roll, fee, spread, and latency mechanics.

Therefore, the correct scientific stance is:

- current results are promising research evidence for forecast correction
- they are not yet definitive real-world trading evidence

## 2. Scientific Objective

The objective is not merely to show that the system has a positive backtest.

The objective is to test the following confirmatory claim:

> A predeclared UKA futures trading strategy driven by the frozen forecast system generates positive net economic value, after realistic costs and execution assumptions, out of sample and prospectively, relative to predeclared benchmark strategies.

That claim breaks into three hypotheses:

- `H1 Forecast`: the frozen candidate improves predictive loss relative to benchmark on untouched future data.
- `H2 Economic`: the frozen candidate improves cost-adjusted trading performance relative to benchmark on untouched future data.
- `H3 Robustness`: the economic result survives reasonable execution-cost, roll, and liquidity stress scenarios.

The key principle is that `H2` is primary. Forecast MSE is supportive evidence, not the main endpoint.

## 3. Research Principles

The protocol should follow these principles.

### 3.1 Freeze Before Test

No scientific test exists without a frozen system.

The frozen object must include:

- model architecture
- prompt contract
- tool stack
- feature set
- example selection policy
- horizon under test
- position-sizing rule
- execution rule
- roll rule
- cost model
- benchmark definitions

Any change to one of those items creates a new strategy and resets the confirmatory clock.

### 3.2 Previously Touched History Is Not Confirmatory

This is the single most important point.

Because the current project has already run many iterations on the historical UK sample, the periods used to select prompts, tools, and configurations can no longer be treated as unbiased final test evidence.

That means:

- all historical runs up to the freeze date are research/development evidence
- the first confirmatory evidence begins only after the freeze date

This is not optional. It is the only scientifically defensible position.

### 3.3 Point-in-Time Data Only

Every feature, benchmark, signal, example, and execution input must exist in the exact form and timestamp in which it would have been available at the trade decision time.

### 3.4 Tradability Over Proxy Performance

The target of evaluation is not the continuous synthetic UKA price series. The target is the actual tradable UKA futures contract path and the achievable execution path on ICE.

### 3.5 LLM Training-Cutoff Interpretation

For the LLM component, there is an additional interpretive note that should be carried through every serious evaluation report.

The working assumption in this project is that the model's pretraining cutoff is roughly around the start of 2025. Therefore:

- if a prediction set lies entirely on or after `2025-01-01`, the risk of direct training-period contamination is materially lower
- if a prediction set lies before `2025-01-01`, contamination cannot be ruled out confidently
- if a prediction set crosses `2025-01-01`, results should be split or at least annotated as mixed

This note does not replace leakage control. It sits on top of it.

Even a post-2025 prediction set can still leak if the workflow is sloppy. But a pre-2025 prediction set carries an extra ambiguity that must be stated explicitly when interpreting LLM results.

## 4. Data Requirements

The current forecasting data is enough for research. It is not enough for a bulletproof real-world trading study.

The trading protocol should require the following data.

### 4.1 Tradable Market Data

For ICE UKA futures, the official contract is a deliverable futures contract on 1,000 UK allowances, quoted in GBP per tonne. See ICE official product specifications: [UKA Futures](https://www.ice.com/products/80216150) and [UKA UK Auction](https://www.ice.com/products/80216146/UKA-UK-Auction).

Required market data:

- point-in-time contract chain, not only a continuous back-adjusted series
- open, high, low, close, settlement for each listed contract
- preferably bid, ask, spread, volume, open interest
- contract last-trading-day and delivery schedule
- roll dates or all data needed to implement a predeclared roll rule
- holiday and shortened-session calendars

Minimum acceptable historical dataset:

- daily contract-level OHLCV plus open interest for each listed UKA contract

Preferred dataset:

- intraday bars or top-of-book quotes so next-session execution can be modeled credibly

### 4.2 Point-in-Time Exogenous Data

All auxiliary features must be timestamped as-of:

- UKA auction calendar and auction outcomes
- any ICAP/secondary market prints
- any macro or energy series used as exogenous variables
- any LLM tool inputs

Rules:

- no revised data unless the first-release snapshot is preserved
- no global backfilling
- no forward-fill across a timestamp boundary that would not have been known at decision time

### 4.3 Point-in-Time LLM Example Bank

For each historical example used in LLM prompting, store:

- example decision timestamp
- example forecast horizon
- example truth availability timestamp
- example lesson text creation timestamp

At decision time `t`, only examples whose truths would already have been fully known by `t` may be retrieved.

This is especially important because teaching-example leakage was already discovered earlier in the project. See [PROJECT_EVALUATION_AND_FUTURE_DIRECTIONS.md](/Users/davidwilkinson/Desktop/ETS%202/docs/PROJECT_EVALUATION_AND_FUTURE_DIRECTIONS.md) and [OPTION4_REPRODUCTION_AND_PRODUCTION_RUNBOOK.md](/Users/davidwilkinson/Desktop/ETS%202/docs/OPTION4_REPRODUCTION_AND_PRODUCTION_RUNBOOK.md).

## 5. Trading Rule Must Be Predeclared

The system cannot be judged scientifically if the trading rule is chosen after inspecting the data.

The protocol should predeclare:

- one primary model
- one primary horizon
- one primary signal-to-position mapping
- one primary execution convention

### 5.1 Recommended Primary Strategy

Given the current results, the most defensible primary choice is:

- instrument: ICE UKA futures
- model: one frozen LLM-corrected TSM path
- benchmark 1: base TSM
- benchmark 2: linear ridge
- benchmark 3: sign-only persistence or flat/no-trade benchmark
- primary horizon: either `h20` or `h30`, but only one should be primary
- signal: sign of predicted cumulative return at the primary horizon
- sizing: fixed unit size or predeclared volatility-targeted sizing based only on trailing realized volatility

Do not choose the best horizon after the test. Predeclare one.

### 5.2 Signal-to-Position Mapping

Use a simple mapping:

- long if predicted cumulative return > threshold
- short if predicted cumulative return < -threshold
- flat otherwise

Threshold must be fixed ex ante. If threshold tuning is allowed, it must happen only inside inner validation folds.

### 5.3 Position Sizing

Two acceptable choices:

- fixed one-lot sign strategy
- volatility-targeted position sizing using only information available up to decision time

Do not use optimized dynamic sizing unless it is independently validated. It introduces another layer of overfitting.

## 6. Execution Protocol

This is where most informal backtests fail.

### 6.1 Decision Time and Fill Time

If the strategy uses end-of-day features including the same-day close or settlement, it cannot assume execution at that same close unless the signal is generated before that close using only pre-close information.

The safest convention is:

- compute signal after market close on day `t`
- execute on day `t+1` at a predeclared executable benchmark, such as opening auction, first 5-minute VWAP, or next-session settlement proxy

If only daily data is available historically, the study should use the most conservative feasible assumption and treat that as approximate.

### 6.2 Roll Rule

A continuous back-adjusted series is not enough.

The study must predeclare a contract roll rule such as:

- roll X trading days before last trading day, or
- roll when next contract open interest exceeds current contract open interest, subject to a minimum notice period

The roll rule must be identical in backtest, paper trading, and live trading.

### 6.3 Capacity and Participation Constraints

To be scientifically credible, the strategy must obey participation caps:

- maximum fraction of daily volume
- maximum fraction of open interest
- minimum liquidity threshold for trade entry

If the strategy would exceed those limits, the trade is either scaled down or rejected.

## 7. Leakage-Safe Historical Experimental Design

Historical testing should be nested and purged.

### 7.1 Split Structure

Use an expanding-window outer loop:

- outer train
- outer validation
- outer test

The outer test blocks must be disjoint chronological blocks.

### 7.2 Purge and Embargo

Because labels and positions overlap through the forecast horizon, splits must be purged and embargoed.

The purge/embargo length should be at least:

- the maximum forecast horizon in trading days
- plus any additional days implied by execution lag and holding overlap

That principle is aligned with the logic behind purged and embargoed cross-validation in financial ML and with the leakage protections already implemented in the repo.

### 7.3 Inner Tuning Only

All selection must happen inside the outer training/validation data:

- prompt variants
- thresholds
- tool settings
- position-size parameters
- cost assumptions for optimization

The outer test block is never used for any choice.

### 7.4 Research vs Confirmatory Historical Role

The historical nested study serves two purposes:

- estimate how unstable performance is
- decide whether the frozen system is worth a forward paper-trading test

It does not create final production proof once the same sample has been used repeatedly for research.

## 8. Statistical Testing Plan

This is where scientific quality lives or dies.

### 8.1 Primary Economic Endpoint

Primary endpoint:

- cost-adjusted daily mark-to-market PnL series of the frozen strategy over the confirmatory evaluation period

The reason to use daily marked PnL rather than only non-overlapping trade outcomes is sample size. Trade counts at `h20/h30` are too small on a single holdout to support strong inference by themselves.

### 8.2 Trade-Level Secondary Endpoint

Also report:

- non-overlapping trade returns
- number of trades
- win rate
- average trade return
- downside-tail trade statistics

These are secondary because they are low-N.

### 8.3 Performance Metrics

Primary:

- net Sharpe ratio
- net mean daily return
- net max drawdown

Secondary:

- total return
- annualized return
- Calmar ratio
- Sortino ratio
- turnover
- participation rate
- exposure decomposition
- hit rate
- tail loss metrics such as CVaR

Forecast-support metrics:

- path MSE
- horizon MSE
- direction accuracy

### 8.4 Inference

Use dependence-aware methods.

Recommended:

- stationary or moving-block bootstrap for confidence intervals on daily PnL metrics
- Diebold-Mariano only for forecast loss comparisons, not as the main trading-profit test. See [Comparing Predictive Accuracy](https://www.nber.org/papers/t0169) and Diebold’s later cautionary note [Twenty Years Later](https://www.nber.org/papers/w18391).
- Hansen’s SPA or a stepwise SPA-style procedure to adjust for data snooping across many candidate strategies. See the discussion of Reality Check and SPA in [this overview](https://link.springer.com/article/10.1007/s11408-023-00433-2).
- White-style reality-check logic for the full searched model family, again to account for extensive specification search. Same source above.
- Model Confidence Set when comparing a family of surviving strategies, not just one pairwise winner. See [The Model Confidence Set](https://www.econometricsociety.org/publications/econometrica/2011/03/01/model-confidence-set).
- Deflated Sharpe Ratio to correct for multiple testing and non-normal returns. See [Bailey and López de Prado (2014)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551) and [journal version](https://www.pm-research.com/content/iijpormgmt/40/5/94).
- Probability of Backtest Overfitting to quantify whether the observed selected strategy is likely a search artifact. See [Bailey et al. (2015)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253).

### 8.5 Multiple-Testing Discipline

The system has already explored many prompts, tools, and strategy variants. Therefore the final statistical analysis must not behave as if only one strategy was tried.

The model family submitted to SPA/Reality-Check/DSR should include:

- all serious candidate variants evaluated during the locked research phase
- not only the winner

This is non-negotiable if the goal is scientific credibility.

## 9. Transaction Costs and Market Impact

The current fixed-bps assumptions are useful but not enough.

The protocol should include three cost layers.

### 9.1 Hard Fees

- exchange fees
- clearing fees
- broker commissions

### 9.2 Spread and Slippage

At minimum:

- half-spread cost for entry
- half-spread cost for exit
- additional slippage buffer for less liquid days

If quotes are unavailable historically, estimate spread conservatively from:

- official bid/ask data if obtainable
- or a documented proxy by volume/open-interest regime

### 9.3 Market Impact

For any strategy above trivial size, include an impact model.

A simple first pass can use:

- linear or square-root impact as a function of participation
- stress scenarios at 1x / 2x / 3x assumed impact

For orientation on why real trading costs matter and why they must be modeled from actual market conditions, see [Frazzini, Israel, and Moskowitz](https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID2294498_code753937.pdf?abstractid=2294498&mirid=1).

## 10. Minimum Evidence Threshold for a Claim

The project should set a high bar in advance.

A strategy should only be called a real trading improvement if all conditions hold:

- positive net Sharpe on the confirmatory period
- positive net return after all declared costs
- better than both base TSM and ridge on the primary economic metric
- survives SPA/Reality-Check-style multiple-testing adjustment at the prespecified level
- positive Deflated Sharpe Ratio assessment
- acceptable drawdown and turnover profile
- survives at least one harsher cost stress
- no evidence of leakage from a full code/data audit

And for stronger claims:

- survives a prospective paper-trading phase
- survives a small live pilot

## 11. Prospective Confirmation Plan

This is the scientific core.

### Phase A: Research Freeze

Immediately:

- select one frozen candidate
- select one frozen benchmark set
- select one frozen trading rule
- archive hashes of code, config, prompts, and datasets
- register the experiment plan in a dated markdown or YAML manifest

### Phase B: Prospective Paper Trading

Duration:

- minimum 6 months
- preferred 9 to 12 months

Protocol:

- generate signals every trading day in real time
- record timestamp of data snapshot, model output, and intended order
- simulate execution using next-session tradable benchmarks
- never rewrite signals after the fact

Output:

- immutable daily signal log
- immutable hypothetical order log
- daily marked PnL under the declared cost model

This is the first truly confirmatory economic test.

### Phase C: Micro-Capital Live Pilot

Only if Phase B is successful.

Protocol:

- very small size
- same exact strategy as paper trading
- no parameter changes
- log realized fills, slippage, rejects, and operational failures

This tests execution realism, not just forecast quality.

### Phase D: Scale Decision

Only after:

- paper trading is positive
- live pilot is positive
- realized costs do not destroy the edge

## 12. What the Repo Should Implement Next

If the goal is to move from research to scientific trading evaluation, the next engineering tasks should be:

1. Build a point-in-time UKA contract-chain store.
2. Implement explicit roll logic and contract-aware backtesting.
3. Add decision-time and fill-time timestamps to every prediction artifact.
4. Add a pre-registration manifest for frozen strategy definitions.
5. Add a purged, embargoed, nested rolling evaluator for trading strategies, not just forecast MSE.
6. Add bootstrap-based confidence intervals and SPA/Reality-Check-style multiple-testing evaluation.
7. Add daily marked PnL and execution logs for paper trading.
8. Add cost stress testing and capacity limits.

## 13. Bottom Line

The scientifically correct way to test real-world trading performance is not to squeeze one more historical holdout.

It is to:

- freeze the system
- treat prior backtests as exploratory
- use point-in-time tradable data
- enforce purge/embargo and inner-only selection
- evaluate with dependence-aware and multiple-testing-aware inference
- and then confirm prospectively through paper trading before any live capital is risked

That is the standard required if the project wants to claim genuine tradable performance rather than interesting historical research.

## References

- ICE UKA Futures contract specifications: https://www.ice.com/products/80216150
- ICE UKA Auction contract specifications: https://www.ice.com/products/80216146/UKA-UK-Auction
- Diebold, F.X., Mariano, R.S. "Comparing Predictive Accuracy": https://www.nber.org/papers/t0169
- Diebold, F.X. "Comparing Predictive Accuracy, Twenty Years Later": https://www.nber.org/papers/w18391
- Hansen, P.R., Lunde, A., Nason, J.M. "The Model Confidence Set": https://www.econometricsociety.org/publications/econometrica/2011/03/01/model-confidence-set
- Bailey, D.H., López de Prado, M. "The Deflated Sharpe Ratio": https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
- Bailey, D.H., Borwein, J., López de Prado, M., Zhu, Q.J. "The Probability of Backtest Overfitting": https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253
- Arian, H.R., Norouzi Mobarekeh, D., Seco, L. "Backtest Overfitting in the Machine Learning Era": https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4778909
- Discussion of White Reality Check / Hansen SPA / stepwise SPA in a recent survey article: https://link.springer.com/article/10.1007/s11408-023-00433-2
- Frazzini, A., Israel, R., Moskowitz, T.J. "Trading Costs of Asset Pricing Anomalies": https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID2294498_code753937.pdf?abstractid=2294498&mirid=1
