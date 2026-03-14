# UK ETS Financial Outcomes

Generated: 2026-03-11T15:09:05.277660

Methodology:
- Non-overlapping backtests averaged across all valid start offsets for each horizon.
- Transaction costs assumed at `10.0` bps per side.
- Legacy offset-0, no-cost metrics are included where the archived run already stored them.

## Forecast Winner

- h5: path MSE `21.852871`, gross offset-avg return `57.42%`, net offset-avg return `48.60%`, net annualized `100.90%`, net Sharpe `2.876`, net max drawdown `7.24%`
- h20: path MSE `21.852871`, gross offset-avg return `54.86%`, net offset-avg return `52.65%`, net annualized `117.22%`, net Sharpe `2.384`, net max drawdown `5.88%`
- h30: path MSE `21.852871`, gross offset-avg return `63.35%`, net offset-avg return `61.77%`, net annualized `133.63%`, net Sharpe `2.662`, net max drawdown `1.71%`

## Trading Winner

- h5: path MSE `21.911751`, gross offset-avg return `58.00%`, net offset-avg return `49.15%`, net annualized `102.09%`, net Sharpe `2.905`, net max drawdown `8.03%`
- h20: path MSE `21.911751`, gross offset-avg return `55.55%`, net offset-avg return `53.32%`, net annualized `118.81%`, net Sharpe `2.543`, net max drawdown `5.38%`
- h30: path MSE `21.911751`, gross offset-avg return `64.43%`, net offset-avg return `62.84%`, net annualized `135.90%`, net Sharpe `2.749`, net max drawdown `1.39%`

## Base TSM Reference

- h5: path MSE `22.726353`, gross offset-avg return `56.08%`, net offset-avg return `47.33%`, net annualized `98.21%`, net Sharpe `2.807`, net max drawdown `8.03%`
- h20: path MSE `22.726353`, gross offset-avg return `54.20%`, net offset-avg return `51.99%`, net annualized `115.31%`, net Sharpe `2.351`, net max drawdown `5.88%`
- h30: path MSE `22.726353`, gross offset-avg return `61.00%`, net offset-avg return `59.45%`, net annualized `127.83%`, net Sharpe `2.581`, net max drawdown `2.02%`

## Notes

- Offset averaging matters most at long horizons; single-offset h30 returns were materially more fragile.
- Cost-adjusted returns remain illustrative; they do not include slippage beyond the fixed round-trip cost assumption.
