import numpy as np

from src.eval.return_metrics import (
    assess_llm_training_cutoff,
    compute_daily_mtm_portfolio_metrics,
    compute_long_short_portfolio_metrics,
    compute_primary_horizon_signals,
    evaluate_signal_threshold_candidates,
)


def test_long_short_metrics_include_non_overlapping_total_return_for_longs():
    base_prices = np.array([100.0, 100.0, 100.0, 100.0])
    y_true_prices = np.array(
        [
            [105.0, 110.0],
            [105.0, 110.0],
            [105.0, 110.0],
            [105.0, 110.0],
        ]
    )
    y_pred_prices = np.array(
        [
            [104.0, 108.0],
            [104.0, 108.0],
            [104.0, 108.0],
            [104.0, 108.0],
        ]
    )

    metrics = compute_long_short_portfolio_metrics(
        y_true_prices=y_true_prices,
        y_pred_prices=y_pred_prices,
        base_prices=base_prices,
        horizons=[2],
    )

    row = metrics.loc[2]
    assert row["n_trades_non_overlap"] == 2
    assert np.isclose(row["total_return_non_overlap"], 0.21, atol=1e-8)
    assert np.isclose(row["max_drawdown_non_overlap"], 0.0, atol=1e-10)
    assert np.isclose(row["hit_rate_non_overlap"], 1.0, atol=1e-10)


def test_long_short_metrics_handle_non_overlapping_short_compounding():
    base_prices = np.array([100.0, 100.0, 100.0, 100.0])
    y_true_prices = np.array(
        [
            [95.0, 90.0],
            [95.0, 90.0],
            [95.0, 90.0],
            [95.0, 90.0],
        ]
    )
    y_pred_prices = np.array(
        [
            [96.0, 92.0],
            [96.0, 92.0],
            [96.0, 92.0],
            [96.0, 92.0],
        ]
    )

    metrics = compute_long_short_portfolio_metrics(
        y_true_prices=y_true_prices,
        y_pred_prices=y_pred_prices,
        base_prices=base_prices,
        horizons=[2],
    )

    row = metrics.loc[2]
    single_short_return = (100.0 / 90.0) - 1.0
    expected_total_return = (1.0 + single_short_return) ** 2 - 1.0
    assert row["n_trades_non_overlap"] == 2
    assert np.isclose(row["total_return_non_overlap"], expected_total_return, atol=1e-8)
    assert np.isclose(row["hit_rate_non_overlap"], 1.0, atol=1e-10)
    assert row["annualized_return_non_overlap"] > 0.0


def test_long_short_metrics_include_offset_averaged_non_overlap_returns():
    base_prices = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    y_true_prices = np.array(
        [
            [100.0, 110.0],
            [100.0, 90.0],
            [100.0, 110.0],
            [100.0, 90.0],
            [100.0, 110.0],
        ]
    )
    y_pred_prices = np.array(
        [
            [100.0, 105.0],
            [100.0, 105.0],
            [100.0, 105.0],
            [100.0, 105.0],
            [100.0, 105.0],
        ]
    )

    metrics = compute_long_short_portfolio_metrics(
        y_true_prices=y_true_prices,
        y_pred_prices=y_pred_prices,
        base_prices=base_prices,
        horizons=[2],
        average_non_overlap_offsets=True,
    )

    row = metrics.loc[2]
    offset0_total = (1.10**3) - 1.0
    offset1_total = (0.90**2) - 1.0
    expected_avg = (offset0_total + offset1_total) / 2.0
    assert row["offset_count_non_overlap"] == 2
    assert np.isclose(row["total_return_non_overlap"], offset0_total, atol=1e-8)
    assert np.isclose(row["total_return_non_overlap_offset_avg"], expected_avg, atol=1e-8)
    assert np.isclose(row["total_return_non_overlap_offset_min"], offset1_total, atol=1e-8)
    assert np.isclose(row["total_return_non_overlap_offset_max"], offset0_total, atol=1e-8)


def test_long_short_metrics_include_cost_adjusted_non_overlap_returns():
    base_prices = np.array([100.0, 100.0, 100.0, 100.0])
    y_true_prices = np.array(
        [
            [105.0, 110.0],
            [105.0, 110.0],
            [105.0, 110.0],
            [105.0, 110.0],
        ]
    )
    y_pred_prices = np.array(
        [
            [104.0, 108.0],
            [104.0, 108.0],
            [104.0, 108.0],
            [104.0, 108.0],
        ]
    )

    metrics = compute_long_short_portfolio_metrics(
        y_true_prices=y_true_prices,
        y_pred_prices=y_pred_prices,
        base_prices=base_prices,
        horizons=[2],
        transaction_cost_bps=100.0,
    )

    row = metrics.loc[2]
    net_single_trade = 1.10 * (0.99**2) - 1.0
    expected_total_net = (1.0 + net_single_trade) ** 2 - 1.0
    assert row["transaction_cost_bps"] == 100.0
    assert row["total_return_non_overlap_cost_adj"] < row["total_return_non_overlap"]
    assert np.isclose(row["total_return_non_overlap_cost_adj"], expected_total_net, atol=1e-8)
    assert row["annualized_return_non_overlap_cost_adj"] > 0.0


def test_compute_primary_horizon_signals_uses_horizon_price_move():
    base_prices = np.array([100.0, 100.0, 100.0])
    y_pred_prices = np.array(
        [
            [100.0, 101.0, 103.0],
            [100.0, 99.5, 98.0],
            [100.0, 100.1, 100.2],
        ]
    )

    signals = compute_primary_horizon_signals(
        y_pred_prices=y_pred_prices,
        base_prices=base_prices,
        horizon=3,
        signal_threshold=0.01,
    )

    assert signals.tolist() == [1, -1, 0]


def test_assess_llm_training_cutoff_flags_post_2025_prediction_sets():
    note = assess_llm_training_cutoff(["2025-01-02", "2025-01-03", "2025-02-01"])

    assert note["status"] == "post_cutoff_only"
    assert note["n_pre_cutoff"] == 0
    assert note["n_post_cutoff"] == 3
    assert "cleaner regime" in note["note"]


def test_compute_daily_mtm_portfolio_metrics_tracks_next_day_execution():
    dates = np.array(
        [
            "2025-01-01",
            "2025-01-02",
            "2025-01-03",
            "2025-01-06",
            "2025-01-07",
            "2025-01-08",
        ],
        dtype="datetime64[ns]",
    )
    close_prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    decision_dates = np.array(["2025-01-02"], dtype="datetime64[ns]")
    signals = np.array([1])

    daily_df, metrics = compute_daily_mtm_portfolio_metrics(
        close_prices=close_prices,
        dates=dates,
        decision_dates=decision_dates,
        signals=signals,
        hold_days=2,
        execution_lag_days=1,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
        max_concurrent_positions=2,
    )

    # Signal on 2025-01-02 enters at 2025-01-03 close, then accrues on
    # 2025-01-06 and 2025-01-07 close-to-close returns with half-capital weight.
    expected_returns = np.zeros(len(close_prices))
    expected_returns[3] = 0.5 * ((103.0 / 102.0) - 1.0)
    expected_returns[4] = 0.5 * ((104.0 / 103.0) - 1.0)

    assert np.allclose(daily_df["strategy_return"].to_numpy(), expected_returns, atol=1e-10)
    assert metrics["n_executed_trades"] == 1
    assert metrics["max_concurrent_positions"] == 2
    assert metrics["total_return"] > 0.0


def test_compute_daily_mtm_portfolio_metrics_can_trim_to_active_window():
    dates = np.array(
        [
            "2025-01-01",
            "2025-01-02",
            "2025-01-03",
            "2025-01-06",
            "2025-01-07",
            "2025-01-08",
        ],
        dtype="datetime64[ns]",
    )
    close_prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    decision_dates = np.array(["2025-01-02"], dtype="datetime64[ns]")
    signals = np.array([1])

    full_df, full_metrics = compute_daily_mtm_portfolio_metrics(
        close_prices=close_prices,
        dates=dates,
        decision_dates=decision_dates,
        signals=signals,
        hold_days=2,
        execution_lag_days=1,
        max_concurrent_positions=2,
        trim_to_active_window=False,
    )
    active_df, active_metrics = compute_daily_mtm_portfolio_metrics(
        close_prices=close_prices,
        dates=dates,
        decision_dates=decision_dates,
        signals=signals,
        hold_days=2,
        execution_lag_days=1,
        max_concurrent_positions=2,
        trim_to_active_window=True,
    )

    assert len(full_df) == 6
    assert len(active_df) == 3
    assert active_metrics["active_start_date"] == "2025-01-03"
    assert active_metrics["active_end_date"] == "2025-01-07"
    assert active_metrics["daily_sharpe_annualized"] > full_metrics["daily_sharpe_annualized"]


def test_evaluate_signal_threshold_candidates_prefers_better_validation_gate():
    panel_dates = np.array(
        [
            "2025-01-01",
            "2025-01-02",
            "2025-01-03",
            "2025-01-06",
            "2025-01-07",
            "2025-01-08",
            "2025-01-09",
            "2025-01-10",
        ],
        dtype="datetime64[ns]",
    )
    panel_prices = np.array([100.0, 100.0, 101.0, 102.0, 101.0, 103.0, 104.0, 105.0])
    decision_dates = np.array(
        ["2025-01-02", "2025-01-03", "2025-01-06"],
        dtype="datetime64[ns]",
    )
    base_prices = np.array([100.0, 101.0, 102.0])
    y_pred_prices = np.array(
        [
            [100.0, 101.5],
            [101.0, 101.2],
            [102.0, 104.5],
        ]
    )

    summary, selected_threshold = evaluate_signal_threshold_candidates(
        y_pred_prices=y_pred_prices,
        base_prices=base_prices,
        decision_dates=decision_dates,
        panel_dates=panel_dates,
        panel_prices=panel_prices,
        horizon=2,
        threshold_candidates=[0.0, 0.02],
        execution_lag_days=1,
        hold_days=2,
        max_concurrent_positions=2,
        objective="daily_sharpe_annualized",
        min_executed_trades=1,
        trim_to_active_window=True,
    )

    assert np.isclose(selected_threshold, 0.02)
    assert summary["valid_for_selection"].sum() == 2
