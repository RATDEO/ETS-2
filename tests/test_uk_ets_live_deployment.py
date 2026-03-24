from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.uk_ets_live_deployment as live_deployment
from src.uk_ets_live_deployment import (
    _build_current_online_memory_context,
    _fit_live_learned_gate_bundle,
    _simulate_live_online_memory_history,
    build_backtest_seed_rows,
    backfill_archive_actuals,
    build_recent_live_history_rows,
    export_public_site_files,
    merge_archive_rows,
)
from src.run_experiment import (
    build_online_memory_gate_feature_frame,
    predict_online_memory_learned_gate_scores,
)


def _make_canonical_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "runs" / "20260323_120000_abcd12"
    (run_dir / "data").mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)

    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-03-17",
                    "2026-03-18",
                    "2026-03-19",
                    "2026-03-20",
                    "2026-03-24",
                    "2026-03-25",
                ]
            ),
            "y": [50.0, 51.0, 52.0, 53.0, 54.0, 55.0],
        }
    )
    panel.to_parquet(run_dir / "data" / "panel.parquet")

    config_payload = {
        "run_id": run_dir.name,
        "git_hash": "deadbeef",
        "timestamp": "2026-03-23T12:00:00+00:00",
        "target": {
            "instrument": "UKA_FUTURES",
            "currency": "GBP",
            "mode": "returns",
        },
        "time_series": {
            "pred_len": 3,
        },
    }
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(config_payload, sort_keys=False),
        encoding="utf-8",
    )

    np.savez_compressed(
        run_dir / "predictions" / "TSM+LLM-COT-RF-HDELTA_pred_test_subset.npz",
        dates=np.asarray(["2026-03-17"], dtype="U"),
        y_true=np.asarray([[51.0, 52.0, 53.0]], dtype=float),
        base_pred=np.asarray([[50.8, 51.7, 52.6]], dtype=float),
        yhat=np.asarray([[51.2, 52.3, 53.4]], dtype=float),
    )
    return run_dir


def _live_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_id": "live-run-1",
                "forecast_made_on": "2026-03-24",
                "latest_observed_date": "2026-03-24",
                "latest_observed_price": 54.0,
                "target_date": "2026-03-25",
                "step_index": 1,
                "actual": None,
                "base_tsm_forecast": 54.4,
                "llm_tsm_forecast": 54.8,
                "model_version": "uk_ets_llm_4b_canonical_default",
                "model_commit": "deadbeef",
                "data_version": "2026-03-24",
                "is_realized": False,
                "generated_at": "2026-03-24T19:30:00+00:00",
                "record_source": "main_live_run",
                "source_run_id": "live-run-1",
            },
            {
                "run_id": "live-run-1",
                "forecast_made_on": "2026-03-24",
                "latest_observed_date": "2026-03-24",
                "latest_observed_price": 54.0,
                "target_date": "2026-03-31",
                "step_index": 5,
                "actual": None,
                "base_tsm_forecast": 55.2,
                "llm_tsm_forecast": 55.9,
                "model_version": "uk_ets_llm_4b_canonical_default",
                "model_commit": "deadbeef",
                "data_version": "2026-03-24",
                "is_realized": False,
                "generated_at": "2026-03-24T19:30:00+00:00",
                "record_source": "main_live_run",
                "source_run_id": "live-run-1",
            },
        ]
    )


def test_build_backtest_seed_rows_uses_realized_panel_dates(tmp_path: Path) -> None:
    run_dir = _make_canonical_run_dir(tmp_path)

    archive = build_backtest_seed_rows(run_dir)

    assert len(archive) == 3
    assert archive["target_date"].tolist() == ["2026-03-18", "2026-03-19", "2026-03-20"]
    assert archive["latest_observed_date"].tolist() == ["2026-03-17"] * 3
    assert archive["actual"].tolist() == [51.0, 52.0, 53.0]
    assert archive["base_tsm_forecast"].tolist() == [50.8, 51.7, 52.6]
    assert archive["llm_tsm_forecast"].tolist() == [51.2, 52.3, 53.4]
    assert archive["is_realized"].tolist() == [True, True, True]


def test_build_backtest_seed_rows_falls_back_to_business_dates_when_panel_is_truncated(
    tmp_path: Path,
) -> None:
    run_dir = _make_canonical_run_dir(tmp_path)
    truncated_panel = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-03-17",
                    "2026-03-18",
                    "2026-03-19",
                ]
            ),
            "y": [50.0, 51.0, 52.0],
        }
    )
    truncated_panel.to_parquet(run_dir / "data" / "panel.parquet")

    archive = build_backtest_seed_rows(run_dir)

    assert archive["target_date"].tolist() == ["2026-03-18", "2026-03-19", "2026-03-20"]
    assert archive["actual"].tolist() == [51.0, 52.0, 53.0]


def test_merge_archive_rows_deduplicates_same_origin_and_step(tmp_path: Path) -> None:
    run_dir = _make_canonical_run_dir(tmp_path)
    archive = build_backtest_seed_rows(run_dir)
    live_rows = _live_rows()

    merged_once = merge_archive_rows(archive, live_rows)
    merged_twice = merge_archive_rows(merged_once, live_rows)

    assert len(merged_once) == len(merged_twice)
    latest = merged_twice[merged_twice["latest_observed_date"] == "2026-03-24"]
    assert latest["step_index"].tolist() == [1, 5]


def test_backfill_and_export_public_site_files(tmp_path: Path) -> None:
    run_dir = _make_canonical_run_dir(tmp_path)
    archive = build_backtest_seed_rows(run_dir)
    archive = merge_archive_rows(archive, _live_rows())

    panel = pd.read_parquet(run_dir / "data" / "panel.parquet")
    backfilled = backfill_archive_actuals(archive, panel)
    export_dir = tmp_path / "exports"
    export_public_site_files(backfilled, export_dir)

    latest = json.loads((export_dir / "latest.json").read_text(encoding="utf-8"))
    horizon_1d = json.loads((export_dir / "horizon_1d.json").read_text(encoding="utf-8"))
    horizon_5d = json.loads((export_dir / "horizon_5d.json").read_text(encoding="utf-8"))
    status = json.loads((export_dir / "status.json").read_text(encoding="utf-8"))

    assert latest["latest_observed_date"] == "2026-03-24"
    assert latest["forecasts"]["1d"]["llm_tsm_value"] == 54.8
    assert latest["forecasts"]["1d"]["predicted_change_abs"] == pytest.approx(0.8)
    assert latest["forecasts"]["5d"]["target_date"] == "2026-03-31"

    assert [row["target_date"] for row in horizon_1d["series"]] == ["2026-03-18", "2026-03-25"]
    assert horizon_1d["series"][-1]["actual"] == 55.0
    assert horizon_1d["series"][-1]["llm_tsm_forecast"] == 54.8
    assert horizon_5d["series"] == []
    assert status["horizon_counts"]["h1"] == 2
    assert status["horizon_counts"]["h5"] == 0


def test_build_recent_live_history_rows_fills_origins_after_cutoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-03-18",
                    "2026-03-19",
                    "2026-03-20",
                    "2026-03-23",
                    "2026-03-24",
                ]
            ),
            "y": [50.0, 51.0, 52.0, 53.0, 54.0],
        }
    )
    config_raw = {
        "time_series": {"pred_len": 1, "seq_len": 1},
        "frontend_live_bundle": {"min_train_windows": 0, "min_val_windows": 0},
    }

    monkeypatch.setattr(
        live_deployment,
        "_load_live_feature_panel",
        lambda **kwargs: panel.copy(),
    )

    seen_origins: list[str] = []

    def _fake_build_live_rows_from_feature_panel(
        *,
        config_raw: dict,
        panel: pd.DataFrame,
        run_dir: str | Path,
        artifact_run_dir: str | Path | None = None,
        model_version: str = "model",
        record_source: str = "historical_live_backfill",
        enable_llm: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        origin = str(pd.Timestamp(panel["date"].iloc[-1]).date())
        seen_origins.append(origin)
        rows = pd.DataFrame(
            [
                {
                    "run_id": f"run-{origin}",
                    "forecast_made_on": origin,
                    "latest_observed_date": origin,
                    "latest_observed_price": float(panel["y"].iloc[-1]),
                    "target_date": origin,
                    "step_index": 1,
                    "actual": None,
                    "base_tsm_forecast": float(panel["y"].iloc[-1]) + 0.1,
                    "llm_tsm_forecast": float(panel["y"].iloc[-1]) + 0.2,
                    "model_version": model_version,
                    "model_commit": "",
                    "data_version": origin,
                    "is_realized": False,
                    "generated_at": "2026-03-24T12:00:00+00:00",
                    "record_source": record_source,
                    "source_run_id": "fake",
                }
            ]
        )
        return rows, panel[["date", "y"]].copy(), {"origin_date": origin}

    monkeypatch.setattr(
        live_deployment,
        "_build_live_rows_from_feature_panel",
        _fake_build_live_rows_from_feature_panel,
    )

    rows, panel_prices = build_recent_live_history_rows(
        config_raw=config_raw,
        data_dir=tmp_path,
        run_dir=tmp_path / "live-run",
        start_after_date="2026-03-19",
        model_version="uk_ets_llm_4b_canonical_default",
        record_source="historical_live_backfill",
    )

    assert seen_origins == ["2026-03-20", "2026-03-23"]
    assert rows["latest_observed_date"].tolist() == ["2026-03-20", "2026-03-23"]
    assert rows["record_source"].tolist() == ["historical_live_backfill", "historical_live_backfill"]
    assert panel_prices["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-03-18",
        "2026-03-19",
        "2026-03-20",
        "2026-03-23",
        "2026-03-24",
    ]


def test_build_recent_live_history_rows_can_include_latest_origin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-03-20", "2026-03-23", "2026-03-24"]),
            "y": [52.0, 53.0, 54.0],
        }
    )
    config_raw = {
        "time_series": {"pred_len": 1, "seq_len": 1},
        "frontend_live_bundle": {"min_train_windows": 0, "min_val_windows": 0},
    }

    monkeypatch.setattr(
        live_deployment,
        "_load_live_feature_panel",
        lambda **kwargs: panel.copy(),
    )

    seen_origins: list[str] = []

    def _fake_build_live_rows_from_feature_panel(
        *,
        config_raw: dict,
        panel: pd.DataFrame,
        run_dir: str | Path,
        artifact_run_dir: str | Path | None = None,
        model_version: str = "model",
        record_source: str = "historical_live_backfill",
        enable_llm: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        origin = str(pd.Timestamp(panel["date"].iloc[-1]).date())
        seen_origins.append(origin)
        return (
            pd.DataFrame(
                [
                    {
                        "run_id": f"run-{origin}",
                        "forecast_made_on": origin,
                        "latest_observed_date": origin,
                        "latest_observed_price": float(panel["y"].iloc[-1]),
                        "target_date": origin,
                        "step_index": 1,
                        "actual": None,
                        "base_tsm_forecast": float(panel["y"].iloc[-1]) + 0.1,
                        "llm_tsm_forecast": float(panel["y"].iloc[-1]) + 0.2,
                        "model_version": model_version,
                        "model_commit": "",
                        "data_version": origin,
                        "is_realized": False,
                        "generated_at": "2026-03-24T12:00:00+00:00",
                        "record_source": record_source,
                        "source_run_id": "fake",
                    }
                ]
            ),
            panel[["date", "y"]].copy(),
            {"origin_date": origin},
        )

    monkeypatch.setattr(
        live_deployment,
        "_build_live_rows_from_feature_panel",
        _fake_build_live_rows_from_feature_panel,
    )

    rows, _panel_prices = build_recent_live_history_rows(
        config_raw=config_raw,
        data_dir=tmp_path,
        run_dir=tmp_path / "live-run",
        start_after_date="2026-03-19",
        include_latest_origin=True,
    )

    assert seen_origins == ["2026-03-20", "2026-03-23", "2026-03-24"]
    assert rows["latest_observed_date"].tolist() == ["2026-03-20", "2026-03-23", "2026-03-24"]


class _FakeRefiner:
    def __init__(self, outputs: list[np.ndarray]) -> None:
        self._outputs = [np.asarray(output, dtype=float) for output in outputs]
        self.calls: list[dict[str, object]] = []

    def refine(
        self,
        *,
        method: str,
        history: np.ndarray,
        dates: list[str],
        tsm_forecast: np.ndarray,
        pred_len: int,
        exogenous_summary: dict | None,
        price_base: float,
        teaching_examples: list[dict],
        sentiment_history: np.ndarray | None,
    ) -> tuple[np.ndarray | None, dict]:
        self.calls.append(
            {
                "method": method,
                "history_len": len(history),
                "pred_len": pred_len,
                "teaching_examples": len(teaching_examples),
            }
        )
        if not self._outputs:
            return None, {"success": False}
        return self._outputs.pop(0), {"success": True}


def test_simulated_online_memory_history_blocks_current_live_case() -> None:
    panel_dates = pd.bdate_range("2026-01-01", periods=30).to_numpy()
    pool_dates = panel_dates[10:13]
    pool_histories = np.asarray(
        [
            [10.0, 10.1, 10.2, 10.3, 10.4],
            [10.1, 10.2, 10.3, 10.4, 10.5],
            [10.2, 10.3, 10.4, 10.5, 10.6],
        ],
        dtype=float,
    )
    pool_forecasts = np.asarray(
        [
            [10.5, 10.6, 10.7, 10.8, 10.9],
            [10.6, 10.7, 10.8, 10.9, 11.0],
            [10.7, 10.8, 10.9, 11.0, 11.1],
        ],
        dtype=float,
    )
    pool_truth = np.asarray(
        [
            [10.5, 10.6, 10.7, 10.8, 10.9],
            [10.2, 10.2, 10.2, 10.2, 10.2],
            [10.3, 10.3, 10.3, 10.3, 10.3],
        ],
        dtype=float,
    )
    pool_windows = pool_histories[:, :, None]
    llm_cfg = {
        "history_points": 5,
        "max_exogenous_features": 1,
        "cot_rf": {
            "example_selection": "recent",
            "feature_window": 5,
            "k_examples": 1,
            "test_pool_mode": "online_realized_memory",
            "online_memory_policy": {
                "enabled": True,
                "support_examples": 1,
                "positive_examples": 1,
                "negative_examples": 1,
                "max_total_examples": 3,
                "positive_margin": 0.01,
                "negative_margin": -0.01,
                "warmup_min_realized": 1,
                "gate": {
                    "min_positive_examples": 1,
                    "min_positive_signal": 0.0,
                    "min_net_signal": 0.0,
                    "max_negative_signal": 1.0,
                    "min_abs_base_h20_pct": 0.0,
                },
            },
        },
    }
    fake_refiner = _FakeRefiner(
        outputs=[
            np.asarray([11.6, 11.6, 11.6, 11.6, 11.6], dtype=float),
            np.asarray([11.7, 11.7, 11.7, 11.7, 11.7], dtype=float),
        ]
    )
    date_to_idx = {
        pd.Timestamp(value).normalize(): idx
        for idx, value in enumerate(pd.to_datetime(panel_dates))
    }

    simulated = _simulate_live_online_memory_history(
        method_name="TSM+LLM-COT-RF-HDELTA",
        refiner=fake_refiner,
        pool_dates=pool_dates,
        pool_histories=pool_histories,
        pool_forecasts=pool_forecasts,
        pool_truth=pool_truth,
        pool_windows=pool_windows,
        panel_dates=panel_dates,
        date_to_idx=date_to_idx,
        llm_cfg=llm_cfg,
        target_col="y",
        feature_cols=["y"],
        max_exogenous_features=1,
        preferred_feature_order=None,
        sentiment_map=None,
    )

    assert len(fake_refiner.calls) == 2
    assert len(simulated["records"]) == 2
    assert {record["admission_label"] for record in simulated["records"]} == {"negative"}

    support_examples = [
        {
            "history": pool_histories[-1],
            "forecast": pool_forecasts[-1],
            "truth": pool_truth[-1],
            "date": str(pd.Timestamp(pool_dates[-1]).date()),
            "sentiment_history": [],
        }
    ]
    teaching_examples, gate_decision = _build_current_online_memory_context(
        current_origin_date=pd.Timestamp(panel_dates[20]),
        current_history=np.asarray([10.4, 10.5, 10.6, 10.7, 10.8], dtype=float),
        current_forecast=np.asarray([10.9, 11.0, 11.1, 11.2, 11.3], dtype=float),
        current_price=10.8,
        current_window=np.asarray([[10.4], [10.5], [10.6], [10.7], [10.8]], dtype=float),
        current_history_dates=[str(pd.Timestamp(value).date()) for value in panel_dates[15:20]],
        support_examples=support_examples,
        online_memory_records=simulated["records"],
        llm_cfg=llm_cfg,
        gate_cfg=llm_cfg["cot_rf"]["online_memory_policy"]["gate"],
        learned_gate_bundle=None,
        target_col="y",
        feature_cols=["y"],
        max_exogenous_features=1,
        preferred_feature_order=None,
    )

    assert gate_decision["apply_llm"] is False
    assert gate_decision["reason"] == "memory_gate_blocked"
    assert gate_decision["positive_example_count"] == 0
    assert gate_decision["negative_example_count"] >= 1
    assert len(teaching_examples) >= 2


def test_fit_live_learned_gate_bundle_uses_selection_payload_threshold() -> None:
    gate_feature_rows = [
        {
            "warmup_ready": True,
            "positive_count": 3,
            "negative_count": 0,
            "positive_signal": 0.30,
            "negative_signal": 0.00,
            "net_signal": 0.30,
            "support_example_count": 2,
            "base_move_h20_pct": 0.5,
        },
        {
            "warmup_ready": True,
            "positive_count": 0,
            "negative_count": 2,
            "positive_signal": 0.00,
            "negative_signal": 0.25,
            "net_signal": -0.25,
            "support_example_count": 2,
            "base_move_h20_pct": 0.5,
        },
        {
            "warmup_ready": True,
            "positive_count": 2,
            "negative_count": 0,
            "positive_signal": 0.20,
            "negative_signal": 0.00,
            "net_signal": 0.20,
            "support_example_count": 2,
            "base_move_h20_pct": 0.5,
        },
        {
            "warmup_ready": True,
            "positive_count": 0,
            "negative_count": 3,
            "positive_signal": 0.00,
            "negative_signal": 0.30,
            "net_signal": -0.30,
            "support_example_count": 2,
            "base_move_h20_pct": 0.5,
        },
    ]
    base_pred = np.asarray(
        [
            [10.0, 10.1, 10.2, 10.3, 10.4],
            [10.0, 10.1, 10.2, 10.3, 10.4],
            [10.0, 10.1, 10.2, 10.3, 10.4],
            [10.0, 10.1, 10.2, 10.3, 10.4],
        ],
        dtype=float,
    )
    llm_pred = np.asarray(
        [
            [10.0, 10.1, 10.2, 10.3, 10.4],
            [10.8, 10.8, 10.8, 10.8, 10.8],
            [10.0, 10.1, 10.2, 10.3, 10.4],
            [10.9, 10.9, 10.9, 10.9, 10.9],
        ],
        dtype=float,
    )
    y_true = np.asarray(
        [
            [10.0, 10.1, 10.2, 10.3, 10.4],
            [10.0, 10.1, 10.2, 10.3, 10.4],
            [10.0, 10.1, 10.2, 10.3, 10.4],
            [10.0, 10.1, 10.2, 10.3, 10.4],
        ],
        dtype=float,
    )
    selection_payload = {
        "gate_mode": "learned",
        "model_type": "logistic",
        "feature_columns": ["positive_count", "negative_count", "net_signal"],
        "threshold": 0.3,
        "regime_thresholds": {},
        "regime_threshold_cfg": {},
        "selection_path": "/tmp/online_memory_gate_selection.json",
    }

    bundle = _fit_live_learned_gate_bundle(
        gate_feature_rows=gate_feature_rows,
        y_true=y_true,
        base_pred=base_pred,
        llm_pred=llm_pred,
        gate_cfg={"learned": {"enabled": False}},
        selection_payload=selection_payload,
    )

    assert bundle is not None
    assert bundle["threshold"] == pytest.approx(0.3)
    assert bundle["selection_path"] == selection_payload["selection_path"]

    feature_df = build_online_memory_gate_feature_frame(gate_feature_rows)
    scores = predict_online_memory_learned_gate_scores(feature_df, bundle)
    assert scores.shape == (4,)
    assert scores[0] != scores[1]
