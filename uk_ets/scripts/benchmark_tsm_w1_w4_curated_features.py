#!/usr/bin/env python3
"""Benchmark curated engineered base-feature slates on difficult windows W1-W4."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.run_experiment import run_experiment


DATA_DIR = "uk_ets/Data_auto_uk"
CONFIG_PATH = "uk_ets/config/uk_ets_llm_4b_tsm_live_gbdt_top20.yaml"


@dataclass(frozen=True)
class WindowSpec:
    name: str
    train_end: str
    val_end: str
    test_end: str


WINDOWS = [
    WindowSpec("W1", "2023-10-27", "2024-10-26", "2025-06-30"),
    WindowSpec("W2", "2023-02-22", "2024-02-22", "2024-10-26"),
    WindowSpec("W3", "2022-06-20", "2023-06-20", "2024-02-22"),
    WindowSpec("W4", "2021-10-16", "2022-10-16", "2023-06-20"),
]


BASE_MODEL = {
    "learning_rate": 0.001,
    "max_epochs": 40,
    "early_stopping_patience": 8,
    "dropout": 0.5,
    "weight_decay": 0.02,
    "dlinear_individual": False,
    "loss_type": "huber",
    "loss_huber_beta": 0.5,
}


CORE_TARGET = [
    "target_range_pct",
    "target_volume",
    "y_vol_20d",
    "y_ma_5d",
    "y_momentum_5d",
    "y_momentum_20d",
]

CURRENT_LIVE_ORDER = [
    "target_range_pct",
    "target_volume",
    "is_auction_day",
    "uk_icap_primary_print_day",
    "uk_icap_secondary_print_day",
    "y_vol_20d",
    "y_ma_5d",
    "y_momentum_5d",
    "y_momentum_20d",
    "uk_power_return",
    "uk_gas_return",
    "brent_return",
    "coal_return",
    "uk_hdd18",
    "uk_hdd18_7d_ma",
    "uk_temp_mean_c",
]

ENGINEERED_CURATED = [
    "target_range_pct",
    "target_volume",
    "y_vol_20d",
    "y_ma_5d",
    "y_momentum_5d",
    "y_momentum_20d",
    "uk_icap_secondary",
    "is_auction_day",
    "auction_price_lag_20",
    "coal_brent_ratio",
    "uka_brent_ratio",
    "uk_power_gas_vol_ratio_20d",
    "uk_gas_hdd18_surprise_interaction",
    "uk_gas_vol_20d",
    "uk_temp_mean_c",
    "uk_hdd18_7d_ma",
]

LASSO_SCREEN_COMPACT = [
    "target_range_pct",
    "target_volume",
    "y_vol_20d",
    "y_ma_20d",
    "y_return",
    "is_auction_day",
    "uk_icap_secondary",
    "uk_icap_primary_lag_20",
    "auction_price_lag_20",
    "coal_brent_ratio",
    "coal_vol_5d",
    "uk_gas_vol_20d",
    "uk_temp_mean_c",
    "uk_temp_min_c",
    "uk_temp_max_c",
]

ENERGY_INTERACTIONS = [
    "target_range_pct",
    "target_volume",
    "y_vol_20d",
    "y_ma_5d",
    "y_momentum_20d",
    "is_auction_day",
    "uk_icap_secondary_print_day",
    "coal_brent_ratio",
    "coal_brent_ratio_z20",
    "uk_power_gas_vol_ratio_20d",
    "uk_gas_hdd18_surprise_interaction",
    "uk_hdd18_7d_ma",
    "uka_brent_ratio",
    "uka_brent_ratio_z20",
    "uk_gas_vol_20d",
    "uk_temp_mean_c",
]


def _merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _feature_overrides(order: list[str]) -> dict[str, Any]:
    return {
        "max_exogenous_features_model": len(order),
        "preferred_feature_order": order,
    }


def _candidate_overrides() -> dict[str, dict[str, Any]]:
    return {
        "huber_live_current16": {
            "llm": {"methods": []},
            "time_series": {"seq_len": 20, "label_len": 10},
            "model": BASE_MODEL,
            "features": _feature_overrides(CURRENT_LIVE_ORDER),
        },
        "huber_engineered_curated16": {
            "llm": {"methods": []},
            "time_series": {"seq_len": 20, "label_len": 10},
            "model": BASE_MODEL,
            "features": _feature_overrides(ENGINEERED_CURATED),
        },
        "huber_lasso_screen15": {
            "llm": {"methods": []},
            "time_series": {"seq_len": 20, "label_len": 10},
            "model": BASE_MODEL,
            "features": _feature_overrides(LASSO_SCREEN_COMPACT),
        },
        "huber_energy_interactions16": {
            "llm": {"methods": []},
            "time_series": {"seq_len": 20, "label_len": 10},
            "model": BASE_MODEL,
            "features": _feature_overrides(ENERGY_INTERACTIONS),
        },
    }


def _run_candidate(window: WindowSpec, candidate_name: str, base_overrides: dict[str, Any]) -> Path:
    overrides = _merge(
        base_overrides,
        {
            "split": {
                "train_end": window.train_end,
                "val_end": window.val_end,
                "test_end": window.test_end,
            }
        },
    )
    return Path(
        run_experiment(
            config_path=CONFIG_PATH,
            overrides=overrides,
            data_dir=DATA_DIR,
        )
    ).resolve()


def _extract_metrics(run_dir: Path) -> dict[str, float]:
    path_df = pd.read_csv(run_dir / "results" / "path_metrics.csv")
    horizon_df = pd.read_csv(run_dir / "results" / "metrics_by_horizon.csv")
    tsm_row = path_df[path_df["model"] == "tsm"].iloc[0]
    h_rows = horizon_df[horizon_df["model"] == "tsm"].set_index("horizon")
    return {
        "path_mse": float(tsm_row["mse_path"]),
        "h1": float(h_rows.loc[1, "mse"]),
        "h5": float(h_rows.loc[5, "mse"]),
        "h20": float(h_rows.loc[20, "mse"]),
        "h30": float(h_rows.loc[30, "mse"]),
    }


def _write_summary(out_dir: Path, df: pd.DataFrame) -> None:
    mean_df = (
        df.groupby("candidate")[["path_mse", "h1", "h5", "h20", "h30"]]
        .mean()
        .sort_values("path_mse")
        .reset_index()
    )
    baseline = mean_df[mean_df["candidate"] == "huber_live_current16"].iloc[0]
    best = mean_df.iloc[0]
    gain = 1.0 - (float(best["path_mse"]) / float(baseline["path_mse"]))
    best_rows = df.loc[df.groupby("window")["path_mse"].idxmin()].sort_values("window")
    lines = [
        "# W1-W4 Curated Feature Benchmark",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Setup",
        "",
        "- Base only: `llm.methods=[]`.",
        "- Fixed base anchor: shared `DLinear`, `seq_len=20`, `label_len=10`, regularized, `Huber` loss.",
        "- Only curated feature slates vary.",
        "",
        "## Results",
        "",
        df.to_markdown(index=False),
        "",
        "## Mean By Candidate",
        "",
        mean_df.to_markdown(index=False),
        "",
        "## Best Candidate Per Window",
        "",
        best_rows.to_markdown(index=False),
        "",
        "## Headline",
        "",
        f"- Best mean candidate: `{best['candidate']}` with path MSE `{best['path_mse']:.6f}`.",
        f"- Baseline `huber_live_current16` path MSE: `{baseline['path_mse']:.6f}`.",
        f"- Improvement vs baseline: `{gain:.2%}`.",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    out_dir = (
        PROJECT_ROOT
        / "reports"
        / "uk_ets_tsm_w1_w4_curated_feature_benchmark"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for candidate_name, overrides in _candidate_overrides().items():
        feature_order = overrides["features"]["preferred_feature_order"]
        for window in WINDOWS:
            run_dir = _run_candidate(window, candidate_name, overrides)
            metrics = _extract_metrics(run_dir)
            rows.append(
                {
                    "candidate": candidate_name,
                    "window": window.name,
                    "train_end": window.train_end,
                    "val_end": window.val_end,
                    "test_end": window.test_end,
                    "run_dir": str(run_dir),
                    "n_features": len(feature_order),
                    "feature_order": "|".join(feature_order),
                    **metrics,
                }
            )
            pd.DataFrame(rows).to_csv(out_dir / "results.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "results.csv", index=False)
    _write_summary(out_dir, df)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
