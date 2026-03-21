#!/usr/bin/env python3
"""Benchmark the live top20 gate stack on W1-W4 with the improved Huber energy-interaction base."""

from __future__ import annotations

import argparse
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


CONFIG_PATH = "uk_ets/config/uk_ets_llm_4b_tsm_live_gbdt_top20.yaml"
DATA_DIR = "uk_ets/Data_auto_uk"
METHOD_NAME = "TSM+LLM-COT-RF-HDELTA"
BASELINE_RESULTS = (
    PROJECT_ROOT
    / "reports"
    / "uk_ets_gate_w1_w4_regularized_base"
    / "20260319_000250"
    / "results.csv"
)

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


def _merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _candidate_overrides() -> dict[str, dict[str, Any]]:
    return {
        "gate_top20_huber_energy16": {
            "time_series": {
                "seq_len": 20,
                "label_len": 10,
            },
            "model": {
                "learning_rate": 0.001,
                "max_epochs": 40,
                "early_stopping_patience": 8,
                "dropout": 0.5,
                "weight_decay": 0.02,
                "dlinear_individual": False,
                "loss_type": "huber",
                "loss_huber_beta": 0.5,
            },
            "features": {
                "max_exogenous_features_model": len(ENERGY_INTERACTIONS),
                "preferred_feature_order": ENERGY_INTERACTIONS,
            },
        }
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
    rows = path_df[path_df["model"].isin(["tsm_llm_subset", METHOD_NAME])]
    h_rows = horizon_df[horizon_df["model"].isin(["tsm_llm_subset", METHOD_NAME])]
    out: dict[str, float] = {}
    for model_name, prefix in [("tsm_llm_subset", "tsm"), (METHOD_NAME, "llm")]:
        out[f"{prefix}_path_mse"] = float(rows[rows["model"] == model_name].iloc[0]["mse_path"])
        by_h = h_rows[h_rows["model"] == model_name].set_index("horizon")
        out[f"{prefix}_h20"] = float(by_h.loc[20, "mse"])
        out[f"{prefix}_h30"] = float(by_h.loc[30, "mse"])
    out["path_improvement_pct"] = 1.0 - (out["llm_path_mse"] / out["tsm_path_mse"])
    out["h20_improvement_pct"] = 1.0 - (out["llm_h20"] / out["tsm_h20"])
    out["h30_improvement_pct"] = 1.0 - (out["llm_h30"] / out["tsm_h30"])
    return out


def _load_baseline() -> pd.DataFrame:
    df = pd.read_csv(BASELINE_RESULTS)
    return df[df["candidate"] == "gate_top20"].copy().reset_index(drop=True)


def _write_summary(out_dir: Path, df: pd.DataFrame) -> None:
    baseline_df = _load_baseline()
    compare = (
        baseline_df[["window", "tsm_path_mse", "llm_path_mse", "path_improvement_pct"]]
        .rename(
            columns={
                "tsm_path_mse": "baseline_tsm_path_mse",
                "llm_path_mse": "baseline_llm_path_mse",
                "path_improvement_pct": "baseline_path_improvement_pct",
            }
        )
        .merge(
            df[["window", "tsm_path_mse", "llm_path_mse", "path_improvement_pct"]],
            on="window",
            how="inner",
        )
    )
    compare["base_path_gain_pct"] = 1.0 - (
        compare["tsm_path_mse"] / compare["baseline_tsm_path_mse"]
    )
    compare["llm_path_gain_pct"] = 1.0 - (
        compare["llm_path_mse"] / compare["baseline_llm_path_mse"]
    )

    baseline_means = baseline_df[["tsm_path_mse", "llm_path_mse", "path_improvement_pct"]].mean()
    new_means = df[["tsm_path_mse", "llm_path_mse", "path_improvement_pct"]].mean()

    lines = [
        "# W1-W4 Live Gate Benchmark With Improved Huber Energy Base",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Setup",
        "",
        "- Live stack frozen except for the improved base.",
        "- Same `top20` learned gate feature schema.",
        "- Same online-memory / live LLM refinement path.",
        "- Base changed to shared `DLinear` with `Huber(beta=0.5)` and the curated `energy_interactions16` feature slate.",
        "- Data root fixed to `uk_ets/Data_auto_uk`.",
        "",
        "## Results",
        "",
        df.to_markdown(index=False),
        "",
        "## Comparison vs Prior Regularized-Base Gate Benchmark",
        "",
        compare.to_markdown(index=False),
        "",
        "## Mean Comparison",
        "",
        f"- Prior base mean path MSE: `{baseline_means['tsm_path_mse']:.6f}`",
        f"- New base mean path MSE: `{new_means['tsm_path_mse']:.6f}`",
        f"- Prior live-stack mean path MSE: `{baseline_means['llm_path_mse']:.6f}`",
        f"- New live-stack mean path MSE: `{new_means['llm_path_mse']:.6f}`",
        f"- Prior mean LLM uplift vs base: `{baseline_means['path_improvement_pct']:.2%}`",
        f"- New mean LLM uplift vs base: `{new_means['path_improvement_pct']:.2%}`",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = (
            PROJECT_ROOT
            / "reports"
            / "uk_ets_gate_w1_w4_huber_energy_base"
            / datetime.now().strftime("%Y%m%d_%H%M%S")
        ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / "results.csv"
    if results_path.exists():
        rows = pd.read_csv(results_path).to_dict(orient="records")
    else:
        rows = []
    completed = {(str(row["candidate"]), str(row["window"])) for row in rows}

    for candidate_name, overrides in _candidate_overrides().items():
        for window in WINDOWS:
            key = (candidate_name, window.name)
            if key in completed:
                print(f"Skipping completed {candidate_name} {window.name}", flush=True)
                continue
            print(f"Running {candidate_name} {window.name}", flush=True)
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
                    **metrics,
                }
            )
            pd.DataFrame(rows).to_csv(results_path, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(results_path, index=False)
    _write_summary(out_dir, df)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
