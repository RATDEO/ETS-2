#!/usr/bin/env python3
"""Benchmark teaching-example count on W1-W4 using the improved Huber energy-interaction base."""

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
from uk_ets.scripts.run_tsm_live_policy_build import _candidates as live_policy_candidates


CONFIG_PATH = "uk_ets/config/uk_ets_llm_4b_current_default.yaml"
DATA_DIR = "uk_ets/Data_auto_uk"
METHOD_NAME = "TSM+LLM-COT-RF-HDELTA"
BASELINE_RESULTS = (
    PROJECT_ROOT
    / "reports"
    / "uk_ets_live_policy_w1_w4_huber_energy"
    / "20260319_222459"
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

K_VALUES = [2, 3, 4, 5, 6, 20, 30]


def _merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _improved_base_overrides() -> dict[str, Any]:
    return {
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


def _baseline_policy_overrides() -> dict[str, Any]:
    candidate_map = {candidate.name: candidate.overrides for candidate in live_policy_candidates()}
    return candidate_map["baseline_regime_specific"]


def _k_overrides(k_examples: int) -> dict[str, Any]:
    return {
        "llm": {
            "cot_rf": {
                "k_examples": int(k_examples),
                "n_similarity_examples": int(min(3, k_examples)),
            }
        }
    }


def _run_candidate(window: WindowSpec, k_examples: int) -> Path:
    overrides = _merge(
        _merge(
            _merge(_improved_base_overrides(), _baseline_policy_overrides()),
            _k_overrides(k_examples),
        ),
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


def _import_k4_rows() -> list[dict[str, Any]]:
    df = pd.read_csv(BASELINE_RESULTS)
    df = df[df["candidate"] == "baseline_regime_specific"].copy()
    df["k_examples"] = 4
    return df.to_dict(orient="records")


def _write_summary(out_dir: Path, df: pd.DataFrame) -> None:
    mean_df = (
        df.groupby("k_examples")[["tsm_path_mse", "llm_path_mse", "path_improvement_pct", "h20_improvement_pct", "h30_improvement_pct"]]
        .mean()
        .sort_values("llm_path_mse")
        .reset_index()
    )
    lines = [
        "# W1-W4 Teaching-Example Count Benchmark On Improved Base",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Setup",
        "",
        "- Base fixed at the improved hard-regime anchor: shared `DLinear`, `seq_len=20`, regularized, `Huber(beta=0.5)`, `energy_interactions16`.",
        "- Gate fixed at the current best heuristic live policy.",
        "- Only `llm.cot_rf.k_examples` changes across candidates.",
        "- `n_similarity_examples` is set to `min(3, k_examples)`.",
        "",
        "## Results",
        "",
        df.to_markdown(index=False),
        "",
        "## Mean By k_examples",
        "",
        mean_df.to_markdown(index=False),
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
            / "uk_ets_k_examples_w1_w4_huber_energy"
            / datetime.now().strftime("%Y%m%d_%H%M%S")
        ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / "results.csv"
    if results_path.exists():
        rows = pd.read_csv(results_path).to_dict(orient="records")
    else:
        rows = _import_k4_rows()
        pd.DataFrame(rows).to_csv(results_path, index=False)
    completed = {(int(row["k_examples"]), str(row["window"])) for row in rows}

    for k_examples in K_VALUES:
        for window in WINDOWS:
            key = (int(k_examples), window.name)
            if key in completed:
                print(f"Skipping completed k={k_examples} {window.name}", flush=True)
                continue
            print(f"Running k={k_examples} {window.name}", flush=True)
            run_dir = _run_candidate(window, k_examples)
            metrics = _extract_metrics(run_dir)
            rows.append(
                {
                    "k_examples": int(k_examples),
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
