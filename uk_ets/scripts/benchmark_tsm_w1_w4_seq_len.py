#!/usr/bin/env python3
"""Benchmark longer-context DLinear candidates on difficult windows W1-W4."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

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


def _load_cfg(path: str) -> dict[str, Any]:
    with (PROJECT_ROOT / path).open() as f:
        return yaml.safe_load(f)


LIVE_CFG = _load_cfg(CONFIG_PATH)


def _merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _label_len_for(seq_len: int) -> int:
    return max(10, seq_len // 2)


def _candidate_overrides() -> dict[str, dict[str, Any]]:
    base_model = {
        "learning_rate": 0.001,
        "max_epochs": 40,
        "early_stopping_patience": 8,
        "dropout": 0.5,
        "weight_decay": 0.02,
    }
    candidates: dict[str, dict[str, Any]] = {}
    for seq_len in (20, 40, 60, 90):
        candidates[f"seq{seq_len}_reg_shared"] = {
            "llm": {"methods": []},
            "time_series": {
                "seq_len": seq_len,
                "label_len": _label_len_for(seq_len),
            },
            "model": base_model,
        }
    return candidates


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


def _extract_metrics(run_dir: Path) -> dict[str, float | str]:
    path_df = pd.read_csv(run_dir / "results" / "path_metrics.csv")
    horizon_df = pd.read_csv(run_dir / "results" / "metrics_by_horizon.csv")
    tsm_rows = path_df[path_df["model"] == "tsm"]
    if tsm_rows.empty:
        return {
            "status": "tsm_unavailable",
            "path_mse": float("nan"),
            "h1": float("nan"),
            "h5": float("nan"),
            "h20": float("nan"),
            "h30": float("nan"),
        }
    tsm_row = tsm_rows.iloc[0]
    h_rows = horizon_df[horizon_df["model"] == "tsm"].set_index("horizon")
    return {
        "status": "ok",
        "path_mse": float(tsm_row["mse_path"]),
        "h1": float(h_rows.loc[1, "mse"]),
        "h5": float(h_rows.loc[5, "mse"]),
        "h20": float(h_rows.loc[20, "mse"]),
        "h30": float(h_rows.loc[30, "mse"]),
    }


def _write_plan(out_dir: Path) -> None:
    lines = [
        "# W1-W4 Base Rebuild Plan",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Objective",
        "",
        "Improve the hard-regime base `TSM` (`DLinear`) on `W1-W4` before spending more time on the LLM gate.",
        "",
        "## Diagnosis",
        "",
        "- The corrected regularized shared `DLinear` already improved mean `W1-W4` base path MSE from `61.326236` to `54.606079`.",
        "- Restoring the wider `29`-feature gate did not materially help.",
        "- The remaining weakness is concentrated in the base model, especially `W3/W4` long horizons.",
        "",
        "## Planned Steps",
        "",
        "1. Longer-context `DLinear` sweep",
        "   - Test `seq_len` `20 / 40 / 60 / 90`.",
        "   - Keep the best known regularized shared recipe fixed.",
        "   - Expected outcome: `40` or `60` should outperform `20` if the hard windows need longer regime memory.",
        "   - Success threshold: beat mean `W1-W4` base path MSE `54.606079`.",
        "",
        "2. Shared vs individual `DLinear`",
        "   - Freeze the best `seq_len` from step 1.",
        "   - Test `dlinear_individual=true` against shared.",
        "   - Expected outcome: modest gain if per-channel decomposition helps the harder windows.",
        "",
        "3. Loss shaping",
        "   - Freeze the best architecture from steps 1-2.",
        "   - Test robust and/or horizon-weighted training.",
        "   - Expected outcome: improve `h20/h30` tail behavior in `W3/W4`.",
        "",
        "4. Curated engineered base features",
        "   - Add only tightly engineered UK energy/weather regime features.",
        "   - Avoid the broad raw expansion that previously degraded the base.",
        "",
        "5. Hybrid base if needed",
        "   - If pure `DLinear` stalls, test a validation-selected `DLinear + ridge` base blend.",
        "",
        "## Current Step",
        "",
        "Run step 1 only: longer-context `seq_len` sweep on the same `W1-W4` benchmark geometry.",
    ]
    (out_dir / "plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary(out_dir: Path, df: pd.DataFrame) -> None:
    valid_df = df[df["status"] == "ok"].copy()
    mean_df = (
        valid_df.groupby("candidate")[["path_mse", "h1", "h5", "h20", "h30"]]
        .mean()
        .sort_values("path_mse")
        .reset_index()
    )
    feasible_counts = valid_df.groupby("candidate")["window"].nunique().reset_index(name="n_valid_windows")
    feasible_mean_df = mean_df.merge(feasible_counts, on="candidate", how="left")
    fully_feasible_df = feasible_mean_df[feasible_mean_df["n_valid_windows"] == len(WINDOWS)].copy()
    best = fully_feasible_df.iloc[0] if not fully_feasible_df.empty else feasible_mean_df.iloc[0]
    baseline_row = mean_df[mean_df["candidate"] == "seq20_reg_shared"].iloc[0]
    gain = 1.0 - (float(best["path_mse"]) / float(baseline_row["path_mse"]))
    best_rows = valid_df.loc[valid_df.groupby("window")["path_mse"].idxmin()].sort_values("window")
    invalid_rows = df[df["status"] != "ok"].copy()
    common_windows = set(df["window"])
    for candidate_name, group in valid_df.groupby("candidate"):
        common_windows &= set(group["window"])
    common_df = valid_df[valid_df["window"].isin(sorted(common_windows))].copy()
    common_mean_df = (
        common_df.groupby("candidate")[["path_mse", "h1", "h5", "h20", "h30"]]
        .mean()
        .sort_values("path_mse")
        .reset_index()
    )
    lines = [
        "# W1-W4 Seq-Len Sweep",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        f"Data dir forced to `{DATA_DIR}`.",
        "",
        "## Setup",
        "",
        "- Base only: `llm.methods=[]`.",
        "- Fixed regularized shared `DLinear`: `lr=0.001`, `epochs=40`, `patience=8`, `dropout=0.5`, `weight_decay=0.02`.",
        "- Only `seq_len` and matching `label_len` vary.",
        "",
        "## Results",
        "",
        df.to_markdown(index=False),
        "",
        "## Mean By Candidate",
        "",
        feasible_mean_df.to_markdown(index=False),
        "",
        "## Mean By Candidate (Common Feasible Windows Only)",
        "",
        common_mean_df.to_markdown(index=False),
        "",
        "## Best Candidate Per Window",
        "",
        best_rows.to_markdown(index=False),
    ]
    if not invalid_rows.empty:
        lines.extend(
            [
                "",
                "## Infeasible Candidates",
                "",
                invalid_rows.to_markdown(index=False),
            ]
        )
    lines.extend(
        [
        "",
        "## Headline",
        "",
        f"- Best mean candidate: `{best['candidate']}` with path MSE `{best['path_mse']:.6f}`.",
        f"- Baseline `seq20_reg_shared` path MSE: `{baseline_row['path_mse']:.6f}`.",
        f"- Improvement vs `seq20_reg_shared`: `{gain:.2%}`.",
        f"- Feasible windows required for headline: `{len(WINDOWS)}`.",
    ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    out_dir = (
        PROJECT_ROOT
        / "reports"
        / "uk_ets_tsm_w1_w4_seq_len_benchmark"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_plan(out_dir)

    rows: list[dict[str, Any]] = []
    for candidate_name, overrides in _candidate_overrides().items():
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
                    "seq_len": int(overrides["time_series"]["seq_len"]),
                    "label_len": int(overrides["time_series"]["label_len"]),
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
