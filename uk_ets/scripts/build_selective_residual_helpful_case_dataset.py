#!/usr/bin/env python3
"""Build a case-level helpful/harmful LLM refinement dataset for UK ETS."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


FEATURE_COLUMNS = [
    "y",
    "y_return",
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
class CaseRun:
    window: str
    train_end: str
    val_end: str
    test_end: str
    run_dir: Path


DEFAULT_RUNS = [
    CaseRun("W1", "2023-10-27", "2024-10-26", "2025-06-30", PROJECT_ROOT / "runs" / "20260320_182015_da9e9d"),
    CaseRun("W2", "2023-02-22", "2024-02-22", "2024-10-26", PROJECT_ROOT / "runs" / "20260320_183041_c7484b"),
    CaseRun("W3", "2022-06-20", "2023-06-20", "2024-02-22", PROJECT_ROOT / "runs" / "20260320_183637_a4d80c"),
    CaseRun("W4", "2021-10-16", "2022-10-16", "2023-06-20", PROJECT_ROOT / "runs" / "20260320_184725_2cded6"),
]


def _load_runs(runs_csv: str) -> list[CaseRun]:
    if not runs_csv:
        return list(DEFAULT_RUNS)
    df = pd.read_csv(runs_csv)
    required = {"window", "train_end", "val_end", "test_end", "run_dir"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"runs-csv missing columns: {sorted(missing)}")
    return [
        CaseRun(
            window=str(row["window"]),
            train_end=str(row["train_end"]),
            val_end=str(row["val_end"]),
            test_end=str(row["test_end"]),
            run_dir=Path(str(row["run_dir"])).expanduser().resolve(),
        )
        for _, row in df.iterrows()
    ]


def _safe_pct_improvement(base: np.ndarray, refined: np.ndarray) -> np.ndarray:
    out = np.full_like(base, np.nan, dtype=float)
    mask = np.abs(base) > 1e-12
    out[mask] = 1.0 - (refined[mask] / base[mask])
    return out


def _safe_return_pct(current: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    out = np.full_like(forecast, np.nan, dtype=float)
    mask = np.abs(current) > 1e-12
    out[mask] = ((forecast[mask] / current[mask]) - 1.0) * 100.0
    return out


def _load_apply_logs(run_dir: Path) -> list[dict[str, Any]]:
    log_path = run_dir / "llm" / "logs" / "llm_calls.jsonl"
    rows: list[dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if str(record.get("method", "")).endswith(":apply"):
                rows.append(record)
    return rows


def _parse_adjustments(response_content: str) -> dict[str, float]:
    try:
        payload = json.loads(response_content)
        agg = payload.get("aggregated_adjustments", {}) or {}
        return {str(k): float(v) for k, v in agg.items()}
    except Exception:
        return {}


def _assign_quantile_bins(series: pd.Series, labels: list[str]) -> pd.Series:
    ranked = series.rank(method="first")
    return pd.qcut(ranked, q=len(labels), labels=labels)


def _build_case_frame(case_run: CaseRun) -> pd.DataFrame:
    llm_npz = np.load(case_run.run_dir / "predictions" / "TSM+LLM-COT-RF-HDELTA_pred_test_subset.npz", allow_pickle=True)
    dates = pd.to_datetime(llm_npz["dates"])
    y_true = llm_npz["y_true"].astype(float)
    base_pred = llm_npz["base_pred"].astype(float)
    llm_pred = llm_npz["yhat"].astype(float)

    base_path = np.mean((base_pred - y_true) ** 2, axis=1)
    llm_path = np.mean((llm_pred - y_true) ** 2, axis=1)
    base_h1 = (base_pred[:, 0] - y_true[:, 0]) ** 2
    llm_h1 = (llm_pred[:, 0] - y_true[:, 0]) ** 2
    base_h5 = (base_pred[:, 4] - y_true[:, 4]) ** 2
    llm_h5 = (llm_pred[:, 4] - y_true[:, 4]) ** 2
    base_h20 = (base_pred[:, 19] - y_true[:, 19]) ** 2
    llm_h20 = (llm_pred[:, 19] - y_true[:, 19]) ** 2
    base_h30 = (base_pred[:, 29] - y_true[:, 29]) ** 2
    llm_h30 = (llm_pred[:, 29] - y_true[:, 29]) ** 2

    panel = pd.read_parquet(case_run.run_dir / "data" / "panel.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    feature_cols = [col for col in FEATURE_COLUMNS if col in panel.columns]
    panel_slice = panel[["date"] + feature_cols].copy()

    apply_logs = _load_apply_logs(case_run.run_dir)
    applied_mask = np.any(np.abs(llm_pred - base_pred) > 1e-12, axis=1)
    applied_indices = np.flatnonzero(applied_mask)
    if len(apply_logs) != len(applied_indices):
        raise RuntimeError(
            f"Apply-log count mismatch for {case_run.run_dir}: {len(apply_logs)} logs vs {len(applied_indices)} changed forecast cases"
        )
    apply_log_by_idx = {int(case_idx): apply_logs[pos] for pos, case_idx in enumerate(applied_indices)}

    rows: list[dict[str, Any]] = []
    for idx, date in enumerate(dates):
        apply_log = apply_log_by_idx.get(idx)
        metadata = (apply_log or {}).get("metadata", {}) or {}
        adjustments = _parse_adjustments(str(((apply_log or {}).get("response") or {}).get("content", "")))
        row = {
            "window": case_run.window,
            "train_end": case_run.train_end,
            "val_end": case_run.val_end,
            "test_end": case_run.test_end,
            "run_dir": str(case_run.run_dir),
            "date": pd.Timestamp(date).normalize(),
            "base_path_mse": float(base_path[idx]),
            "llm_path_mse": float(llm_path[idx]),
            "path_uplift_abs": float(base_path[idx] - llm_path[idx]),
            "path_uplift_pct": float(_safe_pct_improvement(base_path[idx:idx + 1], llm_path[idx:idx + 1])[0]),
            "base_h1_mse": float(base_h1[idx]),
            "llm_h1_mse": float(llm_h1[idx]),
            "h1_uplift_abs": float(base_h1[idx] - llm_h1[idx]),
            "h1_uplift_pct": float(_safe_pct_improvement(base_h1[idx:idx + 1], llm_h1[idx:idx + 1])[0]),
            "base_h5_mse": float(base_h5[idx]),
            "llm_h5_mse": float(llm_h5[idx]),
            "h5_uplift_abs": float(base_h5[idx] - llm_h5[idx]),
            "h5_uplift_pct": float(_safe_pct_improvement(base_h5[idx:idx + 1], llm_h5[idx:idx + 1])[0]),
            "base_h20_mse": float(base_h20[idx]),
            "llm_h20_mse": float(llm_h20[idx]),
            "h20_uplift_abs": float(base_h20[idx] - llm_h20[idx]),
            "h20_uplift_pct": float(_safe_pct_improvement(base_h20[idx:idx + 1], llm_h20[idx:idx + 1])[0]),
            "base_h30_mse": float(base_h30[idx]),
            "llm_h30_mse": float(llm_h30[idx]),
            "h30_uplift_abs": float(base_h30[idx] - llm_h30[idx]),
            "h30_uplift_pct": float(_safe_pct_improvement(base_h30[idx:idx + 1], llm_h30[idx:idx + 1])[0]),
            "base_bias_h20": float(base_pred[idx, 19] - y_true[idx, 19]),
            "base_bias_h30": float(base_pred[idx, 29] - y_true[idx, 29]),
            "llm_bias_h20": float(llm_pred[idx, 19] - y_true[idx, 19]),
            "llm_bias_h30": float(llm_pred[idx, 29] - y_true[idx, 29]),
            "llm_applied": int(apply_log is not None),
            "apply_prompt_length": int((apply_log or {}).get("prompt_length", 0) or 0),
            "teaching_examples": int(metadata.get("teaching_examples", 0) or 0),
            "matched_teaching_count": len(metadata.get("matched_teaching_dates", []) or []),
            "support_example_count": int(metadata.get("support_example_count", 0) or 0),
            "positive_memory_count": int(metadata.get("positive_memory_count", 0) or 0),
            "negative_memory_count": int(metadata.get("negative_memory_count", 0) or 0),
            "dynamic_frozen_horizons": len(metadata.get("dynamic_frozen_horizons", []) or []),
            "adjust_h1": float(adjustments.get("1", 0.0)),
            "adjust_h5": float(adjustments.get("5", 0.0)),
            "adjust_h20": float(adjustments.get("20", 0.0)),
            "adjust_h30": float(adjustments.get("30", 0.0)),
        }
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame = frame.merge(panel_slice, on="date", how="left", validate="one_to_one")
    if "y" in frame.columns:
        frame["base_move_h20_pct"] = _safe_return_pct(frame["y"].to_numpy(dtype=float), base_pred[:, 19])
        frame["base_move_h30_pct"] = _safe_return_pct(frame["y"].to_numpy(dtype=float), base_pred[:, 29])
    else:
        frame["base_move_h20_pct"] = np.nan
        frame["base_move_h30_pct"] = np.nan

    frame["helpful_loose"] = (frame["path_uplift_abs"] > 0).astype(int)
    frame["helpful_long_only"] = ((frame["h20_uplift_abs"] > 0) & (frame["h30_uplift_abs"] > 0)).astype(int)
    frame["helpful_strict"] = (
        (frame["path_uplift_pct"] >= 0.01)
        & (frame["h20_uplift_abs"] > 0)
        & (frame["h30_uplift_abs"] > 0)
        & (frame["h5_uplift_pct"].fillna(0.0) >= -0.10)
    ).astype(int)
    frame["harmful_strict"] = (
        (frame["path_uplift_pct"] <= -0.01)
        & ((frame["h20_uplift_abs"] < 0) | (frame["h30_uplift_abs"] < 0))
    ).astype(int)
    return frame


def _write_methodology(out_dir: Path, runs: list[CaseRun]) -> None:
    lines = [
        "# Selective Residual Refinement Methodology",
        "",
        "## Objective",
        "- Replace broad always-on LLM refinement with a scientifically testable selective residual-correction program.",
        "- Learn from realized helpful vs harmful LLM cases before changing the live refiner again.",
        "",
        "## Phase I Scope",
        "- Use like-for-like improved-base UK runs only.",
        f"- Current source runs: {', '.join(run.window for run in runs)}.",
        "- Do not mix in older weaker-base runs when defining helpful regimes.",
        "",
        "## Case-Level Dataset",
        "- One row per forecast origin date.",
        "- Extract realized base and LLM errors from saved prediction arrays.",
        "- Join contemporaneous panel features from the saved run panel.",
        "- Join LLM apply metadata from the saved JSONL call logs.",
        "",
        "## Primary Labels",
        "- `helpful_loose`: path MSE improved.",
        "- `helpful_long_only`: both h20 and h30 improved.",
        "- `helpful_strict`: path uplift >= 1%, h20 and h30 both improved, h5 not damaged by more than 10%.",
        "- `harmful_strict`: path uplift <= -1% and at least one of h20/h30 worsened.",
        "",
        "## Regime Definitions",
        "- Volatility regime: tertiles of `y_vol_20d`.",
        "- Base long-horizon move regime: tertiles of `abs(base_move_h20_pct)`.",
        "- Retrieval-support regime: tertiles of `matched_teaching_count`.",
        "",
        "## Scientific Use",
        "- This phase is descriptive and label-building only.",
        "- No new live policy is fit on the same rows being evaluated.",
        "- The purpose is to define a preregistered helpful-case regime for the next benchmark tranche.",
    ]
    (out_dir / "methodology.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary(out_dir: Path, cases: pd.DataFrame, runs: list[CaseRun]) -> None:
    window_summary = (
        cases.groupby("window")[["path_uplift_pct", "h20_uplift_pct", "h30_uplift_pct", "helpful_loose", "helpful_long_only", "helpful_strict", "harmful_strict"]]
        .mean()
        .reset_index()
    )
    regime_summary = (
        cases.groupby(["vol_regime", "move_regime"])
        .agg(
            n_cases=("date", "size"),
            mean_path_uplift_pct=("path_uplift_pct", "mean"),
            mean_h20_uplift_pct=("h20_uplift_pct", "mean"),
            mean_h30_uplift_pct=("h30_uplift_pct", "mean"),
            helpful_strict_rate=("helpful_strict", "mean"),
            harmful_strict_rate=("harmful_strict", "mean"),
        )
        .reset_index()
        .sort_values(["mean_path_uplift_pct", "helpful_strict_rate"], ascending=[False, False])
    )
    candidate_regimes = regime_summary[(regime_summary["n_cases"] >= 8) & (regime_summary["mean_path_uplift_pct"] > 0)].copy()

    lines = [
        "# Helpful-Case Mining Summary",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Source Runs",
    ]
    for case_run in runs:
        lines.append(f"- `{case_run.window}`: `{case_run.run_dir}`")
    lines.extend(
        [
            "",
            "## Overall",
            f"- Cases: `{len(cases)}`",
            f"- Mean path uplift: `{cases['path_uplift_pct'].mean():.2%}`",
            f"- Mean h20 uplift: `{cases['h20_uplift_pct'].mean():.2%}`",
            f"- Mean h30 uplift: `{cases['h30_uplift_pct'].mean():.2%}`",
            f"- Helpful loose rate: `{cases['helpful_loose'].mean():.2%}`",
            f"- Helpful strict rate: `{cases['helpful_strict'].mean():.2%}`",
            f"- Harmful strict rate: `{cases['harmful_strict'].mean():.2%}`",
            "",
            "## By Window",
            "",
            window_summary.to_markdown(index=False),
            "",
            "## Volatility x Base-Move Regimes",
            "",
            regime_summary.to_markdown(index=False),
            "",
            "## Candidate Helpful Regimes",
            "",
            candidate_regimes.to_markdown(index=False) if not candidate_regimes.empty else "No regime cells met the positive-uplift and minimum-count criteria.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    regime_summary.to_csv(out_dir / "regime_summary.csv", index=False)
    candidate_regimes.to_csv(out_dir / "candidate_helpful_regimes.csv", index=False)
    window_summary.to_csv(out_dir / "window_summary.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--runs-csv", default="")
    args = parser.parse_args()
    runs = _load_runs(args.runs_csv)

    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = (
            PROJECT_ROOT
            / "reports"
            / "uk_ets_selective_residual_helpful_cases"
            / datetime.now().strftime("%Y%m%d_%H%M%S")
        ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_methodology(out_dir, runs)

    cases = pd.concat([_build_case_frame(case_run) for case_run in runs], ignore_index=True)
    cases["vol_regime"] = _assign_quantile_bins(cases["y_vol_20d"], ["low_vol", "mid_vol", "high_vol"])
    cases["move_regime"] = _assign_quantile_bins(cases["base_move_h20_pct"].abs(), ["small_move", "mid_move", "large_move"])
    cases["retrieval_regime"] = _assign_quantile_bins(cases["matched_teaching_count"], ["low_support", "mid_support", "high_support"])

    cases.to_csv(out_dir / "helpful_case_dataset.csv", index=False)
    _write_summary(out_dir, cases, runs)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
