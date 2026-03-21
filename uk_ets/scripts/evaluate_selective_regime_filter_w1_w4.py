#!/usr/bin/env python3
"""Evaluate leave-one-window-out selective regime filters for UK ETS refinement."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "uk_ets_selective_residual_helpful_cases"
    / "20260321_180909"
)
SOURCE_DATASET = SOURCE_REPORT / "helpful_case_dataset.csv"


@dataclass(frozen=True)
class WindowRun:
    window: str
    run_dir: Path


RUNS = [
    WindowRun("W1", PROJECT_ROOT / "runs" / "20260320_182015_da9e9d"),
    WindowRun("W2", PROJECT_ROOT / "runs" / "20260320_183041_c7484b"),
    WindowRun("W3", PROJECT_ROOT / "runs" / "20260320_183637_a4d80c"),
    WindowRun("W4", PROJECT_ROOT / "runs" / "20260320_184725_2cded6"),
]


def _safe_pct(base: float, refined: float) -> float:
    if abs(base) <= 1e-12:
        return np.nan
    return 1.0 - (refined / base)


def _window_predictions(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(run_dir / "predictions" / "TSM+LLM-COT-RF-HDELTA_pred_test_subset.npz", allow_pickle=True) as data:
        return (
            pd.to_datetime(data["dates"]).normalize().to_numpy(),
            data["y_true"].astype(float),
            data["base_pred"].astype(float),
            data["yhat"].astype(float),
        )


def _select_vm_regimes(train_df: pd.DataFrame) -> set[tuple[str, str]]:
    summary = (
        train_df.groupby(["vol_regime", "move_regime"], observed=False)
        .agg(
            n_cases=("date", "size"),
            mean_path_uplift_abs=("path_uplift_abs", "mean"),
            helpful_strict_rate=("helpful_strict", "mean"),
            harmful_strict_rate=("harmful_strict", "mean"),
        )
        .reset_index()
    )
    selected = summary[
        (summary["n_cases"] >= 20)
        & (summary["mean_path_uplift_abs"] > 0.0)
        & (summary["helpful_strict_rate"] >= summary["harmful_strict_rate"])
    ]
    return set(zip(selected["vol_regime"], selected["move_regime"]))


def _select_vmr_regimes(train_df: pd.DataFrame) -> set[tuple[str, str, str]]:
    summary = (
        train_df.groupby(["vol_regime", "move_regime", "retrieval_regime"], observed=False)
        .agg(
            n_cases=("date", "size"),
            mean_path_uplift_abs=("path_uplift_abs", "mean"),
            helpful_strict_rate=("helpful_strict", "mean"),
            harmful_strict_rate=("harmful_strict", "mean"),
        )
        .reset_index()
    )
    selected = summary[
        (summary["n_cases"] >= 12)
        & (summary["mean_path_uplift_abs"] > 0.0)
        & (summary["helpful_strict_rate"] > summary["harmful_strict_rate"])
    ]
    return set(zip(selected["vol_regime"], selected["move_regime"], selected["retrieval_regime"]))


def _evaluate_variant(
    name: str,
    cases: pd.DataFrame,
    selector,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    run_map = {run.window: run for run in RUNS}

    for target_window in [run.window for run in RUNS]:
        train_df = cases[cases["window"] != target_window].copy()
        test_df = cases[cases["window"] == target_window].copy()
        selected = selector(train_df)

        dates, y_true, base_pred, llm_pred = _window_predictions(run_map[target_window].run_dir)
        test_df = test_df.copy()
        test_df["date"] = pd.to_datetime(test_df["date"]).dt.normalize()
        pred_frame = pd.DataFrame(
            {
                "date": dates,
                "case_idx": np.arange(len(dates)),
            }
        )
        test_df = test_df.merge(pred_frame, on="date", how="left", validate="one_to_one").sort_values("case_idx")
        if test_df["case_idx"].isna().any():
            raise RuntimeError(f"Missing prediction alignment in {target_window}")
        case_idx = test_df["case_idx"].astype(int).to_numpy()

        if name == "vm_regime_filter":
            allow_mask = np.array(
                [(row.vol_regime, row.move_regime) in selected for row in test_df.itertuples()],
                dtype=bool,
            )
        else:
            allow_mask = np.array(
                [
                    (row.vol_regime, row.move_regime, row.retrieval_regime) in selected
                    for row in test_df.itertuples()
                ],
                dtype=bool,
            )

        gated_pred = llm_pred.copy()
        gated_pred[case_idx[~allow_mask]] = base_pred[case_idx[~allow_mask]]

        base_path = float(np.mean((base_pred - y_true) ** 2))
        llm_path = float(np.mean((llm_pred - y_true) ** 2))
        gated_path = float(np.mean((gated_pred - y_true) ** 2))
        rows.append(
            {
                "variant": name,
                "window": target_window,
                "n_test_cases": int(len(test_df)),
                "selected_regime_count": int(len(selected)),
                "apply_share_after_filter": float(np.mean(allow_mask & (test_df["llm_applied"].to_numpy(dtype=int) == 1))),
                "llm_apply_share_before_filter": float(np.mean(test_df["llm_applied"].to_numpy(dtype=int) == 1)),
                "base_path_mse": base_path,
                "wide8_path_mse": llm_path,
                "filtered_path_mse": gated_path,
                "uplift_vs_base_pct": _safe_pct(base_path, gated_path),
                "uplift_vs_wide8_pct": _safe_pct(llm_path, gated_path),
            }
        )
        for regime in sorted(selected):
            selection_rows.append(
                {
                    "variant": name,
                    "target_window": target_window,
                    "selected_regime": "|".join(regime),
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(selection_rows)


def _write_plan(out_dir: Path) -> None:
    lines = [
        "# Selective Regime Filter Plan",
        "",
        "## Objective",
        "- Test whether a preregistered selective regime filter can preserve the positive residual cases from `effective_retrieval_wide8` while suppressing harmful ones.",
        "- Evaluate scientifically via leave-one-window-out selection.",
        "",
        "## Protocol",
        "- Use the completed improved-base `wide8` W1-W4 runs only.",
        "- For each held-out window, derive helpful regimes from the other three windows only.",
        "- Apply the frozen regime filter offline to the held-out saved predictions by reverting non-selected cases from LLM output back to raw base output.",
        "",
        "## Variants",
        "- `vm_regime_filter`: select on `(vol_regime, move_regime)`.",
        "- `vmr_regime_filter`: select on `(vol_regime, move_regime, retrieval_regime)`.",
        "",
        "## Selection Rule",
        "- Positive mean path uplift in training windows.",
        "- Helpful strict rate at least as high as harmful strict rate.",
        "- Minimum support threshold: `20` cases for VM, `12` cases for VMR.",
        "",
        "## Expected Outcomes",
        "- `vm_regime_filter`: expected to improve mean W1-W4 path MSE by `0.1%` to `0.5%` vs `wide8`; likely still close to the improved base.",
        "- `vmr_regime_filter`: higher variance; may help `W2/W4` more, but may overfit and lose `W1/W3` support.",
        "",
        "## Primary Endpoint",
        "- Mean W1-W4 path MSE versus both the improved base and the current always-on `wide8` refiner.",
    ]
    (out_dir / "plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary(out_dir: Path, results: pd.DataFrame) -> None:
    overall = (
        results.groupby("variant", as_index=False)[
            ["base_path_mse", "wide8_path_mse", "filtered_path_mse", "uplift_vs_base_pct", "uplift_vs_wide8_pct", "apply_share_after_filter"]
        ]
        .mean()
    )
    lines = [
        "# Selective Regime Filter Summary",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Overall",
        "",
        overall.to_markdown(index=False),
        "",
        "## By Window",
        "",
        results.to_markdown(index=False),
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    out_dir = (
        PROJECT_ROOT
        / "reports"
        / "uk_ets_selective_regime_filter_w1_w4"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_plan(out_dir)

    cases = pd.read_csv(SOURCE_DATASET)
    cases["date"] = pd.to_datetime(cases["date"]).dt.normalize()

    vm_results, vm_selections = _evaluate_variant("vm_regime_filter", cases, _select_vm_regimes)
    vmr_results, vmr_selections = _evaluate_variant("vmr_regime_filter", cases, _select_vmr_regimes)

    results = pd.concat([vm_results, vmr_results], ignore_index=True)
    selections = pd.concat([vm_selections, vmr_selections], ignore_index=True)
    results.to_csv(out_dir / "results.csv", index=False)
    selections.to_csv(out_dir / "selected_regimes.csv", index=False)
    _write_summary(out_dir, results)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
