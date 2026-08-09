#!/usr/bin/env python3
"""Build a causally aligned v2 helpful/harmful refinement dataset.

The legacy builder joined panel values on the prediction start date and exposed
future-error columns such as ``base_bias_h20``. This wrapper keeps the verified
outcomes/LLM metadata but replaces every market feature with the most recent
strictly earlier panel row, recomputes base moves from that last observed price,
and removes future-error columns from the saved dataset.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uk_ets.scripts import build_selective_residual_helpful_case_dataset as legacy  # noqa: E402

FUTURE_ERROR_COLUMNS = (
    "base_bias_h20",
    "base_bias_h30",
    "llm_bias_h20",
    "llm_bias_h30",
)


def attach_last_observed_features(
    frame: pd.DataFrame,
    panel: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """Attach the latest panel row strictly before each prediction start date."""
    left = frame[["date"]].copy()
    left["_row_order"] = np.arange(len(left))
    left["date"] = pd.to_datetime(left["date"], errors="raise")
    left = left.rename(columns={"date": "forecast_date"}).sort_values("forecast_date")

    available = [col for col in feature_columns if col in panel.columns]
    right = panel[["date"] + available].copy()
    right["date"] = pd.to_datetime(right["date"], errors="raise")
    right = right.sort_values("date").drop_duplicates("date", keep="last")
    right = right.rename(columns={"date": "feature_date", **{c: f"asof_{c}" for c in available}})

    merged = pd.merge_asof(
        left,
        right,
        left_on="forecast_date",
        right_on="feature_date",
        direction="backward",
        allow_exact_matches=False,
    ).sort_values("_row_order")
    if merged["feature_date"].isna().any():
        raise ValueError("At least one forecast origin has no strictly earlier panel row")
    if not (merged["feature_date"] < merged["forecast_date"]).all():
        raise AssertionError("Feature alignment leaked a forecast-date or future panel row")
    merged["asof_age_days"] = (merged["forecast_date"] - merged["feature_date"]).dt.days
    return merged.drop(columns="_row_order").reset_index(drop=True)


def rebuild_case_frame(case_run: legacy.CaseRun) -> pd.DataFrame:
    # The legacy frame is used only for verified saved outcomes and LLM-call
    # metadata. Every contemporaneous market feature and all future-error
    # columns are removed before output.
    frame = legacy._build_case_frame(case_run).reset_index(drop=True)
    panel = pd.read_parquet(case_run.run_dir / "data" / "panel.parquet")
    safe = attach_last_observed_features(frame, panel, legacy.FEATURE_COLUMNS)

    drop_columns = list(legacy.FEATURE_COLUMNS) + list(FUTURE_ERROR_COLUMNS) + [
        "base_move_h20_pct",
        "base_move_h30_pct",
        "run_dir",
    ]
    result = frame.drop(columns=drop_columns, errors="ignore").copy()
    safe_columns = [c for c in safe.columns if c not in {"forecast_date"}]
    result = pd.concat([result.reset_index(drop=True), safe[safe_columns].reset_index(drop=True)], axis=1)
    result["date"] = safe["forecast_date"].to_numpy()
    result["run_id"] = case_run.run_dir.name

    prediction_path = case_run.run_dir / "predictions" / "TSM+LLM-COT-RF-HDELTA_pred_test_subset.npz"
    with np.load(prediction_path, allow_pickle=True) as payload:
        saved_dates = pd.to_datetime(payload["dates"])
        base_pred = np.asarray(payload["base_pred"], dtype=float)
    if not np.array_equal(saved_dates.to_numpy(), pd.to_datetime(result["date"]).to_numpy()):
        raise AssertionError("Prediction-array dates no longer align with case rows")
    if "asof_y" not in result:
        raise ValueError("Corrected dataset requires the last observed target price as asof_y")
    current = result["asof_y"].to_numpy(dtype=float)
    result["base_move_h20_pct"] = legacy._safe_return_pct(current, base_pred[:, 19])
    result["base_move_h30_pct"] = legacy._safe_return_pct(current, base_pred[:, 29])

    leaked = set(FUTURE_ERROR_COLUMNS).intersection(result.columns)
    if leaked:
        raise AssertionError(f"Future-error columns survived v2 sanitisation: {sorted(leaked)}")
    if not (pd.to_datetime(result["feature_date"]) < pd.to_datetime(result["date"])).all():
        raise AssertionError("Saved v2 rows violate feature_date < forecast date")
    return result


def write_methodology(out_dir: Path, runs: Sequence[legacy.CaseRun]) -> None:
    lines = [
        "# Selective residual dataset v2 methodology",
        "",
        "## Corrections",
        "",
        "- Market features are attached from the most recent panel row strictly before the prediction start date (`feature_date < date`).",
        "- Every lag-safe market column is namespaced with `asof_`.",
        "- `base_move_h20_pct` and `base_move_h30_pct` use `asof_y`, not the first realised forecast-day price.",
        "- Future-error fields (`base_bias_h20`, `base_bias_h30`, `llm_bias_h20`, `llm_bias_h30`) are not saved as predictors.",
        "- Absolute local paths are replaced with a portable `run_id`.",
        "",
        "## Evaluation warning",
        "",
        "Only historical rows where the previous system actually produced an LLM adjustment reveal that adjustment's counterfactual. A pre-call gate evaluated on this dataset is therefore conditional on the old opportunity set.",
        "",
        "## Source runs",
        "",
    ]
    lines.extend(f"- {run.window}: `{run.run_dir.name}`" for run in runs)
    (out_dir / "methodology.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-csv", type=str, default="")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/agent_selective_dataset_v2"),
    )
    args = parser.parse_args()
    runs = legacy._load_runs(args.runs_csv)
    frames = [rebuild_case_frame(run) for run in runs]
    dataset = pd.concat(frames, ignore_index=True).sort_values(["date", "window"]).reset_index(drop=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.out_dir / "helpful_case_dataset_v2.csv", index=False)
    write_methodology(args.out_dir, runs)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(dataset),
        "windows": dataset.groupby("window").size().to_dict(),
        "min_date": str(pd.to_datetime(dataset["date"]).min().date()),
        "max_date": str(pd.to_datetime(dataset["date"]).max().date()),
        "alignment": "strictly previous panel row",
        "removed_future_error_columns": list(FUTURE_ERROR_COLUMNS),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
