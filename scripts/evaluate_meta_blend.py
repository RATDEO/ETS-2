#!/usr/bin/env python3
"""
Fit a per-horizon ridge meta-blend on validation predictions and apply it to test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eval.meta_blend import MetaBlender, StatefulMetaBlender
from eval.metrics import compute_metrics_by_horizon, compute_path_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate per-horizon ridge meta-blend for TSM + LLM forecasts.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("reports/cot_sent_signal_tune/strategy2_hdelta_small_model_full_20260301"),
        help="Run directory containing stage2_full_val and stage3_test prediction npz files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <run-dir>/meta_blend_eval).",
    )
    return parser.parse_args()


def _metric_row(name: str, y_true: np.ndarray, yhat: np.ndarray) -> dict[str, float | str]:
    metrics = compute_metrics_by_horizon(y_true, yhat, [1, 5, 20, 30]).reset_index()
    row: dict[str, float | str] = {"name": name, "path_mse": float(compute_path_metrics(y_true, yhat)["mse_path"])}
    for _, metric_row in metrics.iterrows():
        row[f"h{int(metric_row['horizon'])}_mse"] = float(metric_row["mse"])
    return row


def _load_base_run(run_dir: Path) -> Path:
    summary_path = run_dir / "selection_summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        base_run = payload.get("base_run")
        if base_run:
            return Path(base_run)
    raise FileNotFoundError(f"Could not infer base_run from {summary_path}")


def _load_state_frame(base_run: Path) -> pd.DataFrame:
    with (base_run / "config_resolved.yaml").open("r") as handle:
        cfg = yaml.safe_load(handle)

    panel = pd.read_parquet(base_run / "data" / "panel.parquet").copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel["ret1"] = np.log(pd.to_numeric(panel["y"], errors="coerce")).diff()
    panel["vol20"] = panel["ret1"].rolling(20, min_periods=5).std().fillna(0.0)

    llm_cfg = (cfg.get("llm", {}) or {})
    sent_cfg = (llm_cfg.get("sentiment", {}) or {})
    sent_path = sent_cfg.get("path", "data/news/daily_sentiment.csv")
    sent_path = Path(sent_path)
    if not sent_path.is_absolute():
        sent_path = ROOT / sent_path
    sent_df = pd.read_csv(sent_path)
    date_col = sent_cfg.get("date_col", "seendate")
    score_col = sent_cfg.get("score_col", "sent_score")
    sent_df["date"] = pd.to_datetime(sent_df[date_col]).dt.normalize()
    sent_df[score_col] = pd.to_numeric(sent_df[score_col], errors="coerce").fillna(0.0)
    sent_df = (
        sent_df.groupby("date", as_index=False)[score_col]
        .mean()
        .rename(columns={score_col: "sent_score"})
    )

    state = panel[["date", "vol20"]].merge(sent_df, on="date", how="left")
    state["sent_score"] = state["sent_score"].fillna(0.0)
    state["sent3"] = state["sent_score"].rolling(3, min_periods=1).mean()
    return state[["date", "vol20", "sent3"]]


def _align_state_features(dates: np.ndarray, state_frame: pd.DataFrame) -> np.ndarray:
    q = pd.DataFrame({"date": pd.to_datetime(dates).normalize()})
    merged = q.merge(state_frame, on="date", how="left")
    merged[["vol20", "sent3"]] = merged[["vol20", "sent3"]].fillna(0.0)
    return merged[["vol20", "sent3"]].to_numpy(dtype=float)


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else (ROOT / args.run_dir)
    output_dir = (
        args.output_dir if args.output_dir is not None else run_dir / "meta_blend_eval"
    )
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    val_npz = np.load(
        run_dir / "stage2_full_val" / "baseline_daily_sentiment" / "val" / "val_predictions.npz",
        allow_pickle=True,
    )
    test_npz = np.load(
        run_dir / "stage3_test" / "baseline_daily_sentiment" / "test" / "test_predictions.npz",
        allow_pickle=True,
    )
    base_run = _load_base_run(run_dir)
    state_frame = _load_state_frame(base_run)

    val_true = np.asarray(val_npz["y_true"], dtype=float)
    val_tsm = np.asarray(val_npz["tsm_pred"], dtype=float)
    val_llm = np.asarray(val_npz["yhat"], dtype=float)

    test_true = np.asarray(test_npz["y_true"], dtype=float)
    test_tsm = np.asarray(test_npz["tsm_pred"], dtype=float)
    test_llm = np.asarray(test_npz["yhat"], dtype=float)
    val_state = _align_state_features(val_npz["dates"], state_frame)
    test_state = _align_state_features(test_npz["dates"], state_frame)

    blender = MetaBlender()
    blender.fit(val_tsm, val_llm, val_true)
    val_blend = blender.predict(val_tsm, val_llm)
    test_blend = blender.predict(test_tsm, test_llm)
    stateful = StatefulMetaBlender()
    stateful.fit(val_tsm, val_llm, val_true, val_state)
    val_stateful = stateful.predict(val_tsm, val_llm, val_state)
    test_stateful = stateful.predict(test_tsm, test_llm, test_state)

    summary_rows = blender.summary_rows()
    pd.DataFrame(summary_rows).to_csv(output_dir / "blend_coefficients.csv", index=False)
    pd.DataFrame(stateful.summary_rows()).to_csv(output_dir / "stateful_blend_coefficients.csv", index=False)

    val_rows = [
        _metric_row("tsm", val_true, val_tsm),
        _metric_row("llm_hdelta", val_true, val_llm),
        _metric_row("meta_blend", val_true, val_blend),
        _metric_row("stateful_meta_blend", val_true, val_stateful),
    ]
    test_rows = [
        _metric_row("tsm", test_true, test_tsm),
        _metric_row("llm_hdelta", test_true, test_llm),
        _metric_row("meta_blend", test_true, test_blend),
        _metric_row("stateful_meta_blend", test_true, test_stateful),
    ]
    pd.DataFrame(val_rows).to_csv(output_dir / "val_comparison.csv", index=False)
    pd.DataFrame(test_rows).to_csv(output_dir / "test_comparison.csv", index=False)

    np.savez_compressed(
        output_dir / "test_meta_blend_predictions.npz",
        y_true=test_true,
        tsm_pred=test_tsm,
        llm_pred=test_llm,
        blend_pred=test_blend,
        stateful_blend_pred=test_stateful,
        dates=test_npz["dates"],
    )

    summary = {
        "run_dir": str(run_dir),
        "val_results": {row["name"]: row for row in val_rows},
        "test_results": {row["name"]: row for row in test_rows},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
