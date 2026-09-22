#!/usr/bin/env python3
"""Run the staged UK ETS LLM refinement remediation ladder and emit per-stage diagnostics."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.run_experiment import blend_forecasts, llm_result_name, run_experiment


CONFIG_PATH = "uk_ets/config/uk_ets_llm_4b_current_default_lasso.yaml"
DATA_DIR = "uk_ets/Data_auto_uk"
METHOD_NAME = "TSM+LLM-COT-RF-HDELTA"
BASELINE_RAW_LINEAR = PROJECT_ROOT / "runs" / "20260325_121940_2367c2"

COMMON_OVERRIDES: dict[str, Any] = {
    "target": {
        "max_date": "2026-03-23",
    },
    "split": {
        "train_end": "2024-06-30",
        "val_end": "2025-06-30",
        "test_end": "2026-03-23",
    },
    "robustness": {
        "subperiods": [
            {
                "name": "post_uk_ets_launch",
                "start": "2021-05-19",
                "end": "2023-12-31",
            },
            {
                "name": "recent",
                "start": "2024-01-01",
                "end": "2026-03-23",
            },
        ]
    },
    "output": {
        "generate_plots": False,
        "generate_paper": False,
        "write_project_paper": False,
        "write_manuscript": False,
    },
}


@dataclass(frozen=True)
class StageSpec:
    name: str
    description: str
    overrides: dict[str, Any]
    use_blend_selection: bool = False


def _merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for key, value in b.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _result_slug(result_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(result_name))


def _candidate_stages() -> list[StageSpec]:
    return [
        StageSpec(
            name="stage1_bias_neutral_guards",
            description="Remove asymmetric long-horizon positive-bias guards while keeping h1/h5 frozen.",
            overrides={
                "hdelta": {
                    "structured_coherence_guards": True,
                    "structured_zero_negative_long_when_h5_positive": False,
                    "structured_zero_h20_negative_when_h30_zero": False,
                    "structured_zero_mixed_long_signs": True,
                    "structured_prefer_positive_long_conflicts": False,
                }
            },
        ),
        StageSpec(
            name="stage2_similarity_retrieval",
            description="Switch from recent_high_error to similarity retrieval and slightly widen the context pool.",
            overrides={
                "llm": {
                    "cot_rf": {
                        "example_selection": "similarity",
                        "lookback_days": 365,
                        "feature_window": 18,
                        "k_examples": 5,
                    }
                }
            },
        ),
        StageSpec(
            name="stage3_apply_skip_gate",
            description="Add a conservative pre-apply no-op gate for low-confidence or conflicting long-horizon guidance.",
            overrides={
                "llm": {
                    "cot_rf": {
                        "apply_skip_gate": {
                            "enabled": True,
                            "required_horizons": [20, 30],
                            "min_confidence": "high",
                            "require_same_nonzero_preferred_sign": True,
                        }
                    }
                }
            },
        ),
        StageSpec(
            name="stage4_blend_grid",
            description="Re-enable validation-tuned blend selection as a safety rail on top of the best prior stage.",
            overrides={
                "llm": {
                    "blend_grid": {
                        "enabled": True,
                        "schedule": "ramp",
                        "weights": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0],
                        "tune_split": "val",
                        "metric": "mse_path",
                    }
                }
            },
            use_blend_selection=True,
        ),
        StageSpec(
            name="stage5_sample_diversity",
            description="Increase reflect/apply sampling and temperature only if earlier stages still fail.",
            overrides={
                "llm": {
                    "cot_rf": {
                        "reflect_samples": 3,
                        "apply_samples": 3,
                        "reflect_temperature": 0.2,
                        "apply_temperature": 0.2,
                    }
                }
            },
        ),
    ]


def _seed_overrides(base_model: str) -> dict[str, Any]:
    return _merge(
        COMMON_OVERRIDES,
        {
        "llm": {
            "base_model": str(base_model),
        },
        "hdelta": {
            "freeze_horizons": [1, 5],
        },
        },
    )


def _model_row(df: pd.DataFrame, model_name: str) -> pd.Series:
    row = df[df["model"] == model_name]
    if row.empty:
        raise KeyError(f"Missing model row for {model_name}")
    return row.iloc[0]


def _effective_predictions(run_dir: Path, base_model: str, use_blend_selection: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    result_name = llm_result_name(METHOD_NAME, base_model)
    with np.load(run_dir / "predictions" / f"{result_name}_pred_test_subset.npz", allow_pickle=True) as data:
        dates = pd.to_datetime(data["dates"]).to_numpy()
        y_true = np.asarray(data["y_true"], dtype=float)
        base_pred = np.asarray(data["base_pred"], dtype=float)
        llm_pred = np.asarray(data["yhat"], dtype=float)

    blend_meta: dict[str, Any] = {"mode": "raw"}
    pred = llm_pred
    if use_blend_selection:
        selection_path = run_dir / "llm" / f"blend_grid_selection_{_result_slug(result_name)}.json"
        payload = json.loads(selection_path.read_text())
        weight = float(payload["best_w_path"])
        schedule = str(payload.get("schedule", "ramp"))
        min_weight = float(payload.get("min_weight", 0.0))
        power = float(payload.get("power", 1.0))
        pred = blend_forecasts(
            base_pred=base_pred,
            llm_pred=llm_pred,
            strength=weight,
            schedule=schedule,
            pred_len=base_pred.shape[1],
            min_weight=min_weight,
            power=power,
        )
        blend_meta = {
            "mode": "blend",
            "weight": weight,
            "schedule": schedule,
            "min_weight": min_weight,
            "power": power,
        }
    return dates, y_true, base_pred, pred, blend_meta


def _load_adjustment_pattern_stats(run_dir: Path) -> dict[str, Any]:
    test_meta_path = next(iter(sorted(run_dir.glob("llm/*_test_metadata.jsonl"))), None)
    gate_skip_count = 0
    if test_meta_path and test_meta_path.exists():
        for line in test_meta_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if bool(row.get("apply_skipped_by_gate", False)):
                gate_skip_count += 1

    log_path = run_dir / "llm" / "logs" / "llm_calls.jsonl"
    if not log_path.exists():
        return {
            "dominant_apply_pattern_share": None,
            "dominant_apply_pattern": None,
            "apply_skip_count": gate_skip_count,
        }

    apply_rows = []
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("method", "")).endswith(":apply"):
            apply_rows.append(row)
    if not apply_rows:
        return {
            "dominant_apply_pattern_share": None,
            "dominant_apply_pattern": None,
            "apply_skip_count": gate_skip_count,
        }

    pattern_counts: dict[tuple[tuple[str, float | None], ...], int] = {}
    skip_count = 0
    for row in apply_rows:
        try:
            content = json.loads(row.get("response", {}).get("content", "{}"))
        except Exception:
            content = {}
        if content.get("skipped"):
            skip_count += 1
            continue
        agg = content.get("aggregated_adjustments", {}) or {}
        key = (
            ("h1", agg.get("1")),
            ("h5", agg.get("5")),
            ("h20", agg.get("20")),
            ("h30", agg.get("30")),
        )
        pattern_counts[key] = pattern_counts.get(key, 0) + 1
    dominant_pattern = None
    dominant_share = None
    if pattern_counts:
        dominant_pattern, dominant_count = max(pattern_counts.items(), key=lambda item: item[1])
        dominant_share = float(dominant_count / max(1, len(apply_rows) - skip_count))
    return {
        "dominant_apply_pattern_share": dominant_share,
        "dominant_apply_pattern": dominant_pattern,
        "apply_skip_count": int(max(skip_count, gate_skip_count)),
    }


def _stage_summary(
    run_dir: Path,
    stage: StageSpec,
    base_model: str,
    prior_stage_path_mse: float | None,
    template_share_baseline: float | None,
) -> dict[str, Any]:
    path_df = pd.read_csv(run_dir / "results" / "path_metrics.csv")
    horizon_df = pd.read_csv(run_dir / "results" / "metrics_by_horizon.csv")
    ls_df = pd.read_csv(run_dir / "results" / "long_short_metrics.csv")

    result_name = llm_result_name(METHOD_NAME, base_model)
    evaluated_model = (
        f"{result_name}_blend_ramp_bestval_path" if stage.use_blend_selection else result_name
    )
    base_model_row = _model_row(path_df, base_model)
    eval_row = _model_row(path_df, evaluated_model)
    base_h = horizon_df[horizon_df["model"] == base_model].set_index("horizon")
    eval_h = horizon_df[horizon_df["model"] == evaluated_model].set_index("horizon")
    base_ls = ls_df[ls_df["model"] == base_model].set_index("horizon")
    eval_ls = ls_df[ls_df["model"] == evaluated_model].set_index("horizon")

    dates, y_true, base_pred, pred, blend_meta = _effective_predictions(
        run_dir,
        base_model,
        stage.use_blend_selection,
    )
    path_se_base = ((base_pred - y_true) ** 2).mean(axis=1)
    path_se_pred = ((pred - y_true) ** 2).mean(axis=1)
    path_delta = path_se_pred - path_se_base
    top_harm_idx = np.argsort(path_delta)[-5:][::-1]
    harmful_dates = [
        {
            "date": str(pd.Timestamp(dates[idx]).date()),
            "path_harm": float(path_delta[idx]),
            "h20_adjustment": float(pred[idx, 19] - base_pred[idx, 19]),
            "h30_adjustment": float(pred[idx, 29] - base_pred[idx, 29]),
        }
        for idx in top_harm_idx
    ]

    pattern_stats = _load_adjustment_pattern_stats(run_dir)
    summary = {
        "stage": stage.name,
        "description": stage.description,
        "run_dir": str(run_dir),
        "base_model": base_model,
        "evaluated_model": evaluated_model,
        "result_name": result_name,
        "blend_meta": blend_meta,
        "path_mse": float(eval_row["mse_path"]),
        "base_path_mse": float(base_model_row["mse_path"]),
        "delta_vs_base": float(eval_row["mse_path"] - base_model_row["mse_path"]),
        "delta_vs_prior_stage": (
            None if prior_stage_path_mse is None else float(eval_row["mse_path"] - prior_stage_path_mse)
        ),
        "mse_h5": float(eval_h.loc[5, "mse"]),
        "mse_h20": float(eval_h.loc[20, "mse"]),
        "mse_h30": float(eval_h.loc[30, "mse"]),
        "base_mse_h5": float(base_h.loc[5, "mse"]),
        "base_mse_h20": float(base_h.loc[20, "mse"]),
        "base_mse_h30": float(base_h.loc[30, "mse"]),
        "sharpe_h20": float(eval_ls.loc[20, "sharpe_non_overlap_offset_avg"]),
        "sharpe_h30": float(eval_ls.loc[30, "sharpe_non_overlap_offset_avg"]),
        "base_sharpe_h20": float(base_ls.loc[20, "sharpe_non_overlap_offset_avg"]),
        "base_sharpe_h30": float(base_ls.loc[30, "sharpe_non_overlap_offset_avg"]),
        "mean_h20_adjustment": float(np.mean(pred[:, 19] - base_pred[:, 19])),
        "mean_h30_adjustment": float(np.mean(pred[:, 29] - base_pred[:, 29])),
        "h20_positive_frac": float(np.mean((pred[:, 19] - base_pred[:, 19]) > 0)),
        "h20_negative_frac": float(np.mean((pred[:, 19] - base_pred[:, 19]) < 0)),
        "h20_zero_frac": float(np.mean(np.isclose(pred[:, 19] - base_pred[:, 19], 0.0))),
        "h30_positive_frac": float(np.mean((pred[:, 29] - base_pred[:, 29]) > 0)),
        "h30_negative_frac": float(np.mean((pred[:, 29] - base_pred[:, 29]) < 0)),
        "h30_zero_frac": float(np.mean(np.isclose(pred[:, 29] - base_pred[:, 29], 0.0))),
        "top_harmful_dates": harmful_dates,
        **pattern_stats,
    }
    dominant_share = summary["dominant_apply_pattern_share"]
    no_new_template_collapse = True
    if dominant_share is not None and template_share_baseline is not None:
        no_new_template_collapse = float(dominant_share) <= float(template_share_baseline + 0.05)
    summary["keep"] = bool(
        summary["path_mse"] < summary["base_path_mse"]
        and summary["sharpe_h20"] >= summary["base_sharpe_h20"] - 0.05
        and summary["sharpe_h30"] >= summary["base_sharpe_h30"] - 0.05
        and no_new_template_collapse
    )
    summary["keep_reason"] = (
        "beat_base_and_sharpe_ok" if summary["keep"] else "failed_keep_criteria"
    )
    return summary


def _write_summary(out_dir: Path, records: list[dict[str, Any]]) -> None:
    json_path = out_dir / "stage_results.json"
    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    rows = []
    for record in records:
        rows.append(
            {
                "stage": record["stage"],
                "base_model": record["base_model"],
                "run_dir": record["run_dir"],
                "path_mse": record["path_mse"],
                "base_path_mse": record["base_path_mse"],
                "delta_vs_base": record["delta_vs_base"],
                "delta_vs_prior_stage": record["delta_vs_prior_stage"],
                "mse_h5": record["mse_h5"],
                "mse_h20": record["mse_h20"],
                "mse_h30": record["mse_h30"],
                "sharpe_h20": record["sharpe_h20"],
                "sharpe_h30": record["sharpe_h30"],
                "mean_h20_adjustment": record["mean_h20_adjustment"],
                "mean_h30_adjustment": record["mean_h30_adjustment"],
                "dominant_apply_pattern_share": record["dominant_apply_pattern_share"],
                "apply_skip_count": record["apply_skip_count"],
                "keep": record["keep"],
                "keep_reason": record["keep_reason"],
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "stage_results.csv", index=False)


def _print_stage_report(summary: dict[str, Any], stage_overrides: dict[str, Any]) -> None:
    print(f"\n=== {summary['stage']} ({summary['base_model']}) ===", flush=True)
    print(summary["description"], flush=True)
    print("overrides=", json.dumps(stage_overrides, sort_keys=True), flush=True)
    path_line = (
        f"path_mse={summary['path_mse']:.6f} "
        f"(base={summary['base_path_mse']:.6f}, "
        f"delta_vs_base={summary['delta_vs_base']:+.6f}"
    )
    if summary["delta_vs_prior_stage"] is not None:
        path_line += f", delta_vs_prior={summary['delta_vs_prior_stage']:+.6f}"
    path_line += ")"
    print(path_line, flush=True)
    print(
        "mse[h5,h20,h30]="
        f"{summary['mse_h5']:.6f}, {summary['mse_h20']:.6f}, {summary['mse_h30']:.6f}",
        flush=True,
    )
    print(
        "sharpe[h20,h30]="
        f"{summary['sharpe_h20']:.6f}, {summary['sharpe_h30']:.6f}",
        flush=True,
    )
    print(
        "mean_adj[h20,h30]="
        f"{summary['mean_h20_adjustment']:+.6f}, {summary['mean_h30_adjustment']:+.6f}",
        flush=True,
    )
    print(
        "h20[pos/neg/zero]="
        f"{summary['h20_positive_frac']:.3f}/{summary['h20_negative_frac']:.3f}/{summary['h20_zero_frac']:.3f} "
        "h30[pos/neg/zero]="
        f"{summary['h30_positive_frac']:.3f}/{summary['h30_negative_frac']:.3f}/{summary['h30_zero_frac']:.3f}",
        flush=True,
    )
    print(
        "dominant_pattern_share="
        f"{summary['dominant_apply_pattern_share'] if summary['dominant_apply_pattern_share'] is not None else 'n/a'} "
        f"apply_skip_count={summary['apply_skip_count']}",
        flush=True,
    )
    print("top_harmful_dates=", json.dumps(summary["top_harmful_dates"], indent=2), flush=True)
    print(f"keep={summary['keep']} reason={summary['keep_reason']}", flush=True)


def _run_stage(stage: StageSpec, base_model: str, overrides: dict[str, Any]) -> Path:
    run_dir = Path(
        run_experiment(
            config_path=CONFIG_PATH,
            overrides=overrides,
            data_dir=DATA_DIR,
        )
    ).resolve()
    return run_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--no-stop-on-win", action="store_true")
    parser.add_argument("--seed-run-dir", default="")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = (
            PROJECT_ROOT / "reports" / "uk_ets_llm_refinement_remediation" / datetime.now().strftime("%Y%m%d_%H%M%S")
        ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    raw_stage = StageSpec(
        name="baseline_current_llm",
        description="Existing linear_lasso current LLM baseline.",
        overrides={"llm": {"base_model": "linear_lasso"}},
    )
    raw_summary = _stage_summary(
        BASELINE_RAW_LINEAR,
        raw_stage,
        "linear_lasso",
        prior_stage_path_mse=None,
        template_share_baseline=None,
    )
    raw_summary["run_dir"] = str(BASELINE_RAW_LINEAR)
    records.append(raw_summary)
    _print_stage_report(raw_summary, raw_stage.overrides)

    seed_stage = StageSpec(
        name="seed_freeze_baseline",
        description="Fresh linear_lasso freeze-h1/h5 seed baseline used for staged remediation.",
        overrides=_seed_overrides("linear_lasso"),
    )
    if args.seed_run_dir:
        seed_run_dir = Path(args.seed_run_dir).expanduser().resolve()
    else:
        seed_run_dir = _run_stage(seed_stage, "linear_lasso", seed_stage.overrides)
    seed_summary = _stage_summary(
        seed_run_dir,
        seed_stage,
        "linear_lasso",
        prior_stage_path_mse=float(raw_summary["path_mse"]),
        template_share_baseline=float(raw_summary["dominant_apply_pattern_share"] or 0.0),
    )
    records.append(seed_summary)
    _print_stage_report(seed_summary, seed_stage.overrides)

    cumulative_overrides = _seed_overrides("linear_lasso")
    best_linear_summary = seed_summary
    best_linear_stage: tuple[StageSpec, dict[str, Any], dict[str, Any]] | None = None
    prior_stage_path_mse = float(seed_summary["path_mse"])
    raw_template_share_baseline = raw_summary["dominant_apply_pattern_share"]
    stop_on_win = not args.no_stop_on_win
    last_run_dir = seed_run_dir

    for stage in _candidate_stages():
        cumulative_overrides = _merge(cumulative_overrides, stage.overrides)
        stage_overrides = _merge(
            cumulative_overrides,
            {"llm": {"cache_seed_run_dirs": [str(last_run_dir)]}},
        )
        run_dir = _run_stage(stage, "linear_lasso", stage_overrides)
        summary = _stage_summary(
            run_dir,
            stage,
            "linear_lasso",
            prior_stage_path_mse=prior_stage_path_mse,
            template_share_baseline=raw_template_share_baseline,
        )
        records.append(summary)
        _print_stage_report(summary, stage.overrides)

        if summary["path_mse"] < best_linear_summary["path_mse"]:
            best_linear_summary = summary
            best_linear_stage = (stage, dict(cumulative_overrides), summary)
        elif best_linear_stage is None or summary["path_mse"] < best_linear_stage[2]["path_mse"]:
            best_linear_stage = (stage, dict(cumulative_overrides), summary)

        prior_stage_path_mse = float(summary["path_mse"])
        last_run_dir = run_dir

        _write_summary(out_dir, records)
        if summary["keep"] and stop_on_win:
            break

    if best_linear_stage is not None:
        stage, cumulative, linear_summary = best_linear_stage
        tsm_overrides = _merge(
            _merge(_seed_overrides("tsm"), cumulative),
            {"llm": {"cache_seed_run_dirs": [str(last_run_dir)]}},
        )
        tsm_stage = StageSpec(
            name=f"{stage.name}_tsm_transfer",
            description=f"Transfer of {stage.name} to the tsm base model.",
            overrides=tsm_overrides,
            use_blend_selection=stage.use_blend_selection,
        )
        run_dir = _run_stage(tsm_stage, "tsm", tsm_overrides)
        summary = _stage_summary(
            run_dir,
            tsm_stage,
            "tsm",
            prior_stage_path_mse=None,
            template_share_baseline=raw_template_share_baseline,
        )
        records.append(summary)
        _print_stage_report(summary, tsm_overrides)
        _write_summary(out_dir, records)

    print(f"\nsummary_dir={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
