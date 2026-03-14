#!/usr/bin/env python3
"""Run one full automated pipeline pass: data refresh + experiment."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.run_experiment import run_experiment

DEFAULT_LLM_API_KEY = "deo"
DEFAULT_LOCAL_LLM_BASE_URL = "http://192.168.1.140:9877/v1"


def _run_bootstrap(
    output_dir: Path,
    seed_data_dir: Path,
    include_news: bool,
) -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "bootstrap_data_sources.py"),
        "--output-dir",
        str(output_dir),
        "--seed-data-dir",
        str(seed_data_dir),
    ]
    if include_news:
        cmd.append("--include-news")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def _resolve_seed_data_dir(seed_data_dir: Path, data_dir: Path) -> Path:
    """Resolve seed directory for bootstrap with a self-contained fallback."""
    if seed_data_dir.exists():
        return seed_data_dir
    if data_dir.exists():
        print(
            f"WARNING: seed data dir not found ({seed_data_dir}). "
            f"Falling back to existing data dir as seed source: {data_dir}"
        )
        return data_dir
    raise FileNotFoundError(
        "No seed data directory available for bootstrap. "
        "Provide --seed-data-dir, or create --data-dir with a prior snapshot."
    )


def _resolve_llm_api_key(cli_key: str | None) -> str:
    """Prefer explicit key, then env var, then backward-compatible local default."""
    if cli_key:
        return cli_key
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        return env_key
    return DEFAULT_LLM_API_KEY


def _read_performance_summary(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_path = run_dir / "results" / "metrics_by_horizon.csv"
    path_path = run_dir / "results" / "path_metrics.csv"

    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    path_metrics = pd.read_csv(path_path) if path_path.exists() else pd.DataFrame()
    return metrics, path_metrics


def _print_summary(run_dir: Path, metrics: pd.DataFrame, path_metrics: pd.DataFrame) -> None:
    print("\n=== Automated Pipeline Run Complete ===")
    print(f"Run directory: {run_dir}")

    if not path_metrics.empty:
        keep_models = ["naive_persistence", "seasonal_naive", "linear_ridge", "linear_lasso", "tsm"]
        subset = path_metrics[path_metrics["model"].isin(keep_models)].copy()
        subset = subset.sort_values("mse_path")
        print("\nPath metrics (full subset):")
        print(subset[["model", "n_samples", "mse_path", "rmse_path", "mae_path"]].to_string(index=False))

    if not metrics.empty:
        horizons = [1, 5, 20, 30]
        keep_models = ["naive_persistence", "seasonal_naive", "linear_ridge", "linear_lasso", "tsm"]
        subset = metrics[
            (metrics["model"].isin(keep_models)) & (metrics["horizon"].isin(horizons))
        ].copy()
        subset = subset.sort_values(["model", "horizon"])
        print("\nHorizon MSE (h=1,5,20,30):")
        print(subset[["model", "horizon", "mse", "rmse", "mae"]].to_string(index=False))

    if not path_metrics.empty:
        naive = path_metrics.loc[path_metrics["model"] == "naive_persistence", "mse_path"]
        tsm = path_metrics.loc[path_metrics["model"] == "tsm", "mse_path"]
        if len(naive) and len(tsm):
            ratio = float(tsm.iloc[0]) / float(naive.iloc[0]) if float(naive.iloc[0]) != 0 else float("inf")
            print(f"\nTSM/Naive path-MSE ratio: {ratio:.3f}")
            if ratio > 2.0:
                print("WARNING: TSM underperforming naive baseline. Use --target-mode returns.")


def _prefer_subset(df: pd.DataFrame) -> pd.DataFrame:
    subset_rank = {"full": 0, "llm_subset": 1}
    out = df.copy()
    if "subset" in out.columns:
        out["_subset_rank"] = out["subset"].map(subset_rank).fillna(2)
    else:
        out["_subset_rank"] = 2
    if "n_samples" not in out.columns:
        out["n_samples"] = 0
    return out.sort_values(["_subset_rank", "n_samples"], ascending=[True, False])


def _first_available_model(candidates: list[str], available_models: set[str]) -> str:
    for model_name in candidates:
        if model_name in available_models:
            return model_name
    return candidates[0]


def _build_option_comparison(run_dir: Path, metrics: pd.DataFrame, path_metrics: pd.DataFrame) -> Path | None:
    if metrics.empty and path_metrics.empty:
        return None

    available_models: set[str] = set()
    if not path_metrics.empty and "model" in path_metrics.columns:
        available_models.update(path_metrics["model"].astype(str).tolist())
    if not metrics.empty and "model" in metrics.columns:
        available_models.update(metrics["model"].astype(str).tolist())

    option4_primary_candidates = [
        "TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base",
        "TSM+LLM-COT-RF_blend_ramp_bestval_path",
    ]

    option_model_map = {
        "Option1_Autoformer": "tsm",
        "Option2_CoT_RF": "TSM+LLM-COT-RF",
        "Option3_CoT_SENT_RF": "TSM+LLM-COT-SENT-RF",
        "Option4_CoT_RAMP": _first_available_model(option4_primary_candidates, available_models),
    }

    # If CoT-RF blend wasn't run, allow CoT-SENT-RF blend as fallback Option 4.
    if option_model_map["Option4_CoT_RAMP"] not in available_models:
        sent_blend_candidates = [
            "TSM+LLM-COT-SENT-RF_blend_ramp_bestval_path_h1base",
            "TSM+LLM-COT-SENT-RF_blend_ramp_bestval_path",
        ]
        sent_blend = _first_available_model(sent_blend_candidates, available_models)
        if sent_blend in available_models:
            option_model_map["Option4_CoT_RAMP"] = sent_blend

    rows: list[dict] = []
    for option_name, model_name in option_model_map.items():
        row: dict = {"option": option_name, "model": model_name}

        if not path_metrics.empty:
            pm = path_metrics[path_metrics["model"] == model_name]
            if not pm.empty:
                pm = _prefer_subset(pm).iloc[0]
                row["subset"] = pm.get("subset", "")
                row["n_samples"] = int(pm.get("n_samples", 0))
                row["path_mse"] = float(pm.get("mse_path"))

        if not metrics.empty:
            model_metrics = metrics[metrics["model"] == model_name]
            for h in (1, 5, 20, 30):
                mh = model_metrics[model_metrics["horizon"] == h]
                if mh.empty:
                    continue
                mh = _prefer_subset(mh).iloc[0]
                if "subset" not in row:
                    row["subset"] = mh.get("subset", "")
                if "n_samples" not in row:
                    row["n_samples"] = int(mh.get("n_samples", 0))
                row[f"h{h}_mse"] = float(mh.get("mse"))

        # Keep rows only when at least one metric exists.
        metric_keys = {"path_mse", "h1_mse", "h5_mse", "h20_mse", "h30_mse"}
        if metric_keys.intersection(row):
            rows.append(row)

    if not rows:
        return None

    out_df = pd.DataFrame(rows)
    out_path = run_dir / "results" / "four_option_comparison_with_cot_ramp.csv"
    out_df.to_csv(out_path, index=False)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run automated data + one experiment pass.")
    parser.add_argument("--config", default="src/config/default.yaml")
    parser.add_argument("--data-dir", default="Data_auto")
    parser.add_argument("--seed-data-dir", default="Data")
    parser.add_argument(
        "--target-mode",
        choices=["price", "returns"],
        default="returns",
        help="Training target mode; returns is robust to long-run price-level drift.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional reproducibility seed override.")
    parser.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="Skip refresh and use existing automated data directory as-is.",
    )
    parser.add_argument(
        "--no-paper-snapshot",
        action="store_true",
        help="Disable run-level paper snapshot generation.",
    )
    parser.add_argument(
        "--allow-shared-paper-write",
        action="store_true",
        help="Allow writing to shared project paper directory.",
    )
    parser.add_argument(
        "--enable-llm",
        action="store_true",
        help="Enable LLM refinement methods from config.",
    )
    parser.add_argument(
        "--cot-end-to-end",
        action="store_true",
        help="Run end-to-end CoT flow (TSM+LLM-COT-RF and Option-4 ramp blend artifacts).",
    )
    parser.add_argument(
        "--cot-method",
        choices=["cot_rf", "cot_sent_rf", "both"],
        default="cot_rf",
        help="CoT method profile when --cot-end-to-end is enabled.",
    )
    parser.add_argument(
        "--llm-max-samples",
        type=int,
        default=12,
        help="Max samples for LLM refinement subset (cost/runtime control).",
    )
    parser.add_argument(
        "--llm-model",
        default="qwen3-vl-4b-gpu",
        help="LLM model identifier for OpenAI-compatible endpoint in CoT mode.",
    )
    parser.add_argument(
        "--llm-api-key",
        default=None,
        help="API key for LLM endpoint. Fallback order: --llm-api-key, OPENAI_API_KEY, local default.",
    )
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help="Optional OpenAI-compatible base URL for local LLM endpoint.",
    )
    parser.add_argument(
        "--sentiment-path",
        default="data/news/daily_sentiment.csv",
        help="Daily sentiment CSV path (required for cot_sent_rf / both).",
    )
    parser.add_argument("--include-news", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    seed_data_dir = Path(args.seed_data_dir).resolve()

    if not args.skip_bootstrap:
        seed_data_dir = _resolve_seed_data_dir(seed_data_dir=seed_data_dir, data_dir=data_dir)
        _run_bootstrap(
            output_dir=data_dir,
            seed_data_dir=seed_data_dir,
            include_news=args.include_news,
        )

    overrides = {
        "target": {
            "mode": args.target_mode,
        },
        "output": {
            "generate_paper": not args.no_paper_snapshot,
            "write_project_paper": bool(args.allow_shared_paper_write),
        },
    }

    if args.cot_end_to_end:
        resolved_api_key = _resolve_llm_api_key(args.llm_api_key)
        if args.cot_method == "cot_rf":
            methods = ["TSM+LLM-COT-RF"]
        elif args.cot_method == "cot_sent_rf":
            methods = ["TSM+LLM-COT-SENT-RF"]
        else:
            methods = ["TSM+LLM-COT-RF", "TSM+LLM-COT-SENT-RF"]

        sentiment_enabled = any(m.endswith("COT-SENT-RF") for m in methods)
        if sentiment_enabled and not Path(args.sentiment_path).exists():
            raise FileNotFoundError(
                f"Sentiment file required for {args.cot_method}: {args.sentiment_path}"
            )

        resolved_base_url = (
            args.llm_base_url
            or os.environ.get("OPENAI_BASE_URL")
            or DEFAULT_LOCAL_LLM_BASE_URL
        )

        llm_overrides = {
            "provider": "openai",
            "model": args.llm_model,
            "temperature": 0.0,
            "max_tokens": 512,
            "timeout_seconds": 300,
            "methods": methods,
            "history_points": 18,
            "include_summary_stats": True,
            "max_retries": 3,
            "cache_enabled": True,
            "max_samples": int(args.llm_max_samples),
            "subset": {"strategy": "random", "seed": 42},
            "max_exogenous_features": 6,
            "cot_rf": {
                "k_examples": 5,
                "example_selection": "similarity",
                "feature_window": 18,
                "lookback_days": 365,
                "retain_context": True,
            },
            "prompt_history_points": 18,
            "prompt_sentiment_points": 18,
            "api_key": resolved_api_key,
            "base_url": resolved_base_url,
            "sentiment": {
                "enabled": sentiment_enabled,
                "path": args.sentiment_path,
                "date_col": "seendate",
                "score_col": "sent_score",
                "history_points": 18,
            },
            "blend_grid": {
                "enabled": True,
                "schedule": "ramp",
                "weights": [0.5],
                "key_horizons": [1, 5, 10, 20, 30],
                "tune_split": "val",
                "metric": "mse_path",
                "min_weight": 0.15,
                "power": 1.0,
                "max_samples": int(args.llm_max_samples),
                "methods": methods,
            },
        }
        overrides["llm"] = llm_overrides
    elif args.enable_llm:
        resolved_api_key = _resolve_llm_api_key(args.llm_api_key)
        llm_overrides = {
            "api_key": resolved_api_key,
            "max_samples": int(args.llm_max_samples),
        }
        if args.llm_base_url:
            llm_overrides["base_url"] = args.llm_base_url
        if args.llm_model:
            llm_overrides["model"] = args.llm_model
        overrides["llm"] = llm_overrides
    else:
        overrides["llm"] = {"methods": [], "max_samples": 0}

    if args.seed is not None:
        overrides["reproducibility"] = {"seed": int(args.seed)}

    run_dir = run_experiment(
        config_path=args.config,
        overrides=overrides,
        data_dir=str(data_dir),
    )
    run_dir = Path(run_dir)

    metrics, path_metrics = _read_performance_summary(run_dir)
    option_table_path = _build_option_comparison(run_dir, metrics, path_metrics)
    _print_summary(run_dir, metrics, path_metrics)
    if option_table_path:
        print(f"\nOption comparison table: {option_table_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
