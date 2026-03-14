#!/usr/bin/env python3
"""
Strategy 3: inject sentiment features into the TSM input panel, refit the clean TSM,
then compare the four tracked model options on the same sentiment-aware base.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for p in (ROOT, SRC, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data import build_panel
from eval.metrics import compute_metrics_by_horizon, compute_path_metrics
from tune_clean_tsm_cot_rf import (  # type: ignore
    TSM_CANDIDATES,
    deep_update,
    refit_best_tsm,
    train_tsm_candidate,
)
from tune_paper_cot_sent_hparams import (  # type: ignore
    CoTSentCandidate,
    _evaluate_candidate,
    _metric_row,
    _prepare_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Strategy 3 sentiment-aware TSM comparison.")
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("src/config/paper_llm_cot_sent_rf_qwen_k5_h18_similarity_ctx_full.yaml"),
        help="Base config template.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("Data_auto"),
        help="Data directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory under reports/strategy3_sentiment_tsm/.",
    )
    parser.add_argument(
        "--sentiment-path",
        type=str,
        default="data/news/daily_sentiment.csv",
        help="Daily sentiment CSV to inject into the TSM panel.",
    )
    parser.add_argument(
        "--sentiment-date-col",
        type=str,
        default="seendate",
        help="Date column in the sentiment CSV.",
    )
    parser.add_argument(
        "--sentiment-score-col",
        type=str,
        default="sent_score",
        help="Score column in the sentiment CSV.",
    )
    parser.add_argument(
        "--structured-news",
        action="store_true",
        help="Enable richer structured-news features when available (e.g. news_volume, sent_change).",
    )
    parser.add_argument(
        "--max-exogenous-features",
        type=int,
        default=10,
        help="Number of non-target model input features to keep, including price y.",
    )
    parser.add_argument(
        "--llm-api-key",
        type=str,
        default=None,
        help="LLM API key. Falls back to OPENAI_API_KEY then deo.",
    )
    parser.add_argument(
        "--llm4b-base-url",
        type=str,
        default="http://192.168.1.140:9877/v1",
        help="4B endpoint.",
    )
    parser.add_argument(
        "--llm4b-model",
        type=str,
        default="qwen3-vl-4b-gpu",
        help="4B model id.",
    )
    parser.add_argument(
        "--llm35b-base-url",
        type=str,
        default="http://192.168.1.140:9881/v1",
        help="35B endpoint.",
    )
    parser.add_argument(
        "--llm35b-model",
        type=str,
        default="qwen3.5-35b-a3b-ud-q4-k-xl",
        help="35B model id.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=60.0,
        help="LLM request timeout.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as handle:
        return yaml.safe_load(handle)


def with_strategy3_feature_overrides(raw: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = copy.deepcopy(raw)
    out.setdefault("features", {})
    out["features"].update(
        {
            "include_sentiment_features": True,
            "sentiment_path": args.sentiment_path,
            "sentiment_date_col": args.sentiment_date_col,
            "sentiment_score_col": args.sentiment_score_col,
            "sentiment_windows": [3, 7],
            "include_sentiment_change": True,
            "max_exogenous_features_model": int(args.max_exogenous_features),
        }
    )
    if args.structured_news:
        out["features"].update(
            {
                "include_sentiment_volume": True,
                "include_source_sent_change": True,
                "include_sentiment_interaction": True,
                "sentiment_volume_windows": [3],
                "preferred_feature_order": [
                    "sent_score",
                    "sent_score_3d_ma",
                    "sent_score_7d_ma",
                    "sent_change",
                    "sent_news_volume",
                    "sent_news_volume_3d_ma",
                    "sent_score_x_volume",
                ],
            }
        )
    else:
        out["features"]["preferred_feature_order"] = [
            "sent_score",
            "sent_score_3d_ma",
            "sent_score_7d_ma",
            "sent_score_change_1d",
        ]
    out.setdefault("llm", {})
    out["llm"].setdefault("sentiment", {})
    out["llm"]["sentiment"].update(
        {
            "enabled": True,
            "path": args.sentiment_path,
            "date_col": args.sentiment_date_col,
            "score_col": args.sentiment_score_col,
            "history_points": 18,
        }
    )
    out["target"]["mode"] = "returns"
    out["output"]["generate_paper"] = False
    out["robustness"]["noise_levels"] = []
    return out


def save_base_run(base_run_dir: Path, panel: pd.DataFrame, schema: dict[str, Any], raw_config: dict[str, Any], tsm_model: Any) -> None:
    (base_run_dir / "data").mkdir(parents=True, exist_ok=True)
    (base_run_dir / "models").mkdir(parents=True, exist_ok=True)
    panel.to_parquet(base_run_dir / "data" / "panel.parquet", index=False)
    with (base_run_dir / "data" / "panel_schema.json").open("w") as handle:
        json.dump(schema, handle, indent=2)
    with (base_run_dir / "config_resolved.yaml").open("w") as handle:
        yaml.safe_dump(raw_config, handle, sort_keys=False)
    tsm_model.save(base_run_dir / "models" / "tsm_checkpoint.pt")


def make_llm_cfg(base_cfg: dict[str, Any], api_key: str, base_url: str, model: str, method: str, timeout_seconds: float) -> dict[str, Any]:
    llm_cfg = copy.deepcopy(base_cfg)
    llm_cfg["api_key"] = api_key
    llm_cfg["base_url"] = base_url
    llm_cfg["model"] = model
    llm_cfg["methods"] = [method]
    llm_cfg["max_samples"] = 100000
    llm_cfg["timeout_seconds"] = float(timeout_seconds)
    return llm_cfg


def main() -> int:
    args = parse_args()
    api_key = args.llm_api_key or os.environ.get("OPENAI_API_KEY") or "deo"
    base_config_path = args.base_config if args.base_config.is_absolute() else (ROOT / args.base_config)
    data_dir = args.data_dir if args.data_dir.is_absolute() else (ROOT / args.data_dir)

    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = ROOT / "reports" / "strategy3_sentiment_tsm" / stamp
    else:
        output_dir = args.output_dir if args.output_dir.is_absolute() else (ROOT / args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_raw = with_strategy3_feature_overrides(load_yaml(base_config_path), args)
    panel, schema = build_panel(data_dir=data_dir, config=base_raw)

    print(f"[STRAT3] panel rows={len(panel)} cols={len(panel.columns)}", flush=True)
    tsm_results = []
    splits_cache: dict[tuple[int, int], dict[str, Any]] = {}
    seed = int((base_raw.get("reproducibility", {}) or {}).get("seed", 42))

    for candidate in TSM_CANDIDATES:
        print(f"[STRAT3] tsm candidate {candidate.name}", flush=True)
        result = train_tsm_candidate(
            panel=panel,
            base_raw=base_raw,
            candidate=candidate,
            splits_cache=splits_cache,
            seed=seed,
        )
        tsm_results.append(
            {
                "name": result["name"],
                "val_path_mse": result["val_path_mse"],
                "test_path_mse": result["test_path_mse"],
                "best_val_loss": result["best_val_loss"],
                "epochs_ran": result["epochs_ran"],
            }
        )
        pd.DataFrame(tsm_results).sort_values("val_path_mse").to_csv(output_dir / "tsm_search_results.csv", index=False)
        print(f"[STRAT3] tsm done {candidate.name} val_path_mse={result['val_path_mse']:.6f}", flush=True)

    best_tsm_name = pd.DataFrame(tsm_results).sort_values("val_path_mse").iloc[0]["name"]
    best_candidate = next(candidate for candidate in TSM_CANDIDATES if candidate.name == best_tsm_name)
    best_raw = with_strategy3_feature_overrides(load_yaml(base_config_path), args)
    best_raw = copy.deepcopy(best_raw)
    best_raw["time_series"]["seq_len"] = best_candidate.seq_len
    best_raw["time_series"]["label_len"] = best_candidate.label_len
    best_raw["time_series"]["pred_len"] = 30
    best_raw["model"]["tsm_type"] = "autoformer"
    best_raw["model"]["batch_size"] = best_candidate.batch_size
    best_raw["model"]["learning_rate"] = best_candidate.learning_rate
    best_raw["model"]["d_model"] = best_candidate.d_model
    best_raw["model"]["n_heads"] = best_candidate.n_heads
    best_raw["model"]["e_layers"] = best_candidate.e_layers
    best_raw["model"]["d_layers"] = 1
    best_raw["model"]["d_ff"] = best_candidate.d_ff
    best_raw["model"]["dropout"] = best_candidate.dropout
    best_raw["model"]["weight_decay"] = best_candidate.weight_decay
    best_raw["model"]["grad_clip"] = best_candidate.grad_clip
    best_raw["model"]["max_epochs"] = best_candidate.max_epochs
    best_raw["model"]["early_stopping_patience"] = best_candidate.patience
    best_raw["model"]["use_residual_wrapper"] = best_candidate.use_residual_wrapper
    best_raw["model"]["residual_alpha"] = best_candidate.residual_alpha

    print(f"[STRAT3] refit best tsm {best_candidate.name}", flush=True)
    tsm_artifacts = refit_best_tsm(panel=panel, raw=best_raw, seed=seed)
    base_run_dir = output_dir / "base_run"
    save_base_run(
        base_run_dir=base_run_dir,
        panel=panel,
        schema=schema,
        raw_config=tsm_artifacts["raw_config"],
        tsm_model=tsm_artifacts["model"],
    )

    bundle = _prepare_bundle(base_run_dir)
    full_test_indices = np.arange(len(bundle["splits"]["test"]["dates"]), dtype=int)
    base_llm_cfg = copy.deepcopy(bundle["cfg"].get("llm", {}) or {})

    results = []
    base_tsm_result = _metric_row("base_tsm_sent", bundle["y_test_true"], bundle["tsm_pred_test"])
    base_tsm_result["split"] = "test"
    base_tsm_result["n_samples"] = int(len(full_test_indices))
    results.append(base_tsm_result)

    paper_candidate = CoTSentCandidate(
        name="paper_cot_sent_4b_on_tsm_sent",
        history_points=18,
        prompt_history_points=18,
        prompt_sentiment_points=18,
        sentiment_points=18,
        k_examples=5,
        example_selection="similarity",
        feature_window=18,
        lookback_days=365,
        retain_context=True,
        strict_json_prompt=False,
    )
    hdelta_4b_candidate = CoTSentCandidate(
        name="hdelta_4b_on_tsm_sent",
        history_points=18,
        prompt_history_points=18,
        prompt_sentiment_points=18,
        sentiment_points=18,
        k_examples=5,
        example_selection="similarity",
        feature_window=18,
        lookback_days=365,
        retain_context=True,
        strict_json_prompt=False,
    )
    hdelta_35b_candidate = CoTSentCandidate(
        name="hdelta_35b_guarded_on_tsm_sent",
        history_points=18,
        prompt_history_points=18,
        prompt_sentiment_points=18,
        sentiment_points=18,
        k_examples=5,
        example_selection="similarity",
        feature_window=18,
        lookback_days=365,
        retain_context=False,
        strict_json_prompt=True,
    )

    print("[STRAT3] evaluate paper 4B CoT-SENT", flush=True)
    paper_cfg = make_llm_cfg(
        base_cfg=base_llm_cfg,
        api_key=api_key,
        base_url=args.llm4b_base_url,
        model=args.llm4b_model,
        method="TSM+LLM-COT-SENT-RF",
        timeout_seconds=args.timeout_seconds,
    )
    paper_result = _evaluate_candidate(
        bundle=bundle,
        base_llm_cfg=paper_cfg,
        candidate=paper_candidate,
        split_name="test",
        eval_indices=full_test_indices,
        output_dir=output_dir / "paper_cot_sent_4b",
        method="TSM+LLM-COT-SENT-RF",
    )
    results.append(paper_result)

    print("[STRAT3] evaluate 4B HDELTA", flush=True)
    hdelta_4b_cfg = make_llm_cfg(
        base_cfg=base_llm_cfg,
        api_key=api_key,
        base_url=args.llm4b_base_url,
        model=args.llm4b_model,
        method="TSM+LLM-COT-SENT-RF-HDELTA",
        timeout_seconds=args.timeout_seconds,
    )
    hdelta_4b_cfg.setdefault("hdelta", {})
    deep_update(
        hdelta_4b_cfg,
        {
            "hdelta": {
                "key_horizons": [1, 5, 20, 30],
                "max_adjustment_pct": 3.0,
                "freeze_horizons": [],
                "sentiment_secondary": False,
            }
        },
    )
    hdelta_4b_result = _evaluate_candidate(
        bundle=bundle,
        base_llm_cfg=hdelta_4b_cfg,
        candidate=hdelta_4b_candidate,
        split_name="test",
        eval_indices=full_test_indices,
        output_dir=output_dir / "hdelta_4b",
        method="TSM+LLM-COT-SENT-RF-HDELTA",
    )
    results.append(hdelta_4b_result)

    print("[STRAT3] evaluate 35B guarded HDELTA", flush=True)
    hdelta_35b_cfg = make_llm_cfg(
        base_cfg=base_llm_cfg,
        api_key=api_key,
        base_url=args.llm35b_base_url,
        model=args.llm35b_model,
        method="TSM+LLM-COT-SENT-RF-HDELTA",
        timeout_seconds=args.timeout_seconds,
    )
    deep_update(
        hdelta_35b_cfg,
        {
            "cot_rf": {
                "retain_context": False,
                "strict_json_prompt": True,
                "strict_json_response_format": True,
            },
            "hdelta": {
                "key_horizons": [1, 5, 20, 30],
                "max_adjustment_pct": 1.0,
                "freeze_horizons": [1],
                "sentiment_secondary": True,
            },
        },
    )
    hdelta_35b_result = _evaluate_candidate(
        bundle=bundle,
        base_llm_cfg=hdelta_35b_cfg,
        candidate=hdelta_35b_candidate,
        split_name="test",
        eval_indices=full_test_indices,
        output_dir=output_dir / "hdelta_35b_guarded",
        method="TSM+LLM-COT-SENT-RF-HDELTA",
    )
    results.append(hdelta_35b_result)

    result_df = pd.DataFrame(results).sort_values("path_mse").reset_index(drop=True)
    result_df.to_csv(output_dir / "four_option_comparison.csv", index=False)

    summary = {
        "strategy": "strategy3_sentiment_tsm",
        "base_run": str(base_run_dir),
        "sentiment_path": args.sentiment_path,
        "best_tsm_candidate": best_candidate.name,
        "best_tsm_val_path_mse": float(pd.DataFrame(tsm_results).sort_values("val_path_mse").iloc[0]["val_path_mse"]),
        "four_option_results": result_df.to_dict(orient="records"),
    }
    (output_dir / "selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
