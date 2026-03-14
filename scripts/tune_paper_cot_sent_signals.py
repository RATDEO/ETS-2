#!/usr/bin/env python3
"""
Leakage-free search over daily sentiment signal variants for CoT-SENT.

Workflow:
1. Load a saved clean paper-style run and its TSM checkpoint.
2. Load a chosen CoT-SENT hyperparameter candidate.
3. Search sentiment series on a validation subset.
4. Re-evaluate top signals on the full validation split.
5. Run the selected best signal once on the held-out test split.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm.refine import LLMRefiner
from run_experiment import build_history_dates, build_sentiment_histories, load_daily_sentiment
from tune_paper_cot_sent_hparams import (  # type: ignore
    CoTSentCandidate,
    _evaluate_candidate,
    _prepare_bundle,
)


@dataclass(frozen=True)
class SignalCandidate:
    name: str
    path: str | None = None
    score_col: str = "sent_score"
    date_col: str = "seendate"
    source_path: str | None = None
    transform: str | None = None


SIGNAL_CANDIDATES: list[SignalCandidate] = [
    SignalCandidate("baseline_daily_sentiment", "data/news/daily_sentiment.csv"),
    SignalCandidate("qwen_votes3_daily1", "data/news/daily_sentiment_qwen_votes3_daily1.csv"),
    SignalCandidate("qwen_votes3_daily3", "data/news/daily_sentiment_qwen_votes3_daily3.csv"),
    SignalCandidate(
        "qwen_votes3_daily3_gate_policy_shipping",
        "data/news/daily_sentiment_qwen_votes3_daily3_gate_policy_shipping.csv",
    ),
    SignalCandidate(
        "qwen_votes3_daily3_event90_proxy",
        "data/news/daily_sentiment_qwen_votes3_daily3_gate_policy_shipping_event90_proxy.csv",
    ),
    SignalCandidate(
        "qwen_votes3_daily3_calib_regime_trainret",
        "data/news/daily_sentiment_qwen_votes3_daily3_calib_regime_trainret.csv",
    ),
    SignalCandidate(
        "qwen_votes3_daily3_dawidskene",
        "data/news/daily_sentiment_qwen_votes3_daily3_dawidskene_posteriors.csv",
        score_col="sent_score_ds",
    ),
    SignalCandidate(
        "feat4_score",
        source_path="data/news/daily_sentiment_qwen_votes3_daily3_gate_policy_shipping_event90_proxy_feat4.csv",
        transform="identity",
    ),
    SignalCandidate(
        "feat4_score_plus_change",
        source_path="data/news/daily_sentiment_qwen_votes3_daily3_gate_policy_shipping_event90_proxy_feat4.csv",
        transform="score_plus_change",
    ),
    SignalCandidate(
        "feat4_volume_weighted_score",
        source_path="data/news/daily_sentiment_qwen_votes3_daily3_gate_policy_shipping_event90_proxy_feat4.csv",
        transform="volume_weighted_score",
    ),
    SignalCandidate(
        "feat4_combo",
        source_path="data/news/daily_sentiment_qwen_votes3_daily3_gate_policy_shipping_event90_proxy_feat4.csv",
        transform="combo",
    ),
    SignalCandidate(
        "feat4_train_linear_proxy",
        source_path="data/news/daily_sentiment_qwen_votes3_daily3_gate_policy_shipping_event90_proxy_feat4.csv",
        transform="train_linear_proxy",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune CoT-SENT daily signal variants on a clean saved run.")
    parser.add_argument(
        "--base-run",
        type=Path,
        default=Path("runs/20260228_145058_40d91f"),
        help="Existing clean paper-style CoT-SENT run directory.",
    )
    parser.add_argument(
        "--candidate-summary",
        type=Path,
        default=None,
        help="selection_summary.json from tune_paper_cot_sent_hparams.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default under reports/cot_sent_signal_tune).",
    )
    parser.add_argument(
        "--llm-api-key",
        type=str,
        default=None,
        help="LLM API key. Falls back to OPENAI_API_KEY then config value then deo.",
    )
    parser.add_argument(
        "--llm-base-url",
        type=str,
        default=None,
        help="LLM base URL. Falls back to OPENAI_BASE_URL then config value.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Override LLM model id for this run.",
    )
    parser.add_argument(
        "--val-subset-samples",
        type=int,
        default=60,
        help="Validation subset size for stage-1 signal search.",
    )
    parser.add_argument(
        "--top-k-full-val",
        type=int,
        default=2,
        help="How many stage-1 winners to rerun on the full validation split.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="Override LLM request timeout for this run.",
    )
    parser.add_argument(
        "--signal-names",
        type=str,
        default=None,
        help="Comma-separated subset of signal candidate names to evaluate.",
    )
    parser.add_argument(
        "--stop-after-stage1",
        action="store_true",
        help="Stop after the stage-1 subset screen.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="TSM+LLM-COT-SENT-RF",
        choices=["TSM+LLM-COT-SENT-RF", "TSM+LLM-COT-SENT-RF-HDELTA"],
        help="LLM refinement method to evaluate.",
    )
    parser.add_argument(
        "--hdelta-max-adjustment-pct",
        type=float,
        default=3.0,
        help="Max absolute percentage adjustment per key horizon for HDELTA.",
    )
    parser.add_argument(
        "--cot-strict-json-prompt",
        action="store_true",
        help="Use strict JSON reflection prompts for CoT/HDELTA methods.",
    )
    parser.add_argument(
        "--retain-context-override",
        choices=["default", "true", "false"],
        default="default",
        help="Override retain_context for the selected candidate.",
    )
    parser.add_argument(
        "--hdelta-freeze-horizons",
        type=str,
        default=None,
        help="Comma-separated horizons to force to 0.0 adjustment, e.g. 1 or 1,5.",
    )
    parser.add_argument(
        "--hdelta-sentiment-secondary",
        action="store_true",
        help="Treat sentiment as secondary evidence only in the HDELTA apply prompt.",
    )
    parser.add_argument(
        "--example-selection-override",
        type=str,
        default=None,
        choices=["similarity", "recent_high_error", "error_stratified"],
        help="Override teaching-example selection mode for the chosen candidate.",
    )
    return parser.parse_args()


def _load_best_candidate(path: Path | None) -> CoTSentCandidate:
    if path is None:
        return CoTSentCandidate(
            name="paper_sim_k5_h18_lb365_ctx",
            history_points=18,
            prompt_history_points=18,
            prompt_sentiment_points=18,
            sentiment_points=18,
            k_examples=5,
            example_selection="similarity",
            feature_window=18,
            lookback_days=365,
            retain_context=True,
        )
    candidate_summary = path if path.is_absolute() else (ROOT / path)
    payload = json.loads(candidate_summary.read_text(encoding="utf-8"))
    return CoTSentCandidate(**payload["best_candidate"])


def _select_signals(signal_names: str | None) -> list[SignalCandidate]:
    if not signal_names:
        return SIGNAL_CANDIDATES
    requested = [name.strip() for name in signal_names.split(",") if name.strip()]
    selected = [signal for signal in SIGNAL_CANDIDATES if signal.name in requested]
    missing = sorted(set(requested) - {signal.name for signal in selected})
    if missing:
        raise ValueError(f"Unknown signal candidate names: {missing}")
    return selected


def _parse_freeze_horizons(text: str | None) -> list[int]:
    if not text:
        return []
    vals = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        vals.append(int(item))
    return vals


def _fit_ridge_projection(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    eye = np.eye(X.shape[1], dtype=float)
    return np.linalg.solve(X.T @ X + alpha * eye, X.T @ y)


def _build_engineered_signal(
    signal: SignalCandidate,
    prepared_dir: Path,
    bundle: dict[str, Any],
) -> tuple[str, str]:
    if signal.path is not None:
        resolved = ROOT / signal.path if not Path(signal.path).is_absolute() else Path(signal.path)
        return str(resolved), signal.score_col

    if signal.source_path is None or signal.transform is None:
        raise ValueError(f"Signal candidate {signal.name} is missing path/source configuration.")

    source_path = ROOT / signal.source_path if not Path(signal.source_path).is_absolute() else Path(signal.source_path)
    df = pd.read_csv(source_path)
    if signal.date_col not in df.columns:
        raise ValueError(f"Date column {signal.date_col} not found in {source_path}")
    required = {"sent_score", "news_volume", "sent_change"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing engineered-signal columns in {source_path}: {sorted(missing)}")

    sent = pd.to_numeric(df["sent_score"], errors="coerce").fillna(0.0)
    volume = pd.to_numeric(df["news_volume"], errors="coerce").fillna(0.0)
    change = pd.to_numeric(df["sent_change"], errors="coerce").fillna(0.0)

    if signal.transform == "identity":
        derived = sent
    elif signal.transform == "score_plus_change":
        derived = sent + 0.20 * change
    elif signal.transform == "volume_weighted_score":
        derived = sent * (1.0 + 0.15 * (volume - 2.0))
    elif signal.transform == "combo":
        derived = sent * (1.0 + 0.15 * (volume - 2.0)) + 0.20 * change
    elif signal.transform == "train_linear_proxy":
        feat_dates = pd.to_datetime(df[signal.date_col]).dt.normalize()
        train_dates = pd.to_datetime(bundle["splits"]["train"]["dates"]).normalize()
        train_last = np.asarray(bundle["y_train_hist"][:, -1], dtype=float)
        train_day1 = np.asarray(bundle["y_train_true"][:, 0], dtype=float)
        train_target = np.log(np.clip(train_day1, 1e-6, None) / np.clip(train_last, 1e-6, None))
        train_df = pd.DataFrame(
            {
                signal.date_col: train_dates,
                "next_ret": train_target,
            }
        )
        merged = pd.DataFrame(
            {
                signal.date_col: feat_dates,
                "sent_score": sent,
                "news_volume": volume,
                "sent_change": change,
            }
        ).merge(train_df, on=signal.date_col, how="inner")
        X_train = merged[["sent_score", "news_volume", "sent_change"]].to_numpy(dtype=float)
        y_train = merged["next_ret"].to_numpy(dtype=float)
        if len(merged) < 20:
            raise ValueError(f"Insufficient rows to fit train_linear_proxy for {signal.name}: {len(merged)}")
        beta = _fit_ridge_projection(X_train, y_train, alpha=1.0)
        proj_train = X_train @ beta
        scale = float(np.std(proj_train)) if len(proj_train) else 1.0
        scale = max(scale, 1e-6)
        proj_all = np.column_stack([sent, volume, change]).astype(float) @ beta
        derived = np.tanh(proj_all / (2.0 * scale))
    else:
        raise ValueError(f"Unknown engineered signal transform: {signal.transform}")

    prepared_dir.mkdir(parents=True, exist_ok=True)
    output_path = prepared_dir / f"{signal.name}.csv"
    prepared = pd.DataFrame(
        {
            signal.date_col: df[signal.date_col],
            "sent_score": np.clip(np.asarray(derived, dtype=float), -1.0, 1.0),
        }
    )
    prepared.to_csv(output_path, index=False)
    return str(output_path), "sent_score"


def _evaluate_signal(
    bundle: dict[str, Any],
    base_llm_cfg: dict[str, Any],
    cot_candidate: CoTSentCandidate,
    signal: SignalCandidate,
    split_name: str,
    eval_indices: np.ndarray,
    output_dir: Path,
    method: str,
) -> dict[str, Any]:
    llm_cfg = copy.deepcopy(base_llm_cfg)
    llm_cfg.setdefault("sentiment", {})
    prepared_path, score_col = _build_engineered_signal(
        signal,
        output_dir / "_prepared_signals",
        bundle=bundle,
    )
    llm_cfg["sentiment"]["path"] = prepared_path
    llm_cfg["sentiment"]["score_col"] = score_col
    llm_cfg["sentiment"]["date_col"] = signal.date_col
    result = _evaluate_candidate(
        bundle=bundle,
        base_llm_cfg=llm_cfg,
        candidate=CoTSentCandidate(
            name=signal.name,
            history_points=cot_candidate.history_points,
            prompt_history_points=cot_candidate.prompt_history_points,
            prompt_sentiment_points=cot_candidate.prompt_sentiment_points,
            sentiment_points=cot_candidate.sentiment_points,
            k_examples=cot_candidate.k_examples,
            example_selection=cot_candidate.example_selection,
            feature_window=cot_candidate.feature_window,
            lookback_days=cot_candidate.lookback_days,
            retain_context=cot_candidate.retain_context,
            strict_json_prompt=cot_candidate.strict_json_prompt,
        ),
        split_name=split_name,
        eval_indices=eval_indices,
        output_dir=output_dir,
        method=method,
    )
    result["signal_name"] = signal.name
    result["sentiment_path"] = prepared_path
    result["score_col"] = score_col
    result["signal_transform"] = signal.transform or "direct"
    return result


def main() -> int:
    args = parse_args()
    base_run = args.base_run if args.base_run.is_absolute() else (ROOT / args.base_run)
    if not base_run.exists():
        raise FileNotFoundError(f"Run not found: {base_run}")

    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = ROOT / "reports" / "cot_sent_signal_tune" / stamp
    else:
        output_dir = args.output_dir if args.output_dir.is_absolute() else (ROOT / args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = _prepare_bundle(base_run)
    base_llm_cfg = copy.deepcopy(bundle["cfg"].get("llm", {}) or {})
    base_llm_cfg["api_key"] = args.llm_api_key or os.environ.get("OPENAI_API_KEY") or base_llm_cfg.get("api_key") or "deo"
    base_llm_cfg["base_url"] = args.llm_base_url or os.environ.get("OPENAI_BASE_URL") or base_llm_cfg.get("base_url")
    if args.llm_model:
        base_llm_cfg["model"] = args.llm_model
    base_llm_cfg["methods"] = [args.method]
    base_llm_cfg["max_samples"] = 100000
    if args.timeout_seconds is not None:
        base_llm_cfg["timeout_seconds"] = float(args.timeout_seconds)
    if args.method == "TSM+LLM-COT-SENT-RF-HDELTA":
        base_llm_cfg.setdefault("hdelta", {})
        base_llm_cfg["hdelta"]["key_horizons"] = [1, 5, 20, 30]
        base_llm_cfg["hdelta"]["max_adjustment_pct"] = float(args.hdelta_max_adjustment_pct)
        base_llm_cfg["hdelta"]["freeze_horizons"] = _parse_freeze_horizons(args.hdelta_freeze_horizons)
        base_llm_cfg["hdelta"]["sentiment_secondary"] = bool(args.hdelta_sentiment_secondary)
    if args.cot_strict_json_prompt:
        base_llm_cfg.setdefault("cot_rf", {})
        base_llm_cfg["cot_rf"]["strict_json_prompt"] = True
        base_llm_cfg["cot_rf"]["strict_json_response_format"] = True

    best_candidate = _load_best_candidate(args.candidate_summary)
    if args.retain_context_override != "default":
        best_candidate = CoTSentCandidate(
            name=best_candidate.name,
            history_points=best_candidate.history_points,
            prompt_history_points=best_candidate.prompt_history_points,
            prompt_sentiment_points=best_candidate.prompt_sentiment_points,
            sentiment_points=best_candidate.sentiment_points,
            k_examples=best_candidate.k_examples,
            example_selection=best_candidate.example_selection,
            feature_window=best_candidate.feature_window,
            lookback_days=best_candidate.lookback_days,
            retain_context=(args.retain_context_override == "true"),
            strict_json_prompt=best_candidate.strict_json_prompt,
        )
    if args.cot_strict_json_prompt:
        best_candidate = CoTSentCandidate(
            name=best_candidate.name,
            history_points=best_candidate.history_points,
            prompt_history_points=best_candidate.prompt_history_points,
            prompt_sentiment_points=best_candidate.prompt_sentiment_points,
            sentiment_points=best_candidate.sentiment_points,
            k_examples=best_candidate.k_examples,
            example_selection=best_candidate.example_selection,
            feature_window=best_candidate.feature_window,
            lookback_days=best_candidate.lookback_days,
            retain_context=best_candidate.retain_context,
            strict_json_prompt=True,
        )
    if args.example_selection_override:
        best_candidate = CoTSentCandidate(
            name=best_candidate.name,
            history_points=best_candidate.history_points,
            prompt_history_points=best_candidate.prompt_history_points,
            prompt_sentiment_points=best_candidate.prompt_sentiment_points,
            sentiment_points=best_candidate.sentiment_points,
            k_examples=best_candidate.k_examples,
            example_selection=str(args.example_selection_override),
            feature_window=best_candidate.feature_window,
            lookback_days=best_candidate.lookback_days,
            retain_context=best_candidate.retain_context,
            strict_json_prompt=best_candidate.strict_json_prompt,
        )
    selected_signals = _select_signals(args.signal_names)

    val_n = len(bundle["splits"]["val"]["dates"])
    subset_eval_indices = np.asarray(
        np.linspace(0, val_n - 1, min(max(args.val_subset_samples, 1), val_n), dtype=int)
    )
    full_val_indices = np.arange(val_n, dtype=int)
    full_test_indices = np.arange(len(bundle["splits"]["test"]["dates"]), dtype=int)

    manifest = {
        "base_run": str(base_run),
        "method": args.method,
        "cot_strict_json_prompt": bool(args.cot_strict_json_prompt),
        "retain_context_override": args.retain_context_override,
        "hdelta_freeze_horizons": _parse_freeze_horizons(args.hdelta_freeze_horizons),
        "hdelta_sentiment_secondary": bool(args.hdelta_sentiment_secondary),
        "hdelta_max_adjustment_pct": float(args.hdelta_max_adjustment_pct),
        "example_selection_override": args.example_selection_override,
        "best_candidate": best_candidate.__dict__,
        "val_subset_samples": int(len(subset_eval_indices)),
        "top_k_full_val": int(args.top_k_full_val),
        "signals": [signal.__dict__ for signal in selected_signals],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[SIGNAL] stage1 subset size={len(subset_eval_indices)} signals={len(selected_signals)}", flush=True)
    stage1_results = []
    for signal in selected_signals:
        print(f"[SIGNAL] subset {signal.name}", flush=True)
        result = _evaluate_signal(
            bundle=bundle,
            base_llm_cfg=base_llm_cfg,
            cot_candidate=best_candidate,
            signal=signal,
            split_name="val",
            eval_indices=subset_eval_indices,
            output_dir=output_dir / "stage1_subset",
            method=args.method,
        )
        stage1_results.append(result)
        pd.DataFrame(stage1_results).sort_values("path_mse").to_csv(
            output_dir / "stage1_subset_results.csv",
            index=False,
        )
        print(f"[SIGNAL] subset done {signal.name} path_mse={result['path_mse']:.6f}", flush=True)

    stage1_df = pd.DataFrame(stage1_results).sort_values("path_mse")
    if args.stop_after_stage1:
        summary = {
            "base_run": str(base_run),
            "best_candidate": best_candidate.__dict__,
            "stage1_best": stage1_df.to_dict(orient="records"),
        }
        (output_dir / "selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
        print(f"Wrote signal tuning outputs to {output_dir}", flush=True)
        return 0

    top_names = stage1_df["signal_name"].head(int(min(args.top_k_full_val, len(stage1_df)))).tolist()
    top_signals = [signal for signal in selected_signals if signal.name in top_names]

    print(f"[SIGNAL] stage2 full validation top={top_names}", flush=True)
    stage2_results = []
    for signal in top_signals:
        print(f"[SIGNAL] full val {signal.name}", flush=True)
        result = _evaluate_signal(
            bundle=bundle,
            base_llm_cfg=base_llm_cfg,
            cot_candidate=best_candidate,
            signal=signal,
            split_name="val",
            eval_indices=full_val_indices,
            output_dir=output_dir / "stage2_full_val",
            method=args.method,
        )
        stage2_results.append(result)
        pd.DataFrame(stage2_results).sort_values("path_mse").to_csv(
            output_dir / "stage2_full_val_results.csv",
            index=False,
        )
        print(f"[SIGNAL] full val done {signal.name} path_mse={result['path_mse']:.6f}", flush=True)

    stage2_df = pd.DataFrame(stage2_results).sort_values("path_mse")
    best_signal_name = str(stage2_df.iloc[0]["signal_name"])
    best_signal = next(signal for signal in selected_signals if signal.name == best_signal_name)

    print(f"[SIGNAL] stage3 test best={best_signal_name}", flush=True)
    test_result = _evaluate_signal(
        bundle=bundle,
        base_llm_cfg=base_llm_cfg,
        cot_candidate=best_candidate,
        signal=best_signal,
        split_name="test",
        eval_indices=full_test_indices,
        output_dir=output_dir / "stage3_test",
        method=args.method,
    )
    pd.DataFrame([test_result]).to_csv(output_dir / "stage3_test_result.csv", index=False)

    baseline_test_npz = np.load(
        base_run / "predictions" / "TSM+LLM-COT-SENT-RF_pred_test_subset.npz",
        allow_pickle=True,
    )
    baseline_test = {
        "signal_name": "baseline_paper_signal",
        "sentiment_path": str((bundle["cfg"].get("llm", {}) or {}).get("sentiment", {}).get("path", "")),
        "score_col": str((bundle["cfg"].get("llm", {}) or {}).get("sentiment", {}).get("score_col", "sent_score")),
        "path_mse": float(np.mean((baseline_test_npz["y_true"] - baseline_test_npz["yhat"]) ** 2)),
    }
    pd.DataFrame([baseline_test, test_result]).to_csv(output_dir / "final_test_comparison.csv", index=False)

    summary = {
        "base_run": str(base_run),
        "method": args.method,
        "best_candidate": best_candidate.__dict__,
        "best_signal": best_signal.__dict__,
        "stage1_best": stage1_df.head(int(min(args.top_k_full_val, len(stage1_df)))).to_dict(orient="records"),
        "stage2_best": stage2_df.iloc[0].to_dict(),
        "baseline_test": baseline_test,
        "best_signal_test": test_result,
    }
    (output_dir / "selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote signal tuning outputs to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
