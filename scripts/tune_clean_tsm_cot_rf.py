#!/usr/bin/env python3
"""
Validation-only retuning for the leakage-fixed TSM + CoT-RF pipeline.

Workflow:
1. Search a small TSM hyperparameter grid on the validation split only.
2. Refit the best TSM candidate and search CoT-RF prompt/example settings on validation.
3. Write the selected clean config for a final test-only run through the main pipeline.
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
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data import build_panel, select_feature_columns
from data.windows import StandardScaler, TimeSeriesDataset, WindowConfig, make_windows, split_windows
from eval.metrics import compute_metrics_by_horizon, compute_path_metrics
from llm.refine import LLMRefiner
from models.tsm import TSMForecaster
from run_experiment import (
    blend_forecasts,
    build_history_dates,
    evaluate_blend_grid,
    returns_to_prices,
    select_eval_indices,
)
from utils import set_seed

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required for tuning") from exc


@dataclass(frozen=True)
class TSMCandidate:
    name: str
    seq_len: int
    label_len: int
    batch_size: int
    learning_rate: float
    d_model: int
    n_heads: int
    e_layers: int
    d_ff: int
    dropout: float
    weight_decay: float
    grad_clip: float
    max_epochs: int
    patience: int
    use_residual_wrapper: bool = False
    residual_alpha: float = 0.1


@dataclass(frozen=True)
class CoTCandidate:
    name: str
    history_points: int
    prompt_history_points: int
    k_examples: int
    example_selection: str
    feature_window: int
    lookback_days: int | None
    retain_context: bool
    strict_json_prompt: bool = False
    strict_json_response_format: bool = True


TSM_CANDIDATES: list[TSMCandidate] = [
    TSMCandidate(
        name="af_s60_l15_m32_ff64_do030_lr1e3_wd1e2",
        seq_len=60,
        label_len=15,
        batch_size=16,
        learning_rate=1e-3,
        d_model=32,
        n_heads=2,
        e_layers=1,
        d_ff=64,
        dropout=0.30,
        weight_decay=1e-2,
        grad_clip=0.5,
        max_epochs=50,
        patience=12,
    ),
    TSMCandidate(
        name="af_s60_l15_m64_ff128_do030_lr1e3_wd1e2",
        seq_len=60,
        label_len=15,
        batch_size=16,
        learning_rate=1e-3,
        d_model=64,
        n_heads=2,
        e_layers=1,
        d_ff=128,
        dropout=0.30,
        weight_decay=1e-2,
        grad_clip=0.5,
        max_epochs=50,
        patience=15,
    ),
    TSMCandidate(
        name="af_s90_l30_m64_ff128_do030_lr5e4_wd1e2",
        seq_len=90,
        label_len=30,
        batch_size=16,
        learning_rate=5e-4,
        d_model=64,
        n_heads=2,
        e_layers=1,
        d_ff=128,
        dropout=0.30,
        weight_decay=1e-2,
        grad_clip=0.5,
        max_epochs=60,
        patience=15,
    ),
    TSMCandidate(
        name="af_s120_l30_m64_ff128_do020_lr3e4_wd1e3",
        seq_len=120,
        label_len=30,
        batch_size=32,
        learning_rate=3e-4,
        d_model=64,
        n_heads=2,
        e_layers=1,
        d_ff=128,
        dropout=0.20,
        weight_decay=1e-3,
        grad_clip=1.0,
        max_epochs=70,
        patience=15,
    ),
    TSMCandidate(
        name="af_s60_l15_m128_ff256_do020_lr5e4_wd1e3",
        seq_len=60,
        label_len=15,
        batch_size=16,
        learning_rate=5e-4,
        d_model=128,
        n_heads=4,
        e_layers=1,
        d_ff=256,
        dropout=0.20,
        weight_decay=1e-3,
        grad_clip=1.0,
        max_epochs=60,
        patience=15,
    ),
]


CoT_CANDIDATES: list[CoTCandidate] = [
    CoTCandidate(
        name="sim_k5_h18_ctx",
        history_points=18,
        prompt_history_points=18,
        k_examples=5,
        example_selection="similarity",
        feature_window=18,
        lookback_days=365,
        retain_context=True,
    ),
    CoTCandidate(
        name="sim_k3_h18_ctx",
        history_points=18,
        prompt_history_points=18,
        k_examples=3,
        example_selection="similarity",
        feature_window=18,
        lookback_days=365,
        retain_context=True,
    ),
    CoTCandidate(
        name="sim_k10_h18_ctx",
        history_points=18,
        prompt_history_points=18,
        k_examples=10,
        example_selection="similarity",
        feature_window=18,
        lookback_days=365,
        retain_context=True,
    ),
    CoTCandidate(
        name="higherr_k5_h18_ctx",
        history_points=18,
        prompt_history_points=18,
        k_examples=5,
        example_selection="recent_high_error",
        feature_window=18,
        lookback_days=365,
        retain_context=True,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retune clean TSM + CoT-RF on validation only.")
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("src/config/paper_llm_cot_rf_qwen_k5_h18_similarity_ctx_full.yaml"),
        help="Base YAML config used as the template for tuning.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("Data_auto"),
        help="Data directory for panel construction.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for tuning outputs. Defaults under reports/clean_retune/.",
    )
    parser.add_argument(
        "--llm-api-key",
        type=str,
        default=None,
        help="LLM API key. Falls back to OPENAI_API_KEY, then deo.",
    )
    parser.add_argument(
        "--llm-base-url",
        type=str,
        default=None,
        help="LLM base URL. Falls back to OPENAI_BASE_URL, then the base config value.",
    )
    parser.add_argument(
        "--cot-tune-samples",
        type=int,
        default=40,
        help="Number of validation samples used for CoT candidate selection.",
    )
    return parser.parse_args()


def deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as handle:
        return yaml.safe_load(handle)


def build_feature_cols(
    panel: pd.DataFrame,
    target_col: str,
    max_exogenous_features: int = 10,
    preferred_feature_order: list[str] | None = None,
) -> list[str]:
    return select_feature_columns(
        panel,
        target_col=target_col,
        max_exogenous_features=max_exogenous_features,
        preferred_feature_order=preferred_feature_order,
    )


def build_splits(panel: pd.DataFrame, raw_config: dict[str, Any]) -> dict[str, Any]:
    target_mode = str(raw_config.get("target", {}).get("mode", "price")).lower()
    target_col = "y_return" if target_mode == "returns" else "y"
    ts_cfg = raw_config["time_series"]
    feature_cfg = raw_config.get("features", {}) or {}
    feature_cols = build_feature_cols(
        panel,
        target_col,
        max_exogenous_features=int(feature_cfg.get("max_exogenous_features_model", 10)),
        preferred_feature_order=feature_cfg.get("preferred_feature_order"),
    )
    price_idx = feature_cols.index("y") if "y" in feature_cols else None
    if target_mode == "returns" and price_idx is None:
        raise ValueError("Returns target requires price feature 'y' in features.")

    window_config = WindowConfig(
        seq_len=int(ts_cfg["seq_len"]),
        label_len=int(ts_cfg["label_len"]),
        pred_len=int(ts_cfg["pred_len"]),
        target_col=target_col,
        feature_cols=feature_cols,
    )
    X_enc, X_dec, y, dates, meta = make_windows(
        panel,
        window_config,
        mode="MS",
        return_metadata=True,
    )
    split_cfg = raw_config["split"]
    splits = split_windows(
        X_enc,
        X_dec,
        y,
        dates,
        train_end=str(split_cfg["train_end"]),
        val_end=str(split_cfg["val_end"]),
        window_meta=meta,
    )

    if target_mode == "returns":
        y_train_hist = splits["train"]["X_enc"][:, :, price_idx]
        y_val_hist = splits["val"]["X_enc"][:, :, price_idx]
        y_test_hist = splits["test"]["X_enc"][:, :, price_idx]
        y_train_base = y_train_hist[:, -1]
        y_val_base = y_val_hist[:, -1]
        y_test_base = y_test_hist[:, -1]
        y_train_future = returns_to_prices(splits["train"]["y"], y_train_base)
        y_val_future = returns_to_prices(splits["val"]["y"], y_val_base)
        y_test_future = returns_to_prices(splits["test"]["y"], y_test_base)
    else:
        y_train_hist = splits["train"]["X_enc"][:, :, 0]
        y_val_hist = splits["val"]["X_enc"][:, :, 0]
        y_test_hist = splits["test"]["X_enc"][:, :, 0]
        y_train_base = y_train_hist[:, -1]
        y_val_base = y_val_hist[:, -1]
        y_test_base = y_test_hist[:, -1]
        y_train_future = splits["train"]["y"]
        y_val_future = splits["val"]["y"]
        y_test_future = splits["test"]["y"]

    return {
        "target_mode": target_mode,
        "target_col": target_col,
        "feature_cols": feature_cols,
        "window_config": window_config,
        "splits": splits,
        "y_train_hist": y_train_hist,
        "y_val_hist": y_val_hist,
        "y_test_hist": y_test_hist,
        "y_train_base": y_train_base,
        "y_val_base": y_val_base,
        "y_test_base": y_test_base,
        "y_train_future": y_train_future,
        "y_val_future": y_val_future,
        "y_test_future": y_test_future,
    }


def make_dataloaders(
    X_enc_train: np.ndarray,
    X_dec_train: np.ndarray,
    y_train: np.ndarray,
    X_enc_val: np.ndarray,
    X_dec_val: np.ndarray,
    y_val: np.ndarray,
    X_enc_test: np.ndarray,
    X_dec_test: np.ndarray,
    y_test: np.ndarray,
    raw_config: dict[str, Any],
) -> tuple[StandardScaler, dict[str, TimeSeriesDataset], dict[str, DataLoader]]:
    scaler = StandardScaler()
    scaler.fit(X_enc_train)

    datasets = {
        "train": TimeSeriesDataset(
            scaler.transform(X_enc_train),
            scaler.transform(X_dec_train),
            (y_train - scaler.mean_[0]) / scaler.std_[0],
        ),
        "val": TimeSeriesDataset(
            scaler.transform(X_enc_val),
            scaler.transform(X_dec_val),
            (y_val - scaler.mean_[0]) / scaler.std_[0],
        ),
        "test": TimeSeriesDataset(
            scaler.transform(X_enc_test),
            scaler.transform(X_dec_test),
            (y_test - scaler.mean_[0]) / scaler.std_[0],
        ),
    }

    batch_size = int(raw_config["model"].get("batch_size", 32))
    compute_cfg = raw_config.get("compute", {}) or {}
    num_workers = int(compute_cfg.get("num_workers", 0))
    pin_memory = bool(compute_cfg.get("pin_memory", False))

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }
    return scaler, datasets, loaders


def predict_prices(
    model: TSMForecaster,
    scaler: StandardScaler,
    loader: DataLoader,
    base_prices: np.ndarray,
    target_mode: str,
) -> np.ndarray:
    pred_scaled, _ = model.predict(loader)
    pred = scaler.inverse_transform_target(pred_scaled)
    if target_mode == "returns":
        return returns_to_prices(pred, base_prices)
    return pred


def prepare_candidate_config(base_raw: dict[str, Any], candidate: TSMCandidate) -> dict[str, Any]:
    raw = copy.deepcopy(base_raw)
    raw["target"]["mode"] = "returns"
    raw["time_series"]["seq_len"] = candidate.seq_len
    raw["time_series"]["label_len"] = candidate.label_len
    raw["time_series"]["pred_len"] = 30
    raw["model"]["tsm_type"] = "autoformer"
    raw["model"]["batch_size"] = candidate.batch_size
    raw["model"]["learning_rate"] = candidate.learning_rate
    raw["model"]["d_model"] = candidate.d_model
    raw["model"]["n_heads"] = candidate.n_heads
    raw["model"]["e_layers"] = candidate.e_layers
    raw["model"]["d_layers"] = 1
    raw["model"]["d_ff"] = candidate.d_ff
    raw["model"]["dropout"] = candidate.dropout
    raw["model"]["weight_decay"] = candidate.weight_decay
    raw["model"]["grad_clip"] = candidate.grad_clip
    raw["model"]["max_epochs"] = candidate.max_epochs
    raw["model"]["early_stopping_patience"] = candidate.patience
    raw["model"]["use_residual_wrapper"] = candidate.use_residual_wrapper
    raw["model"]["residual_alpha"] = candidate.residual_alpha
    raw["output"]["generate_paper"] = False
    raw["robustness"]["noise_levels"] = []
    return raw


def train_tsm_candidate(
    panel: pd.DataFrame,
    base_raw: dict[str, Any],
    candidate: TSMCandidate,
    splits_cache: dict[tuple[int, int], dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    raw = prepare_candidate_config(base_raw, candidate)
    cache_key = (candidate.seq_len, candidate.label_len)
    if cache_key not in splits_cache:
        splits_cache[cache_key] = build_splits(panel, raw)
    bundle = splits_cache[cache_key]

    feature_count = bundle["splits"]["train"]["X_enc"].shape[-1]
    raw["model"]["enc_in"] = feature_count
    raw["model"]["dec_in"] = feature_count

    set_seed(seed)
    scaler, datasets, loaders = make_dataloaders(
        bundle["splits"]["train"]["X_enc"],
        bundle["splits"]["train"]["X_dec"],
        bundle["splits"]["train"]["y"],
        bundle["splits"]["val"]["X_enc"],
        bundle["splits"]["val"]["X_dec"],
        bundle["splits"]["val"]["y"],
        bundle["splits"]["test"]["X_enc"],
        bundle["splits"]["test"]["X_dec"],
        bundle["splits"]["test"]["y"],
        raw,
    )

    model = TSMForecaster(raw)
    history = model.fit(
        loaders["train"],
        loaders["val"],
        epochs=int(raw["model"]["max_epochs"]),
        patience=int(raw["model"]["early_stopping_patience"]),
        save_path=None,
    )

    val_pred = predict_prices(
        model,
        scaler,
        loaders["val"],
        bundle["y_val_base"],
        bundle["target_mode"],
    )
    test_pred = predict_prices(
        model,
        scaler,
        loaders["test"],
        bundle["y_test_base"],
        bundle["target_mode"],
    )

    val_path = compute_path_metrics(bundle["y_val_future"], val_pred)
    test_path = compute_path_metrics(bundle["y_test_future"], test_pred)
    val_metrics = compute_metrics_by_horizon(
        bundle["y_val_future"],
        val_pred,
        [1, 5, 20, 30],
    )
    test_metrics = compute_metrics_by_horizon(
        bundle["y_test_future"],
        test_pred,
        [1, 5, 20, 30],
    )

    return {
        "name": candidate.name,
        "candidate": candidate,
        "raw_config": raw,
        "val_path_mse": float(val_path["mse_path"]),
        "test_path_mse": float(test_path["mse_path"]),
        "best_val_loss": float(model.best_val_loss),
        "epochs_ran": int(len(history["val_loss"])),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


def refit_best_tsm(
    panel: pd.DataFrame,
    raw: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    set_seed(seed)
    bundle = build_splits(panel, raw)
    feature_count = bundle["splits"]["train"]["X_enc"].shape[-1]
    raw = copy.deepcopy(raw)
    raw["model"]["enc_in"] = feature_count
    raw["model"]["dec_in"] = feature_count

    scaler, datasets, loaders = make_dataloaders(
        bundle["splits"]["train"]["X_enc"],
        bundle["splits"]["train"]["X_dec"],
        bundle["splits"]["train"]["y"],
        bundle["splits"]["val"]["X_enc"],
        bundle["splits"]["val"]["X_dec"],
        bundle["splits"]["val"]["y"],
        bundle["splits"]["test"]["X_enc"],
        bundle["splits"]["test"]["X_dec"],
        bundle["splits"]["test"]["y"],
        raw,
    )

    model = TSMForecaster(raw)
    history = model.fit(
        loaders["train"],
        loaders["val"],
        epochs=int(raw["model"]["max_epochs"]),
        patience=int(raw["model"]["early_stopping_patience"]),
        save_path=None,
    )

    train_loader_eval = DataLoader(
        datasets["train"],
        batch_size=int(raw["model"].get("batch_size", 32)),
        shuffle=False,
    )
    val_loader_eval = DataLoader(
        datasets["val"],
        batch_size=int(raw["model"].get("batch_size", 32)),
        shuffle=False,
    )
    test_loader_eval = DataLoader(
        datasets["test"],
        batch_size=int(raw["model"].get("batch_size", 32)),
        shuffle=False,
    )

    train_pred = predict_prices(
        model,
        scaler,
        train_loader_eval,
        bundle["y_train_base"],
        bundle["target_mode"],
    )
    val_pred = predict_prices(
        model,
        scaler,
        val_loader_eval,
        bundle["y_val_base"],
        bundle["target_mode"],
    )
    test_pred = predict_prices(
        model,
        scaler,
        test_loader_eval,
        bundle["y_test_base"],
        bundle["target_mode"],
    )

    panel_dates = pd.to_datetime(panel["date"]).to_numpy()
    date_to_idx = {pd.Timestamp(d): idx for idx, d in enumerate(panel_dates)}

    return {
        "raw_config": raw,
        "bundle": bundle,
        "model": model,
        "history": history,
        "train_pred": train_pred,
        "val_pred": val_pred,
        "test_pred": test_pred,
        "panel_dates": panel_dates,
        "date_to_idx": date_to_idx,
    }


def _history_features(history: np.ndarray, feature_window: int) -> np.ndarray:
    window = history[-feature_window:] if len(history) >= feature_window else history
    mean = float(np.mean(window)) if len(window) else 0.0
    std = float(np.std(window)) if len(window) else 0.0
    if len(window) > 1:
        slope = float(np.polyfit(np.arange(len(window)), window, 1)[0])
    else:
        slope = 0.0
    return np.array([mean, std, slope], dtype=float)


def prepare_example_pool(
    pool_dates: np.ndarray,
    pool_histories: np.ndarray,
    pool_forecasts: np.ndarray,
    pool_truth: np.ndarray,
    selection_mode: str,
    feature_window: int,
) -> dict[str, Any]:
    dates_arr = pd.to_datetime(pool_dates).to_numpy()
    order = np.argsort(dates_arr)
    prepared = {
        "dates": dates_arr[order],
        "histories": pool_histories[order],
        "forecasts": pool_forecasts[order],
        "truth": pool_truth[order],
    }
    prepared["path_mse"] = np.mean(
        (prepared["forecasts"] - prepared["truth"]) ** 2,
        axis=1,
    )
    if selection_mode == "similarity":
        prepared["history_features"] = np.vstack(
            [_history_features(hist, feature_window) for hist in prepared["histories"]]
        )
    return prepared


def select_example_indices(
    pool: dict[str, Any],
    reference_date: Any,
    reference_history: np.ndarray,
    candidate: CoTCandidate,
) -> np.ndarray:
    ref_date = pd.Timestamp(reference_date).to_datetime64()
    pos = np.searchsorted(pool["dates"], ref_date, side="left")
    if pos <= 0:
        return np.array([], dtype=int)

    candidate_indices = np.arange(pos, dtype=int)
    if candidate.example_selection in ("recent_high_error", "similarity") and candidate.lookback_days is not None:
        cutoff = np.datetime64(pd.Timestamp(reference_date) - pd.Timedelta(days=candidate.lookback_days))
        candidate_indices = candidate_indices[pool["dates"][candidate_indices] >= cutoff]
        if candidate_indices.size == 0:
            candidate_indices = np.arange(pos, dtype=int)

    if candidate.example_selection in ("high_error", "recent_high_error"):
        cand_errors = pool["path_mse"][candidate_indices]
        if len(candidate_indices) > candidate.k_examples:
            order = np.argsort(cand_errors)[-candidate.k_examples :]
            example_indices = candidate_indices[order]
        else:
            example_indices = candidate_indices
        return example_indices[np.argsort(pool["dates"][example_indices])]

    if candidate.example_selection == "similarity":
        sample_feat = _history_features(reference_history, candidate.feature_window)
        cand_feats = pool["history_features"][candidate_indices]
        distances = np.linalg.norm(cand_feats - sample_feat, axis=1)
        order = np.argsort(distances)[: candidate.k_examples]
        example_indices = candidate_indices[order]
        return example_indices[np.argsort(pool["dates"][example_indices])]

    start = max(0, pos - candidate.k_examples)
    return np.arange(start, pos, dtype=int)


def build_examples(pool: dict[str, Any], example_indices: np.ndarray) -> list[dict[str, Any]]:
    examples = []
    for idx in example_indices:
        examples.append(
            {
                "history": pool["histories"][idx],
                "forecast": pool["forecasts"][idx],
                "truth": pool["truth"][idx],
                "date": str(pd.Timestamp(pool["dates"][idx])),
                "sentiment_history": [],
            }
        )
    return examples


def evaluate_cot_candidate(
    base_raw: dict[str, Any],
    tsm_artifacts: dict[str, Any],
    candidate: CoTCandidate,
    output_dir: Path,
    api_key: str,
    base_url: str | None,
    tune_samples: int,
) -> dict[str, Any]:
    raw = copy.deepcopy(base_raw)
    deep_update(
        raw,
        {
            "llm": {
                "provider": "openai",
                "api_key": api_key,
                "base_url": base_url,
                "methods": ["TSM+LLM-COT-RF"],
                "max_samples": 100000,
                "history_points": candidate.history_points,
                "prompt_history_points": candidate.prompt_history_points,
                "subset": {"strategy": "first", "seed": 42, "val_strategy": "first", "val_seed": 42},
                "cot_rf": {
                    "k_examples": candidate.k_examples,
                    "example_selection": candidate.example_selection,
                    "feature_window": candidate.feature_window,
                    "lookback_days": candidate.lookback_days,
                    "retain_context": candidate.retain_context,
                    "strict_json_prompt": candidate.strict_json_prompt,
                    "strict_json_response_format": candidate.strict_json_response_format,
                },
            }
        },
    )

    run_cfg = tsm_artifacts["raw_config"]
    blend_grid_cfg = raw["llm"].get("blend_grid", {}) or {}
    blend_weights = [float(w) for w in blend_grid_cfg.get("weights", [0.0, 0.25, 0.5, 0.75, 1.0])]
    blend_schedule = str(blend_grid_cfg.get("schedule", "ramp")).lower()
    key_horizons = [int(h) for h in blend_grid_cfg.get("key_horizons", [1, 5, 10, 20, 30])]
    min_weight = float(blend_grid_cfg.get("min_weight", 0.0))
    power = float(blend_grid_cfg.get("power", 1.0))

    bundle = tsm_artifacts["bundle"]
    splits = bundle["splits"]
    y_train_hist = bundle["y_train_hist"]
    y_val_hist = bundle["y_val_hist"]
    y_val_future = bundle["y_val_future"]
    train_pred = tsm_artifacts["train_pred"]
    val_pred = tsm_artifacts["val_pred"]
    eligible_idx = np.arange(len(splits["val"]["dates"]), dtype=int)
    val_eval_indices = select_eval_indices(
        eligible_idx,
        max_samples=int(min(max(tune_samples, 1), len(eligible_idx))),
        strategy="spaced",
        seed=42,
    )

    val_pool = prepare_example_pool(
        splits["train"]["dates"],
        y_train_hist,
        train_pred,
        bundle["y_train_future"],
        candidate.example_selection,
        candidate.feature_window,
    )
    val_teaching_examples = []
    for idx in val_eval_indices:
        example_indices = select_example_indices(
            val_pool,
            splits["val"]["dates"][idx],
            y_val_hist[idx],
            candidate,
        )
        val_teaching_examples.append(build_examples(val_pool, example_indices))

    panel_dates = tsm_artifacts["panel_dates"]
    date_to_idx = tsm_artifacts["date_to_idx"]
    val_histories = y_val_hist[val_eval_indices][:, -candidate.history_points :]
    val_dates = build_history_dates(
        panel_dates,
        date_to_idx,
        splits["val"]["dates"][val_eval_indices],
        candidate.history_points,
    )

    cot_dir = output_dir / "llm" / candidate.name
    cot_dir.mkdir(parents=True, exist_ok=True)
    refiner = LLMRefiner(
        raw["llm"],
        cache_dir=cot_dir / "cache",
        log_dir=cot_dir / "logs",
    )

    predictions, metadata = refiner.refine_batch(
        method="TSM+LLM-COT-RF",
        histories=val_histories,
        date_arrays=val_dates,
        tsm_forecasts=val_pred[val_eval_indices],
        pred_len=int(run_cfg["time_series"]["pred_len"]),
        exogenous_summaries=[None for _ in range(len(val_dates))],
        price_bases=None,
        teaching_examples=val_teaching_examples,
        sentiment_histories=None,
    )

    y_val_subset = y_val_future[val_eval_indices]
    base_pred_subset = val_pred[val_eval_indices]
    raw_path = compute_path_metrics(y_val_subset, predictions)
    raw_metrics = compute_metrics_by_horizon(y_val_subset, predictions, [1, 5, 20, 30])

    blend_summary, blend_mse_by_h, blend_best_by_h = evaluate_blend_grid(
        y_true=y_val_subset,
        base_pred=base_pred_subset,
        llm_pred=predictions,
        weights=blend_weights,
        schedule=blend_schedule,
        key_horizons=key_horizons,
        min_weight=min_weight,
        power=power,
    )
    best_idx = blend_summary["mse_path"].idxmin()
    best_w = float(blend_summary.loc[best_idx, "w"])
    blend_pred = blend_forecasts(
        base_pred=base_pred_subset,
        llm_pred=predictions,
        strength=best_w,
        schedule=blend_schedule,
        pred_len=int(run_cfg["time_series"]["pred_len"]),
        min_weight=min_weight,
        power=power,
    )
    blend_path = compute_path_metrics(y_val_subset, blend_pred)
    blend_metrics = compute_metrics_by_horizon(y_val_subset, blend_pred, [1, 5, 20, 30])

    blend_h1base = blend_pred.copy()
    blend_h1base[:, 0] = base_pred_subset[:, 0]
    blend_h1base_path = compute_path_metrics(y_val_subset, blend_h1base)
    blend_h1base_metrics = compute_metrics_by_horizon(y_val_subset, blend_h1base, [1, 5, 20, 30])

    candidates = [
        ("raw", float(raw_path["mse_path"])),
        ("blend_ramp", float(blend_path["mse_path"])),
        ("blend_ramp_h1base", float(blend_h1base_path["mse_path"])),
    ]
    best_variant, best_variant_mse = min(candidates, key=lambda item: item[1])

    blend_summary.to_csv(cot_dir / "blend_summary_val.csv", index=False)
    blend_mse_by_h.to_csv(cot_dir / "blend_mse_by_h_val.csv", index=False)
    blend_best_by_h.to_csv(cot_dir / "blend_best_by_h_val.csv", index=False)
    with (cot_dir / "summary.json").open("w") as handle:
        json.dump(
            {
                "candidate": candidate.name,
                "raw_path_mse": float(raw_path["mse_path"]),
                "blend_path_mse": float(blend_path["mse_path"]),
                "blend_h1base_path_mse": float(blend_h1base_path["mse_path"]),
                "best_variant": best_variant,
                "best_variant_path_mse": best_variant_mse,
                "best_w": best_w,
                "n_val_samples": int(len(val_eval_indices)),
                "metadata_sample": metadata[0] if metadata else {},
            },
            handle,
            indent=2,
        )

    return {
        "name": candidate.name,
        "candidate": candidate,
        "best_variant": best_variant,
        "best_variant_path_mse": best_variant_mse,
        "best_w": best_w,
        "raw_path_mse": float(raw_path["mse_path"]),
        "blend_path_mse": float(blend_path["mse_path"]),
        "blend_h1base_path_mse": float(blend_h1base_path["mse_path"]),
        "n_val_samples": int(len(val_eval_indices)),
        "raw_metrics": raw_metrics,
        "blend_metrics": blend_metrics,
        "blend_h1base_metrics": blend_h1base_metrics,
    }


def candidate_rows(results: list[dict[str, Any]], stage: str) -> pd.DataFrame:
    rows = []
    for result in results:
        row = {"stage": stage, "name": result["name"]}
        for key, value in result.items():
            if isinstance(value, (float, int, str)):
                row[key] = value
        rows.append(row)
    return pd.DataFrame(rows)


def metric_lookup(metrics_df: pd.DataFrame) -> dict[str, float]:
    out = {}
    if "horizon" not in metrics_df.columns:
        metrics_df = metrics_df.reset_index()
    for _, row in metrics_df.iterrows():
        out[f"h{int(row['horizon'])}_mse"] = float(row["mse"])
    return out


def write_final_config(
    output_dir: Path,
    final_raw: dict[str, Any],
) -> Path:
    path = output_dir / "best_clean_tuned_config.yaml"
    with path.open("w") as handle:
        yaml.safe_dump(final_raw, handle, sort_keys=False)
    return path


def main() -> int:
    args = parse_args()
    base_raw = load_yaml(args.base_config)
    api_key = args.llm_api_key or os.environ.get("OPENAI_API_KEY") or "deo"
    base_url = args.llm_base_url or os.environ.get("OPENAI_BASE_URL") or base_raw.get("llm", {}).get("base_url")

    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = ROOT / "reports" / "clean_retune" / stamp
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "tuning_manifest.json").open("w") as handle:
        json.dump(
            {
                "base_config": str(args.base_config),
                "data_dir": str(args.data_dir),
                "llm_base_url": base_url,
                "cot_tune_samples": int(args.cot_tune_samples),
                "tsm_candidates": [candidate.__dict__ for candidate in TSM_CANDIDATES],
                "cot_candidates": [candidate.__dict__ for candidate in CoT_CANDIDATES],
            },
            handle,
            indent=2,
        )

    panel, _schema = build_panel(args.data_dir, base_raw)
    seed = int(base_raw.get("reproducibility", {}).get("seed", 42))
    splits_cache: dict[tuple[int, int], dict[str, Any]] = {}

    tsm_results = []
    for candidate in TSM_CANDIDATES:
        print(f"[TSM] {candidate.name}", flush=True)
        result = train_tsm_candidate(panel, base_raw, candidate, splits_cache, seed)
        result_row = {
            "name": result["name"],
            "val_path_mse": result["val_path_mse"],
            "test_path_mse": result["test_path_mse"],
            "best_val_loss": result["best_val_loss"],
            "epochs_ran": result["epochs_ran"],
        }
        result_row.update(metric_lookup(result["val_metrics"]))
        tsm_results.append({**result, **result_row})
        pd.DataFrame(
            [
                {
                    key: value
                    for key, value in row.items()
                    if isinstance(value, (str, int, float))
                }
                for row in tsm_results
            ]
        ).sort_values("val_path_mse").to_csv(output_dir / "tsm_search_results.partial.csv", index=False)
        print(
            f"[TSM] done {candidate.name} val_path_mse={result['val_path_mse']:.6f} "
            f"test_path_snapshot={result['test_path_mse']:.6f}",
            flush=True,
        )

    tsm_df = candidate_rows(tsm_results, stage="tsm").sort_values("val_path_mse")
    tsm_df.to_csv(output_dir / "tsm_search_results.csv", index=False)
    best_tsm = min(tsm_results, key=lambda item: item["val_path_mse"])

    print(f"[TSM] best={best_tsm['name']} val_path_mse={best_tsm['val_path_mse']:.6f}", flush=True)

    best_tsm_artifacts = refit_best_tsm(panel, best_tsm["raw_config"], seed)

    cot_results = []
    for candidate in CoT_CANDIDATES:
        print(f"[COT] {candidate.name}", flush=True)
        result = evaluate_cot_candidate(
            base_raw=best_tsm_artifacts["raw_config"],
            tsm_artifacts=best_tsm_artifacts,
            candidate=candidate,
            output_dir=output_dir,
            api_key=api_key,
            base_url=base_url,
            tune_samples=args.cot_tune_samples,
        )
        row = {
            "name": result["name"],
            "n_val_samples": result["n_val_samples"],
            "best_variant": result["best_variant"],
            "best_variant_path_mse": result["best_variant_path_mse"],
            "raw_path_mse": result["raw_path_mse"],
            "blend_path_mse": result["blend_path_mse"],
            "blend_h1base_path_mse": result["blend_h1base_path_mse"],
            "best_w": result["best_w"],
        }
        row.update({f"raw_{k}": v for k, v in metric_lookup(result["raw_metrics"]).items()})
        row.update({f"blend_{k}": v for k, v in metric_lookup(result["blend_metrics"]).items()})
        row.update({f"blend_h1base_{k}": v for k, v in metric_lookup(result["blend_h1base_metrics"]).items()})
        cot_results.append({**result, **row})
        pd.DataFrame(
            [
                {
                    key: value
                    for key, value in row.items()
                    if isinstance(value, (str, int, float))
                }
                for row in cot_results
            ]
        ).sort_values("best_variant_path_mse").to_csv(
            output_dir / "cot_search_results.partial.csv",
            index=False,
        )
        print(
            f"[COT] done {candidate.name} best_variant={result['best_variant']} "
            f"val_path_mse={result['best_variant_path_mse']:.6f}",
            flush=True,
        )

    cot_df = candidate_rows(cot_results, stage="cot").sort_values("best_variant_path_mse")
    cot_df.to_csv(output_dir / "cot_search_results.csv", index=False)
    best_cot = min(cot_results, key=lambda item: item["best_variant_path_mse"])

    final_raw = copy.deepcopy(best_tsm_artifacts["raw_config"])
    deep_update(
        final_raw,
        {
            "llm": {
                "provider": "openai",
                "api_key": api_key,
                "base_url": base_url,
                "methods": ["TSM+LLM-COT-RF"],
                "max_samples": 100000,
                "history_points": best_cot["candidate"].history_points,
                "prompt_history_points": best_cot["candidate"].prompt_history_points,
                "subset": {"strategy": "random", "seed": seed, "val_strategy": "random", "val_seed": seed},
                "cot_rf": {
                    "k_examples": best_cot["candidate"].k_examples,
                    "example_selection": best_cot["candidate"].example_selection,
                    "feature_window": best_cot["candidate"].feature_window,
                    "lookback_days": best_cot["candidate"].lookback_days,
                    "retain_context": best_cot["candidate"].retain_context,
                    "strict_json_prompt": best_cot["candidate"].strict_json_prompt,
                    "strict_json_response_format": best_cot["candidate"].strict_json_response_format,
                },
            }
        },
    )

    final_cfg_path = write_final_config(output_dir, final_raw)
    with (output_dir / "selection_summary.json").open("w") as handle:
        json.dump(
            {
                "best_tsm": {
                    "name": best_tsm["name"],
                    "val_path_mse": best_tsm["val_path_mse"],
                    "test_path_mse_snapshot": best_tsm["test_path_mse"],
                },
                "best_cot": {
                    "name": best_cot["name"],
                    "n_val_samples": best_cot["n_val_samples"],
                    "best_variant": best_cot["best_variant"],
                    "best_variant_path_mse": best_cot["best_variant_path_mse"],
                    "best_w": best_cot["best_w"],
                },
                "final_config_path": str(final_cfg_path),
            },
            handle,
            indent=2,
        )

    print(
        json.dumps(
            {
                "best_tsm": best_tsm["name"],
                "best_tsm_val_path_mse": best_tsm["val_path_mse"],
                "best_cot": best_cot["name"],
                "best_cot_variant": best_cot["best_variant"],
                "best_cot_val_path_mse": best_cot["best_variant_path_mse"],
                "final_config": str(final_cfg_path),
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
