#!/usr/bin/env python3
"""Validation-selected compact benchmark for frozen ETS2 UK ETS runs."""
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RUN_IDS = (
    "20260322_194840_381bee",  # W1
    "20260322_213347_11a76f",  # W2
    "20260322_230818_cd22ab",  # W3
    "20260323_003616_212adb",  # W4
)
HORIZONS = (1, 5, 20, 30)


@dataclass
class Split:
    x: np.ndarray
    y: np.ndarray
    dates: np.ndarray
    base: np.ndarray | None = None
    actual: np.ndarray | None = None


def load_split(path: Path, mean: float, std: float) -> Split:
    # The historical NPZ files store dates as an object array, so NumPy must
    # unpickle that field. These are repository-controlled frozen artifacts,
    # just like the adjacent scaler.pkl; never point this benchmark at an
    # untrusted run directory.
    with np.load(path, allow_pickle=True) as z:
        dates = pd.to_datetime(np.asarray(z["dates"]).reshape(-1), errors="raise")
        return Split(
            x=np.asarray(z["X_enc"], dtype=np.float64),
            y=np.asarray(z["y"], dtype=np.float64) * std + mean,
            dates=dates.strftime("%Y-%m-%d").to_numpy(dtype=str),
        )


def load_scaler(path: Path) -> tuple[float, float, list[str]]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    mean = float(np.asarray(payload["mean_"])[0])
    std = float(np.asarray(payload["std_"])[0]) or 1.0
    return mean, std, list(payload.get("feature_names_") or [])


def read_panel(run: Path) -> pd.DataFrame:
    parquet = run / "data/panel.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet)
    return pd.read_csv(run / "data/panel.csv")


def align_prices(panel: pd.DataFrame, split: Split, pred_len: int) -> tuple[np.ndarray, np.ndarray]:
    price_col = "y" if "y" in panel.columns else "close_native"
    p = panel[["date", price_col]].copy()
    p["date"] = pd.to_datetime(p["date"]).dt.normalize()
    p = p.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    dates = pd.DatetimeIndex(p["date"])
    positions = {d: i for i, d in enumerate(dates)}
    values = p[price_col].to_numpy(dtype=np.float64)
    base = np.empty(len(split.dates))
    future = np.empty((len(split.dates), pred_len))
    for row, raw_date in enumerate(split.dates):
        date = pd.Timestamp(str(raw_date)).normalize()
        idx = positions[date]
        base[row] = values[idx - 1]
        future[row] = values[idx : idx + pred_len]
    return base, future


def infer_transform(y: np.ndarray, base: np.ndarray, future: np.ndarray) -> str:
    log_price = base[:, None] * np.exp(np.cumsum(y, axis=1))
    simple_price = base[:, None] * np.cumprod(1.0 + y, axis=1)
    log_mae = float(np.mean(np.abs(log_price - future)))
    simple_mae = float(np.mean(np.abs(simple_price - future)))
    return "log" if log_mae <= simple_mae else "simple"


def prices_from_daily(y: np.ndarray, base: np.ndarray, transform: str) -> np.ndarray:
    if transform == "log":
        return base[:, None] * np.exp(np.cumsum(y, axis=1))
    return base[:, None] * np.cumprod(1.0 + y, axis=1)


def cumulative_target(y: np.ndarray, transform: str) -> np.ndarray:
    if transform == "log":
        return np.cumsum(y, axis=1)
    return np.cumprod(1.0 + y, axis=1) - 1.0


def prices_from_cumulative(y: np.ndarray, base: np.ndarray, transform: str) -> np.ndarray:
    if transform == "log":
        return base[:, None] * np.exp(y)
    return base[:, None] * (1.0 + y)


def slope(x: np.ndarray) -> np.ndarray:
    t = np.arange(x.shape[1], dtype=np.float64)
    t -= t.mean()
    return np.einsum("ntc,t->nc", x, t) / max(float(t @ t), 1.0)


def features(x: np.ndarray, mode: str) -> np.ndarray:
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if mode == "target":
        x = x[:, :, :1]
    n, steps, channels = x.shape
    parts: list[np.ndarray] = []
    if mode in {"target", "full"}:
        parts.append(x.reshape(n, steps * channels))
    parts.append(x[:, -1, :])
    for window in (3, 5, 10, 20):
        recent = x[:, -min(window, steps) :, :]
        parts += [
            recent.mean(1),
            recent.std(1),
            recent.min(1),
            recent.max(1),
            recent[:, -1, :] - recent[:, 0, :],
            slope(recent),
        ]
    return np.column_stack(parts)


def mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean((actual - predicted) ** 2))


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    out = {
        "mse_path": mse(actual, predicted),
        "mae_path": float(np.mean(np.abs(actual - predicted))),
    }
    for h in HORIZONS:
        if h <= actual.shape[1]:
            out[f"mse_h{h}"] = mse(actual[:, h - 1], predicted[:, h - 1])
    return out


def candidate_grid(seed: int):
    for mode in ("target", "summary", "full"):
        for alpha in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0):
            yield (
                f"ridge_{mode}_a{alpha:g}",
                mode,
                lambda a=alpha: make_pipeline(StandardScaler(), Ridge(alpha=a)),
            )
    for mode in ("summary", "full"):
        for leaf in (2, 5, 10):
            for max_features in (0.5, 1.0):
                yield (
                    f"extra_trees_{mode}_leaf{leaf}_mf{max_features:g}",
                    mode,
                    lambda l=leaf, m=max_features: ExtraTreesRegressor(
                        n_estimators=180,
                        min_samples_leaf=l,
                        max_features=m,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                )


def load_reported(run: Path) -> dict[str, float]:
    path = run / "results/path_metrics.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    frame = frame[frame["subset"].astype(str) == "full"]
    return {str(r.model): float(r.mse_path) for r in frame.itertuples()}


def benchmark_run(run: Path, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with (run / "config_resolved.yaml").open() as f:
        config = yaml.safe_load(f) or {}
    mean, std, names = load_scaler(run / "data/datasets/scaler.pkl")
    train = load_split(run / "data/datasets/train.npz", mean, std)
    val = load_split(run / "data/datasets/val.npz", mean, std)
    test = load_split(run / "data/datasets/test.npz", mean, std)
    panel = read_panel(run)
    for split in (train, val, test):
        split.base, future = align_prices(panel, split, split.y.shape[1])
        split.actual = future
    transform = infer_transform(test.y, test.base, test.actual)
    alignment_mae = float(np.mean(np.abs(prices_from_daily(test.y, test.base, transform) - test.actual)))

    train_target = cumulative_target(train.y, transform)
    val_target = cumulative_target(val.y, transform)
    validation: list[dict[str, Any]] = []
    selected: tuple[float, str, str, Any, float] | None = None
    for name, mode, factory in candidate_grid(seed):
        model = factory()
        model.fit(features(train.x, mode), train_target)
        raw = model.predict(features(val.x, mode))
        low = np.quantile(train_target, 0.0025, axis=0)
        high = np.quantile(train_target, 0.9975, axis=0)
        raw = np.clip(raw, low - 0.25 * (high - low), high + 0.25 * (high - low))
        for weight in np.linspace(0.0, 1.0, 21):
            score = mse(val.actual, prices_from_cumulative(raw * weight, val.base, transform))
            if selected is None or (score, name, weight) < (selected[0], selected[1], selected[4]):
                selected = (score, name, mode, factory, float(weight))
        validation.append({"run_id": run.name, "candidate": name, "best_validation_mse": min(
            mse(val.actual, prices_from_cumulative(raw * w, val.base, transform))
            for w in np.linspace(0.0, 1.0, 21)
        )})

    assert selected is not None
    validation_mse, selected_name, selected_mode, factory, weight = selected
    x_fit = np.vstack([features(train.x, selected_mode), features(val.x, selected_mode)])
    y_fit = np.vstack([train_target, val_target])
    model = factory()
    model.fit(x_fit, y_fit)
    raw_test = model.predict(features(test.x, selected_mode))
    low = np.quantile(y_fit, 0.0025, axis=0)
    high = np.quantile(y_fit, 0.9975, axis=0)
    raw_test = np.clip(raw_test, low - 0.25 * (high - low), high + 0.25 * (high - low))
    predicted = prices_from_cumulative(raw_test * weight, test.base, transform)
    result: dict[str, Any] = {
        "run_id": run.name,
        "selected_candidate": selected_name,
        "selected_feature_mode": selected_mode,
        "shrinkage": weight,
        "validation_mse_path": validation_mse,
        "transform": transform,
        "alignment_mae": alignment_mae,
        "n_train": len(train.y),
        "n_val": len(val.y),
        "n_test": len(test.y),
        "n_features": train.x.shape[2],
        "feature_names": json.dumps(names),
        **metrics(test.actual, predicted),
    }
    reported = load_reported(run)
    for name, value in reported.items():
        result[f"reported_{name}_mse"] = value
    for ref in ("tsm", "linear_ridge", "naive_persistence"):
        if ref in reported:
            result[f"improvement_vs_{ref}_pct"] = 100.0 * (reported[ref] - result["mse_path"]) / reported[ref]
    result["target_mode"] = str(config.get("target", {}).get("mode", ""))
    return result, validation


def write_summary(results: pd.DataFrame, path: Path) -> None:
    cols = [
        "run_id",
        "selected_candidate",
        "mse_path",
        "reported_tsm_mse",
        "reported_linear_ridge_mse",
        "improvement_vs_tsm_pct",
        "improvement_vs_linear_ridge_pct",
    ]
    cols = [c for c in cols if c in results.columns]
    lines = [
        "# Frozen compact-model benchmark",
        "",
        "Candidate family, regularisation and shrinkage were selected on validation data only. The chosen model was then refit on train+validation and evaluated once on the frozen test split.",
        "",
        results[cols].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Aggregate",
        "",
    ]
    for col in ("improvement_vs_tsm_pct", "improvement_vs_linear_ridge_pct", "improvement_vs_naive_persistence_pct"):
        if col in results:
            lines.append(f"- Mean {col}: **{results[col].mean():.2f}%**; wins: **{int((results[col] > 0).sum())}/{len(results)}**")
    lines += [
        "",
        "The benchmark also records price-reconstruction alignment. A near-zero `alignment_mae` confirms that the saved target windows were converted back to price paths correctly.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--run-ids", nargs="*", default=list(RUN_IDS))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/agent_compact_benchmark"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for run_id in args.run_ids:
        try:
            result, candidate_rows = benchmark_run(args.runs_root / run_id, args.seed)
            results.append(result)
            validation.extend(candidate_rows)
            print(f"{run_id}: {result['selected_candidate']} MSE={result['mse_path']:.6f}")
        except Exception as exc:
            errors.append({"run_id": run_id, "error": f"{type(exc).__name__}: {exc}"})
            print(f"{run_id}: failed: {exc}")
    frame = pd.DataFrame(results)
    pd.DataFrame(validation).to_csv(args.output_dir / "candidate_validation.csv", index=False)
    pd.DataFrame(errors).to_csv(args.output_dir / "errors.csv", index=False)
    if frame.empty:
        return 1
    frame.to_csv(args.output_dir / "benchmark_results.csv", index=False)
    write_summary(frame, args.output_dir / "summary.md")
    manifest = {
        "protocol": "select on validation, refit on train+validation, evaluate test once",
        "seed": args.seed,
        "run_ids": args.run_ids,
        "errors": errors,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
