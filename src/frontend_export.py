from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_MODEL_PRIORITY = [
    "TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base",
    "TSM+LLM-COT-RF_blend_ramp_bestval_path",
    "TSM+LLM-COT-RF",
    "TSM+LLM-COT-SENT-RF_blend_ramp_bestval_path_h1base",
    "TSM+LLM-COT-SENT-RF_blend_ramp_bestval_path",
    "TSM+LLM-COT-SENT-RF",
    "tsm",
]


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_path_metrics(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "results" / "path_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing path metrics: {path}")
    return pd.read_csv(path)


def _read_metrics_by_horizon(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "results" / "metrics_by_horizon.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics-by-horizon: {path}")
    return pd.read_csv(path)


def _method_slug(method: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", method)


def _load_tsm_predictions(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "predictions" / "tsm_pred_test.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing TSM predictions: {path}")

    df = pd.read_parquet(path)
    pred_cols = [f"yhat_t_plus_{i}" for i in range(1, 31)]
    true_cols = [f"y_true_t_plus_{i}" for i in range(1, 31)]
    return {
        "dates": pd.to_datetime(df["date"]).astype(str).to_numpy(),
        "y_true": df[true_cols].to_numpy(dtype=float),
        "yhat": df[pred_cols].to_numpy(dtype=float),
        "subset": "full",
    }


def _load_llm_subset_predictions(run_dir: Path, method: str) -> dict[str, Any]:
    path = run_dir / "predictions" / f"{method}_pred_test_subset.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing LLM predictions for {method}: {path}")

    data = np.load(path, allow_pickle=True)
    return {
        "dates": np.asarray(data["dates"]).astype(str),
        "y_true": np.asarray(data["y_true"], dtype=float),
        "yhat": np.asarray(data["yhat"], dtype=float),
        "tsm_pred": np.asarray(data["tsm_pred"], dtype=float),
        "subset": "llm_subset",
    }


def _blend_forecasts(
    base_pred: np.ndarray,
    llm_pred: np.ndarray,
    strength: float,
    schedule: str,
    pred_len: int,
    min_weight: float = 0.0,
    power: float = 1.0,
) -> np.ndarray:
    schedule = str(schedule or "uniform").lower()
    if schedule == "uniform":
        return base_pred + float(strength) * (llm_pred - base_pred)
    if schedule != "ramp":
        raise ValueError(f"Unknown blend schedule: {schedule}")
    weights = np.linspace(float(min_weight), float(strength), int(pred_len))
    if float(power) != 1.0:
        weights = np.power(weights, float(power))
    return base_pred * (1.0 - weights) + llm_pred * weights


def _load_blend_selection(run_dir: Path, method: str) -> dict[str, Any]:
    path = run_dir / "llm" / f"blend_grid_selection_{_method_slug(method)}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing blend selection file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_blend_predictions(run_dir: Path, model_name: str) -> dict[str, Any]:
    method, blend_suffix = model_name.split("_blend_", 1)
    llm = _load_llm_subset_predictions(run_dir, method)
    selection = _load_blend_selection(run_dir, method)

    override_h1 = False
    if blend_suffix.endswith("_h1base"):
        override_h1 = True
        blend_suffix = blend_suffix[: -len("_h1base")]

    parts = blend_suffix.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unsupported blend model name: {model_name}")

    schedule = parts[0]
    target_spec = "_".join(parts[1:])

    if target_spec == "bestval_path":
        weight = float(selection["best_w_path"])
    else:
        match = re.fullmatch(r"bestval_h(\d+)", target_spec)
        if not match:
            raise ValueError(f"Unsupported blend target spec: {target_spec}")
        horizon = match.group(1)
        weight = float(selection["best_w_by_horizon"][horizon])

    pred = _blend_forecasts(
        base_pred=llm["tsm_pred"],
        llm_pred=llm["yhat"],
        strength=weight,
        schedule=selection.get("schedule", schedule),
        pred_len=llm["yhat"].shape[1],
        min_weight=float(selection.get("min_weight", 0.0)),
        power=float(selection.get("power", 1.0)),
    )
    if override_h1 and pred.shape[1] > 0:
        pred = pred.copy()
        pred[:, 0] = llm["tsm_pred"][:, 0]

    return {
        "dates": llm["dates"],
        "y_true": llm["y_true"],
        "yhat": pred,
        "subset": llm["subset"],
    }


def load_model_predictions(run_dir: Path, model_name: str) -> dict[str, Any]:
    if model_name == "tsm":
        return _load_tsm_predictions(run_dir)
    if "_blend_" in model_name:
        return _resolve_blend_predictions(run_dir, model_name)
    return _load_llm_subset_predictions(run_dir, model_name)


def select_default_model(path_metrics: pd.DataFrame) -> str:
    available = set(path_metrics["model"].astype(str).tolist())
    for model_name in DEFAULT_MODEL_PRIORITY:
        if model_name in available:
            return model_name
    ranked = _prefer_subset(path_metrics)
    if ranked.empty:
        raise ValueError("No models available in path_metrics.csv")
    return str(ranked.sort_values("mse_path").iloc[0]["model"])


def _collect_recent_history(run_dir: Path, history_days: int) -> tuple[list[dict[str, Any]], str | None, float | None]:
    panel_path = run_dir / "data" / "panel.parquet"
    if not panel_path.exists():
        return [], None, None

    panel = pd.read_parquet(panel_path)
    if "date" not in panel.columns or "y" not in panel.columns:
        return [], None, None

    tail = panel[["date", "y"]].dropna().tail(int(history_days)).copy()
    tail["date"] = pd.to_datetime(tail["date"]).dt.strftime("%Y-%m-%d")
    history = [
        {"date": str(row.date), "price": float(row.y)}
        for row in tail.itertuples(index=False)
    ]
    if not history:
        return history, None, None
    last = history[-1]
    return history, str(last["date"]), float(last["price"])


def _collect_model_metrics(
    path_metrics: pd.DataFrame,
    metrics_by_h: pd.DataFrame,
    model_name: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}

    pm = path_metrics[path_metrics["model"] == model_name]
    if not pm.empty:
        row = _prefer_subset(pm).iloc[0]
        payload["subset"] = str(row.get("subset", ""))
        payload["n_samples"] = int(row.get("n_samples", 0))
        payload["path_mse"] = float(row["mse_path"])
        if "rmse_path" in row:
            payload["path_rmse"] = float(row["rmse_path"])
        if "mae_path" in row:
            payload["path_mae"] = float(row["mae_path"])

    mh = metrics_by_h[metrics_by_h["model"] == model_name]
    if not mh.empty:
        mh = _prefer_subset(mh)
        horizons: dict[str, Any] = {}
        for row in mh.itertuples(index=False):
            horizons[f"h{int(row.horizon)}"] = {
                "mse": float(row.mse),
                "rmse": float(row.rmse),
                "mae": float(row.mae),
            }
        payload["horizons"] = horizons

    return payload


def _build_model_comparison(path_metrics: pd.DataFrame, metrics_by_h: pd.DataFrame) -> dict[str, Any]:
    rows = []
    for model_name in _prefer_subset(path_metrics)["model"].astype(str).unique().tolist():
        row = {"model": model_name}
        row.update(_collect_model_metrics(path_metrics, metrics_by_h, model_name))
        rows.append(row)
    return {"models": rows}


def export_frontend_artifacts(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    publish_dir: str | Path | None = None,
    model_name: str | None = None,
    history_days: int = 90,
) -> Path:
    run_dir = Path(run_dir).resolve()
    if output_dir is None:
        output_dir = run_dir / "frontend_export"
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    path_metrics = _read_path_metrics(run_dir)
    metrics_by_h = _read_metrics_by_horizon(run_dir)
    selected_model = model_name or select_default_model(path_metrics)
    pred = load_model_predictions(run_dir, selected_model)
    recent_history, data_cutoff_date, last_observed_price = _collect_recent_history(
        run_dir=run_dir,
        history_days=history_days,
    )

    latest_idx = int(len(pred["dates"]) - 1)
    origin_date = str(pd.Timestamp(str(pred["dates"][latest_idx])).date())
    latest_forecast = [
        {
            "horizon": int(h + 1),
            "label": f"t+{int(h + 1)}",
            "prediction": float(pred["yhat"][latest_idx, h]),
            "actual": float(pred["y_true"][latest_idx, h]),
        }
        for h in range(pred["yhat"].shape[1])
    ]

    model_metrics = _collect_model_metrics(path_metrics, metrics_by_h, selected_model)
    comparison_payload = _build_model_comparison(path_metrics, metrics_by_h)

    latest_payload = {
        "artifact_version": 1,
        "exported_at_utc": _iso_utc_now(),
        "run_id": run_dir.name,
        "forecast_kind": "backtest_latest_window",
        "model": selected_model,
        "origin_date": origin_date,
        "data_cutoff_date": data_cutoff_date,
        "last_observed_price": last_observed_price,
        "metrics": model_metrics,
        "recent_history": recent_history,
        "forecast": latest_forecast,
    }
    manifest_payload = {
        "artifact_version": 1,
        "exported_at_utc": _iso_utc_now(),
        "run_id": run_dir.name,
        "source_run_dir": str(run_dir),
        "selected_model": selected_model,
        "files": [
            "manifest.json",
            "latest_forecast.json",
            "model_comparison.json",
            "recent_history.json",
        ],
    }
    history_payload = {
        "artifact_version": 1,
        "exported_at_utc": _iso_utc_now(),
        "run_id": run_dir.name,
        "data_cutoff_date": data_cutoff_date,
        "history": recent_history,
    }

    _write_json(output_dir / "latest_forecast.json", latest_payload)
    _write_json(output_dir / "model_comparison.json", comparison_payload)
    _write_json(output_dir / "recent_history.json", history_payload)
    _write_json(output_dir / "manifest.json", manifest_payload)

    if publish_dir is not None:
        publish_dir = Path(publish_dir).resolve()
        publish_dir.mkdir(parents=True, exist_ok=True)
        for name in ("manifest.json", "latest_forecast.json", "model_comparison.json", "recent_history.json"):
            shutil.copy2(output_dir / name, publish_dir / name)

    return output_dir
