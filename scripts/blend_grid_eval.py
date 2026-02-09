#!/usr/bin/env python3
"""
Evaluate a grid of TSM↔LLM blend weights from a saved subset NPZ.

This is intentionally lightweight so we can tune blending strength without
re-running expensive LLM calls.

Expected NPZ keys:
  - y_true: (n, pred_len)
  - tsm_pred: (n, pred_len)
  - yhat: (n, pred_len)  # LLM refined
  - dates: (n,)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple, Literal

import numpy as np
import pandas as pd


def _parse_weights(text: str) -> List[float]:
    weights = []
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        weights.append(float(part))
    if not weights:
        raise ValueError("No weights provided")
    return weights


def _mse_by_horizon(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.mean((y_true - y_pred) ** 2, axis=0)


def _blend(
    base_pred: np.ndarray,
    llm_pred: np.ndarray,
    strength: float,
    schedule: Literal["uniform", "ramp"],
    pred_len: int,
    min_weight: float = 0.0,
    power: float = 1.0,
) -> np.ndarray:
    """Blend predictions with either a uniform weight or a horizon ramp."""
    strength = float(strength)
    if schedule == "uniform":
        return base_pred + strength * (llm_pred - base_pred)

    if schedule != "ramp":
        raise ValueError(f"Unknown schedule: {schedule}")

    w = np.linspace(float(min_weight), strength, pred_len)
    if power != 1.0:
        w = np.power(w, float(power))
    return base_pred * (1.0 - w) + llm_pred * w


def _ramp_weight_at_horizon(
    strength: float,
    horizon: int,
    pred_len: int,
    min_weight: float = 0.0,
    power: float = 1.0,
) -> float:
    """Return the ramp weight applied at a single horizon (1-based)."""
    horizon = int(horizon)
    if horizon < 1 or horizon > pred_len:
        raise ValueError("horizon out of range")
    if pred_len <= 1:
        w = float(strength)
    else:
        w = float(min_weight) + (float(strength) - float(min_weight)) * float(horizon - 1) / float(pred_len - 1)
    if power != 1.0:
        w = float(np.power(w, float(power)))
    return float(w)


def _summarize_grid(
    y_true: np.ndarray,
    base_pred: np.ndarray,
    llm_pred: np.ndarray,
    weights: Iterable[float],
    key_horizons: Iterable[int],
    schedule: Literal["uniform", "ramp"] = "uniform",
    min_weight: float = 0.0,
    power: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred_len = int(base_pred.shape[1])

    grid_rows = []
    mse_rows = []

    mse_base = _mse_by_horizon(y_true, base_pred)
    for w in weights:
        blended = _blend(
            base_pred=base_pred,
            llm_pred=llm_pred,
            strength=float(w),
            schedule=schedule,
            pred_len=pred_len,
            min_weight=min_weight,
            power=power,
        )
        mse_h = _mse_by_horizon(y_true, blended)
        row = {"w": float(w), "mse_path": float(np.mean(mse_h))}
        for h in key_horizons:
            if 1 <= int(h) <= pred_len:
                row[f"h{int(h)}_mse"] = float(mse_h[int(h) - 1])
        grid_rows.append(row)
        for h in range(1, pred_len + 1):
            mse_rows.append({"w": float(w), "horizon": int(h), "mse": float(mse_h[h - 1])})

    summary = pd.DataFrame(grid_rows).sort_values("w")
    mse_by_h = pd.DataFrame(mse_rows).sort_values(["horizon", "w"])

    best_by_h = (
        mse_by_h.loc[mse_by_h.groupby("horizon")["mse"].idxmin()]
        .sort_values("horizon")
        .reset_index(drop=True)
    )

    # "Oracle" blended path: choose the best *grid strength* at each horizon.
    # This is a diagnostic upper-bound (it uses the same dataset to select),
    # but it helps reveal whether horizon-dependent blending is structurally
    # promising.
    oracle_pred = base_pred.copy()
    if schedule == "uniform":
        w_oracle = np.zeros(pred_len, dtype=float)
        for _, r in best_by_h.iterrows():
            w_oracle[int(r["horizon"]) - 1] = float(r["w"])
        oracle_pred = base_pred + (llm_pred - base_pred) * w_oracle
    else:
        for _, r in best_by_h.iterrows():
            h = int(r["horizon"])
            strength = float(r["w"])
            w_h = _ramp_weight_at_horizon(
                strength=strength,
                horizon=h,
                pred_len=pred_len,
                min_weight=min_weight,
                power=power,
            )
            oracle_pred[:, h - 1] = base_pred[:, h - 1] * (1.0 - w_h) + llm_pred[:, h - 1] * w_h
    mse_oracle = _mse_by_horizon(y_true, oracle_pred)
    oracle_summary = {
        "mse_path": float(np.mean(mse_oracle)),
        **{f"h{int(h)}_mse": float(mse_oracle[int(h) - 1]) for h in key_horizons if 1 <= int(h) <= pred_len},
    }
    oracle_df = pd.DataFrame([oracle_summary])

    base_summary = {
        "mse_path": float(np.mean(mse_base)),
        **{f"h{int(h)}_mse": float(mse_base[int(h) - 1]) for h in key_horizons if 1 <= int(h) <= pred_len},
    }
    base_df = pd.DataFrame([base_summary])

    return summary, mse_by_h, best_by_h, pd.concat(
        [
            base_df.assign(model="TSM"),
            oracle_df.assign(model="OraclePerHorizon"),
        ],
        ignore_index=True,
    )


def _save_plots(
    mse_by_h: pd.DataFrame,
    best_by_h: pd.DataFrame,
    out_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: MSE by horizon for each w
    fig, ax = plt.subplots(figsize=(12, 6))
    for w, grp in mse_by_h.groupby("w", sort=True):
        ax.plot(grp["horizon"], grp["mse"], label=f"w={w:g}", linewidth=2)
    ax.set_title("Blend Grid — MSE by Horizon (TSM + w·(LLM−TSM))")
    ax.set_xlabel("Horizon (days ahead)")
    ax.set_ylabel("MSE (EUR^2)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "blend_grid_mse_by_horizon.png", dpi=200)
    plt.close(fig)

    # Plot 2: Best w by horizon (discrete choices)
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.step(best_by_h["horizon"], best_by_h["w"], where="mid", linewidth=2)
    ax.set_title("Blend Grid — Best w by Horizon (min MSE at each horizon)")
    ax.set_xlabel("Horizon (days ahead)")
    ax.set_ylabel("Best w")
    ax.set_yticks(sorted(best_by_h["w"].unique()))
    ax.set_ylim(min(best_by_h["w"].min() - 0.05, -0.05), best_by_h["w"].max() + 0.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "blend_grid_best_w_by_horizon.png", dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, required=True, help="Path to *_pred_test_subset.npz")
    parser.add_argument(
        "--weights",
        type=str,
        default="0,0.25,0.5,0.75,1.0",
        help="Comma-separated blend weights w (TSM + w*(LLM-TSM))",
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default="uniform",
        choices=["uniform", "ramp"],
        help="Blend schedule: uniform w, or horizon ramp from min_weight→w",
    )
    parser.add_argument(
        "--min-weight",
        type=float,
        default=0.0,
        help="Ramp schedule: starting weight at horizon 1 (default 0.0)",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=1.0,
        help="Ramp schedule: exponent applied to the weight curve (default 1.0)",
    )
    parser.add_argument(
        "--key-horizons",
        type=str,
        default="1,5,10,20,30",
        help="Comma-separated horizons to include in summary table",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory (default: alongside run results)",
    )
    args = parser.parse_args()

    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    weights = _parse_weights(args.weights)
    key_horizons = [int(x) for x in _parse_weights(args.key_horizons)]

    data = np.load(npz_path, allow_pickle=False)
    y_true = np.asarray(data["y_true"], dtype=float)
    base_pred = np.asarray(data["tsm_pred"], dtype=float)
    llm_pred = np.asarray(data["yhat"], dtype=float)

    if y_true.shape != base_pred.shape or y_true.shape != llm_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true{y_true.shape} base{base_pred.shape} llm{llm_pred.shape}"
        )

    # Infer run dir for default outputs.
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir is None:
        # runs/<id>/predictions/<file>.npz -> runs/<id>/results/blend_grid
        maybe_run = npz_path.parents[1]
        out_dir = maybe_run / "results" / f"blend_grid_{args.schedule}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary, mse_by_h, best_by_h, anchors = _summarize_grid(
        y_true=y_true,
        base_pred=base_pred,
        llm_pred=llm_pred,
        weights=weights,
        key_horizons=key_horizons,
        schedule=args.schedule,
        min_weight=float(args.min_weight),
        power=float(args.power),
    )

    summary.to_csv(out_dir / "blend_grid_summary.csv", index=False)
    mse_by_h.to_csv(out_dir / "blend_grid_mse_by_horizon.csv", index=False)
    best_by_h.to_csv(out_dir / "blend_grid_best_w_by_horizon.csv", index=False)
    anchors.to_csv(out_dir / "blend_grid_anchors.csv", index=False)

    _save_plots(mse_by_h=mse_by_h, best_by_h=best_by_h, out_dir=out_dir)

    # Print a compact console summary
    print("\nBlend grid summary (key horizons):")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nAnchors (TSM baseline vs Oracle per-horizon best):")
    print(anchors.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nWrote: {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
