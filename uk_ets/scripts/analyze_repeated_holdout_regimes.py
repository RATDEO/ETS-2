#!/usr/bin/env python3
"""Diagnose why W0 outperforms other repeated production holdouts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPEATED_REPORT_DIR = PROJECT_ROOT / "reports" / "uk_ets_repeated_production_holdout" / "20260316_220005"
WINDOW_RESULTS_PATH = REPEATED_REPORT_DIR / "window_results.csv"


@dataclass(frozen=True)
class WindowInfo:
    window: str
    run_dir: Path
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    test_end: pd.Timestamp

    @property
    def test_start(self) -> pd.Timestamp:
        return self.val_end + pd.Timedelta(days=1)


CORE_FEATURES = [
    "positive_count",
    "positive_count_h20",
    "positive_count_h30",
    "positive_signal",
    "positive_signal_h20",
    "positive_signal_h30",
    "negative_count",
    "negative_signal",
    "negative_signal_h30",
    "positive_best_similarity",
    "positive_mean_similarity",
    "positive_mean_helpfulness",
    "negative_mean_helpfulness",
    "base_move_h5_pct",
    "base_move_h20_pct",
    "base_move_h30_pct",
    "profile_change_5",
    "profile_vol_pct",
    "profile_fc_h5",
    "profile_fc_h20",
    "profile_fc_h30",
]


def _load_windows() -> list[WindowInfo]:
    df = pd.read_csv(WINDOW_RESULTS_PATH)
    windows: list[WindowInfo] = []
    for row in df.itertuples(index=False):
        windows.append(
            WindowInfo(
                window=row.window,
                run_dir=Path(row.run_dir),
                train_end=pd.Timestamp(row.train_end),
                val_end=pd.Timestamp(row.val_end),
                test_end=pd.Timestamp(row.test_end),
            )
        )
    return windows


def _gate_paths(run_dir: Path) -> Path:
    return run_dir / "results" / "online_memory_gate" / "TSM_LLM-COT-RF-HDELTA"


def _window_panel_slice(win: WindowInfo) -> pd.DataFrame:
    panel = pd.read_parquet(win.run_dir / "data" / "panel.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    mask = (panel["date"] >= win.test_start) & (panel["date"] <= win.test_end)
    return panel.loc[mask].copy().reset_index(drop=True)


def _plot_window_shapes(windows: list[WindowInfo], out_dir: Path) -> None:
    fig, axes = plt.subplots(len(windows), 1, figsize=(11, 2.4 * len(windows)), sharex=False)
    if len(windows) == 1:
        axes = [axes]
    for ax, win in zip(axes, windows):
        df = _window_panel_slice(win)
        base = float(df["y"].iloc[0])
        norm = df["y"] / base * 100.0
        ax.plot(df["date"], norm, color="#0b5d7a", lw=1.8)
        ax.set_title(f"{win.window} | normalized UKA price path", loc="left", fontsize=10)
        ax.set_ylabel("Index=100")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "window_price_shapes.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_window_volatility(windows: list[WindowInfo], out_dir: Path) -> None:
    fig, axes = plt.subplots(len(windows), 1, figsize=(11, 2.4 * len(windows)), sharex=False)
    if len(windows) == 1:
        axes = [axes]
    for ax, win in zip(axes, windows):
        df = _window_panel_slice(win)
        ax.plot(df["date"], df["y_vol_20d"], color="#7a2e0b", lw=1.8)
        ax.set_title(f"{win.window} | trailing 20d volatility proxy", loc="left", fontsize=10)
        ax.set_ylabel("y_vol_20d")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "window_volatility_shapes.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_overlay(windows: list[WindowInfo], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for win in windows:
        df = _window_panel_slice(win)
        base = float(df["y"].iloc[0])
        norm = (df["y"] / base * 100.0).reset_index(drop=True)
        axes[0].plot(norm.index, norm.values, lw=1.8, label=win.window.split("_")[0])
        axes[1].plot(df["y_vol_20d"].reset_index(drop=True).index, df["y_vol_20d"].reset_index(drop=True).values, lw=1.8, label=win.window.split("_")[0])
    axes[0].set_title("Normalized price paths")
    axes[0].set_ylabel("Index=100")
    axes[0].set_xlabel("Test-window step")
    axes[0].grid(alpha=0.25)
    axes[1].set_title("Volatility proxy paths")
    axes[1].set_ylabel("y_vol_20d")
    axes[1].set_xlabel("Test-window step")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best", ncol=1, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "window_overlay_shapes.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _compute_window_regime_table(windows: list[WindowInfo]) -> pd.DataFrame:
    rows = []
    wr = pd.read_csv(WINDOW_RESULTS_PATH).set_index("window")
    for win in windows:
        gate_dir = _gate_paths(win.run_dir)
        selection = json.loads((win.run_dir / "llm" / "online_memory_gate_selection_TSM_LLM-COT-RF-HDELTA.json").read_text())
        val_labels = pd.read_csv(gate_dir / "val_learned_gate_labels.csv")
        test_labels = pd.read_csv(gate_dir / "test_learned_gate_labels.csv")
        test_probs = pd.read_csv(gate_dir / "test_learned_gate_probabilities.csv")
        panel = _window_panel_slice(win)
        returns = panel["y_return"].dropna()
        rows.append(
            {
                "window": win.window,
                "path_improvement_pct": float(wr.loc[win.window, "path_improvement_pct"]),
                "h20_improvement_pct": float(wr.loc[win.window, "h20_improvement_pct"]),
                "h30_improvement_pct": float(wr.loc[win.window, "h30_improvement_pct"]),
                "gate_threshold": float(selection.get("threshold", np.nan)),
                "val_positive_rate": float(val_labels["label"].mean()),
                "test_positive_rate": float(test_labels["label"].mean()),
                "positive_rate_shift": float(test_labels["label"].mean() - val_labels["label"].mean()),
                "apply_rate": float(test_probs["apply_llm_selected"].mean()),
                "mean_apply_prob": float(test_probs["apply_probability"].mean()),
                "n_test_samples": int(len(test_probs)),
                "mean_y_vol_20d": float(panel["y_vol_20d"].mean()),
                "mean_abs_return": float(returns.abs().mean()),
                "std_return": float(returns.std()),
                "auction_day_share": float(panel["is_auction_day"].mean()),
                "net_price_change_pct": float(panel["y"].iloc[-1] / panel["y"].iloc[0] - 1.0),
            }
        )
    return pd.DataFrame(rows)


def _compute_feature_shift_table(windows: list[WindowInfo]) -> pd.DataFrame:
    feature_means = {}
    for win in windows:
        gate_dir = _gate_paths(win.run_dir)
        val_feats = pd.read_csv(gate_dir / "val_learned_gate_features.csv")
        feature_means[win.window] = val_feats[CORE_FEATURES].mean()
    feature_df = pd.DataFrame(feature_means)
    w0 = feature_df.iloc[:, 0]
    others_mean = feature_df.iloc[:, 1:].mean(axis=1)
    out = pd.DataFrame(
        {
            "feature": feature_df.index,
            "W0_mean": w0.values,
            "other_windows_mean": others_mean.values,
            "difference_W0_minus_others": (w0 - others_mean).values,
            "abs_difference": (w0 - others_mean).abs().values,
        }
    )
    return out.sort_values("abs_difference", ascending=False).reset_index(drop=True)


def _write_report(out_dir: Path, regime_df: pd.DataFrame, feature_df: pd.DataFrame) -> None:
    losers = regime_df[regime_df["path_improvement_pct"] < 0].copy()
    w0 = regime_df.iloc[0]
    lines = [
        "# Repeated Holdout Regime Diagnostic",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Main Finding",
        f"- `W0` is not just a better result; it is a different operating regime for the live stack.",
        f"- `W0` path improvement was `{w0['path_improvement_pct']:.2%}`, while the mean of the four earlier windows was `{regime_df.iloc[1:]['path_improvement_pct'].mean():.2%}`.",
        "",
        "## Why W0 Works Better",
        f"- Lower volatility regime: `W0 mean y_vol_20d = {w0['mean_y_vol_20d']:.3f}` vs earlier-window mean `{regime_df.iloc[1:]['mean_y_vol_20d'].mean():.3f}`.",
        f"- Lower return noise: `W0 mean abs return = {w0['mean_abs_return']:.4f}` vs earlier-window mean `{regime_df.iloc[1:]['mean_abs_return'].mean():.4f}`.",
        f"- Much higher gate conviction: `W0 mean apply probability = {w0['mean_apply_prob']:.3f}` and apply rate `{w0['apply_rate']:.2%}`.",
        f"- Earlier windows are mostly lower-confidence: their mean apply rate was `{regime_df.iloc[1:]['apply_rate'].mean():.2%}`.",
        "",
        "## Important Diagnostic",
        "- The biggest structural issue is validation-to-test helpfulness shift in the gate labels.",
        f"- `W0` was relatively aligned: validation positive rate `{w0['val_positive_rate']:.2%}`, test positive rate `{w0['test_positive_rate']:.2%}`.",
        f"- Some failing windows had large shifts, for example `W3` validation positive rate `{regime_df.loc[regime_df['window'].str.startswith('W3'),'val_positive_rate'].iloc[0]:.2%}` vs test `{regime_df.loc[regime_df['window'].str.startswith('W3'),'test_positive_rate'].iloc[0]:.2%}`.",
        "",
        "## Regime Table",
        "",
        regime_df.to_markdown(index=False),
        "",
        "## Top Gate-Feature Shifts (W0 vs average of W1-W4, using validation features)",
        "",
        feature_df.head(12).to_markdown(index=False),
        "",
        "## Figures",
        f"![Window price shapes]({out_dir / 'window_price_shapes.png'})",
        "",
        f"![Window volatility shapes]({out_dir / 'window_volatility_shapes.png'})",
        "",
        f"![Window overlay]({out_dir / 'window_overlay_shapes.png'})",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    out_dir = (
        PROJECT_ROOT
        / "reports"
        / "uk_ets_repeated_holdout_regime_diagnostic"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    windows = _load_windows()
    regime_df = _compute_window_regime_table(windows)
    feature_df = _compute_feature_shift_table(windows)
    regime_df.to_csv(out_dir / "regime_summary.csv", index=False)
    feature_df.to_csv(out_dir / "feature_shift_summary.csv", index=False)
    _plot_window_shapes(windows, out_dir)
    _plot_window_volatility(windows, out_dir)
    _plot_overlay(windows, out_dir)
    _write_report(out_dir, regime_df, feature_df)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
