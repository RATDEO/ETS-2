#!/usr/bin/env python3
"""Benchmark TSM-only candidates on repeated-holdout difficult windows W1-W4."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.run_experiment import run_experiment


DATA_DIR = "uk_ets/Data_auto_uk"


@dataclass(frozen=True)
class WindowSpec:
    name: str
    train_end: str
    val_end: str
    test_end: str


WINDOWS = [
    WindowSpec("W1", "2023-10-27", "2024-10-26", "2025-06-30"),
    WindowSpec("W2", "2023-02-22", "2024-02-22", "2024-10-26"),
    WindowSpec("W3", "2022-06-20", "2023-06-20", "2024-02-22"),
    WindowSpec("W4", "2021-10-16", "2022-10-16", "2023-06-20"),
]


def _load_cfg(path: str) -> dict[str, Any]:
    with (PROJECT_ROOT / path).open() as f:
        return yaml.safe_load(f)


LIVE_CFG = _load_cfg("uk_ets/config/uk_ets_llm_4b_tsm_live_gbdt_top20.yaml")
TUNED_CFG = _load_cfg("uk_ets/config/uk_ets_tuned.yaml")


def _candidate_overrides() -> dict[str, dict[str, Any]]:
    tuned_model = TUNED_CFG["model"]
    tuned_features = TUNED_CFG["features"]
    live_features = LIVE_CFG["features"]
    return {
        "live_auto_uk": {
            "llm": {"methods": []},
        },
        "tuned_auto_uk": {
            "llm": {"methods": []},
            "model": tuned_model,
            "features": tuned_features,
        },
        "hybrid_tuned_model_live_features": {
            "llm": {"methods": []},
            "model": tuned_model,
            "features": live_features,
        },
    }


def _merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _run_candidate(window: WindowSpec, candidate_name: str, base_overrides: dict[str, Any]) -> Path:
    overrides = _merge(
        base_overrides,
        {
            "split": {
                "train_end": window.train_end,
                "val_end": window.val_end,
                "test_end": window.test_end,
            }
        },
    )
    return Path(
        run_experiment(
            config_path="uk_ets/config/uk_ets_llm_4b_tsm_live_gbdt_top20.yaml",
            overrides=overrides,
            data_dir=DATA_DIR,
        )
    ).resolve()


def _extract_metrics(run_dir: Path) -> dict[str, float]:
    path_df = pd.read_csv(run_dir / "results" / "path_metrics.csv")
    horizon_df = pd.read_csv(run_dir / "results" / "metrics_by_horizon.csv")
    tsm_row = path_df[path_df["model"] == "tsm"].iloc[0]
    h_rows = horizon_df[horizon_df["model"] == "tsm"].set_index("horizon")
    return {
        "path_mse": float(tsm_row["mse_path"]),
        "h1": float(h_rows.loc[1, "mse"]),
        "h5": float(h_rows.loc[5, "mse"]),
        "h20": float(h_rows.loc[20, "mse"]),
        "h30": float(h_rows.loc[30, "mse"]),
    }


def _write_summary(out_dir: Path, df: pd.DataFrame) -> None:
    lines = [
        "# W1-W4 TSM Candidate Benchmark",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        f"Data dir forced to `{DATA_DIR}` to use the UK-specific weather and auction proxy packs.",
        "",
        "## Results",
        "",
        df.to_markdown(index=False),
        "",
        "## Mean By Candidate",
        "",
        df.groupby("candidate")[["path_mse", "h1", "h5", "h20", "h30"]].mean().reset_index().to_markdown(index=False),
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    out_dir = (
        PROJECT_ROOT
        / "reports"
        / "uk_ets_tsm_w1_w4_candidate_benchmark"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for candidate_name, overrides in _candidate_overrides().items():
        for window in WINDOWS:
            run_dir = _run_candidate(window, candidate_name, overrides)
            metrics = _extract_metrics(run_dir)
            rows.append(
                {
                    "candidate": candidate_name,
                    "window": window.name,
                    "train_end": window.train_end,
                    "val_end": window.val_end,
                    "test_end": window.test_end,
                    "run_dir": str(run_dir),
                    **metrics,
                }
            )
            pd.DataFrame(rows).to_csv(out_dir / "results.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "results.csv", index=False)
    _write_summary(out_dir, df)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
