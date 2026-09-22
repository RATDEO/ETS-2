#!/usr/bin/env python3
"""Build a side-by-side TSM vs 4B vs 27B comparison for W0-W4."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


WINDOWS = [
    {
        "window": "W0",
        "run_4b": "20260331_105619_b6bb37",
        "run_27b": "20260331_181339_b40355",
        "origin_start": "2025-07-01",
        "origin_end": "2026-01-22",
    },
    {
        "window": "W1",
        "run_4b": "20260326_115622_da6c51",
        "run_27b": "20260331_192217_2f8fbb",
        "origin_start": "2024-10-28",
        "origin_end": "2025-05-20",
    },
    {
        "window": "W2",
        "run_4b": "20260326_124702_c7ea2d",
        "run_27b": "20260331_200333_fac9f0",
        "origin_start": "2024-02-23",
        "origin_end": "2024-09-16",
    },
    {
        "window": "W3",
        "run_4b": "20260326_133808_f2b915",
        "run_27b": "20260331_205229_5c7de8",
        "origin_start": "2023-06-21",
        "origin_end": "2024-01-12",
    },
    {
        "window": "W4",
        "run_4b": "20260326_135728_a024ac",
        "run_27b": "20260331_214230_22bef9",
        "origin_start": "2022-10-17",
        "origin_end": "2023-05-10",
    },
]


def _read_metrics(run_id: str) -> tuple[dict[str, float], dict[int, dict[str, float]], dict[int, dict[str, float]]]:
    run_dir = PROJECT_ROOT / "runs" / run_id
    path_df = pd.read_csv(run_dir / "results" / "path_metrics.csv")
    horizon_df = pd.read_csv(run_dir / "results" / "metrics_by_horizon.csv")
    sharpe_df = pd.read_csv(run_dir / "results" / "long_short_metrics.csv")

    path_lookup = {
        str(row.model): float(row.mse_path)
        for row in path_df.itertuples(index=False)
    }
    horizon_lookup: dict[int, dict[str, float]] = {}
    for row in horizon_df.itertuples(index=False):
        horizon_lookup.setdefault(int(row.horizon), {})[str(row.model)] = float(row.mse)
    sharpe_lookup: dict[int, dict[str, float]] = {}
    for row in sharpe_df.itertuples(index=False):
        sharpe_lookup.setdefault(int(row.horizon), {})[str(row.model)] = float(row.sharpe_non_overlap_offset_avg)
    return path_lookup, horizon_lookup, sharpe_lookup


def _improvement_pct(base: float, candidate: float) -> float:
    if abs(base) < 1e-12:
        return 0.0
    return ((base - candidate) / base) * 100.0


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_dir = PROJECT_ROOT / "reports" / "uk_ets_tsm_4b_27b_w0_w4" / stamp
    report_dir.mkdir(parents=True, exist_ok=False)

    rows: list[dict[str, object]] = []
    for spec in WINDOWS:
        path_4b, horizon_4b, sharpe_4b = _read_metrics(spec["run_4b"])
        path_27b, horizon_27b, sharpe_27b = _read_metrics(spec["run_27b"])

        base_path = float(path_4b["tsm"])
        llm_4b_path = float(path_4b["TSM+LLM-COT-RF-HDELTA"])
        llm_27b_path = float(path_27b["TSM+LLM-COT-RF-HDELTA"])

        row: dict[str, object] = {
            "window": spec["window"],
            "origin_start": spec["origin_start"],
            "origin_end": spec["origin_end"],
            "run_4b": spec["run_4b"],
            "run_27b": spec["run_27b"],
            "tsm_path_mse": base_path,
            "llm_4b_path_mse": llm_4b_path,
            "llm_27b_path_mse": llm_27b_path,
            "llm_4b_path_improvement_pct": _improvement_pct(base_path, llm_4b_path),
            "llm_27b_path_improvement_pct": _improvement_pct(base_path, llm_27b_path),
        }

        for horizon in (20, 30):
            base_h = float(horizon_4b[horizon]["tsm"])
            llm_4b_h = float(horizon_4b[horizon]["TSM+LLM-COT-RF-HDELTA"])
            llm_27b_h = float(horizon_27b[horizon]["TSM+LLM-COT-RF-HDELTA"])
            row[f"tsm_h{horizon}_mse"] = base_h
            row[f"llm_4b_h{horizon}_mse"] = llm_4b_h
            row[f"llm_27b_h{horizon}_mse"] = llm_27b_h
            row[f"llm_4b_h{horizon}_improvement_pct"] = _improvement_pct(base_h, llm_4b_h)
            row[f"llm_27b_h{horizon}_improvement_pct"] = _improvement_pct(base_h, llm_27b_h)
            row[f"tsm_h{horizon}_sharpe"] = float(sharpe_4b[horizon]["tsm"])
            row[f"llm_4b_h{horizon}_sharpe"] = float(sharpe_4b[horizon]["TSM+LLM-COT-RF-HDELTA"])
            row[f"llm_27b_h{horizon}_sharpe"] = float(sharpe_27b[horizon]["TSM+LLM-COT-RF-HDELTA"])

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("window").reset_index(drop=True)
    means = {
        "mean_llm_4b_path_improvement_pct": float(df["llm_4b_path_improvement_pct"].mean()),
        "mean_llm_27b_path_improvement_pct": float(df["llm_27b_path_improvement_pct"].mean()),
        "mean_llm_4b_h20_improvement_pct": float(df["llm_4b_h20_improvement_pct"].mean()),
        "mean_llm_27b_h20_improvement_pct": float(df["llm_27b_h20_improvement_pct"].mean()),
        "mean_llm_4b_h30_improvement_pct": float(df["llm_4b_h30_improvement_pct"].mean()),
        "mean_llm_27b_h30_improvement_pct": float(df["llm_27b_h30_improvement_pct"].mean()),
    }

    df.to_csv(report_dir / "comparison.csv", index=False)
    (report_dir / "comparison.json").write_text(
        json.dumps({"rows": df.to_dict(orient="records"), "means": means}, indent=2),
        encoding="utf-8",
    )

    summary_lines = [
        "# TSM vs 4B vs 27B on W0-W4",
        "",
        df.to_markdown(index=False),
        "",
        "## Means",
        "",
    ]
    for key, value in means.items():
        summary_lines.append(f"- `{key}`: `{value:.4f}`")
    (report_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Report dir: {report_dir}")
    print(df.to_string(index=False))
    print(json.dumps(means, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
