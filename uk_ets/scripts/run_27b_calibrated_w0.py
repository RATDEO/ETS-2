#!/usr/bin/env python3
"""Run the calibrated 27B W0 candidate and summarize raw vs blended outputs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.run_experiment import run_experiment


CONFIG_PATH = "uk_ets/config/uk_ets_llm_27b_current_calibrated_w0.yaml"
DATA_DIR = "uk_ets/Data_auto_uk"
METHOD_NAME = "TSM+LLM-COT-RF-HDELTA"
BLEND_MODEL_NAME = f"{METHOD_NAME}_blend_ramp_bestval_path"


def _extract_metrics(run_dir: Path, model_name: str) -> dict[str, float]:
    path_df = pd.read_csv(run_dir / "results" / "path_metrics.csv").set_index("model")
    horizon_df = pd.read_csv(run_dir / "results" / "metrics_by_horizon.csv").set_index(["model", "horizon"])
    long_short_df = pd.read_csv(run_dir / "results" / "long_short_metrics.csv").set_index(["model", "horizon"])

    return {
        "path_mse": float(path_df.loc[model_name, "mse_path"]),
        "h20_mse": float(horizon_df.loc[(model_name, 20), "mse"]),
        "h30_mse": float(horizon_df.loc[(model_name, 30), "mse"]),
        "h20_sharpe": float(long_short_df.loc[(model_name, 20), "sharpe_non_overlap_offset_avg"]),
        "h30_sharpe": float(long_short_df.loc[(model_name, 30), "sharpe_non_overlap_offset_avg"]),
    }


def main() -> int:
    run_dir = Path(
        run_experiment(
            config_path=CONFIG_PATH,
            data_dir=DATA_DIR,
        )
    ).resolve()

    base = _extract_metrics(run_dir, "tsm")
    raw = _extract_metrics(run_dir, METHOD_NAME)
    blended = _extract_metrics(run_dir, BLEND_MODEL_NAME)

    selection_path = run_dir / "llm" / "blend_grid_selection_TSM_LLM-COT-RF-HDELTA.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.exists() else {}

    report = {
        "generated_at": datetime.now().isoformat(),
        "config_path": CONFIG_PATH,
        "run_dir": str(run_dir),
        "selection": selection,
        "base": base,
        "raw_27b": raw,
        "blended_27b": blended,
        "improvements_pct": {
            "raw_path": (base["path_mse"] - raw["path_mse"]) / base["path_mse"] * 100.0,
            "blended_path": (base["path_mse"] - blended["path_mse"]) / base["path_mse"] * 100.0,
            "raw_h20": (base["h20_mse"] - raw["h20_mse"]) / base["h20_mse"] * 100.0,
            "blended_h20": (base["h20_mse"] - blended["h20_mse"]) / base["h20_mse"] * 100.0,
            "raw_h30": (base["h30_mse"] - raw["h30_mse"]) / base["h30_mse"] * 100.0,
            "blended_h30": (base["h30_mse"] - blended["h30_mse"]) / base["h30_mse"] * 100.0,
        },
    }

    out_path = run_dir / "results" / "calibrated_27b_w0_summary.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
