#!/usr/bin/env python3
"""Run the prompt-tuned 27B W0 ablation."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.run_experiment import run_experiment


CONFIG_PATH = "uk_ets/config/uk_ets_llm_27b_prompt_tuned_w0.yaml"
DATA_DIR = "uk_ets/Data_auto_uk"


def main() -> int:
    run_dir = run_experiment(
        config_path=CONFIG_PATH,
        data_dir=DATA_DIR,
    )
    print(Path(run_dir).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
