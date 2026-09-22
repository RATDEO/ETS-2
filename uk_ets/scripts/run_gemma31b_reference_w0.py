#!/usr/bin/env python3
"""Run the Gemma 4 31B W0 reference benchmark."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.run_experiment import run_experiment


CONFIG_PATH = "uk_ets/config/uk_ets_llm_gemma31b_current_default_reference.yaml"
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
