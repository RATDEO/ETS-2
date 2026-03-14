#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.frontend_live_bundle import (
    DEFAULT_LOCAL_LLM_API_KEY,
    DEFAULT_LOCAL_LLM_BASE_URL,
    generate_frontend_live_bundle,
)
from src.regime_gate_bundle import generate_regime_gate_bundle
from src.utils import set_seed, setup_logging


def _resolve_api_key(cli_key: str | None) -> str | None:
    if cli_key:
        return cli_key
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        return env_key
    return DEFAULT_LOCAL_LLM_API_KEY


def _resolve_base_url(cli_url: str | None) -> str | None:
    if cli_url:
        return cli_url
    env_url = os.environ.get("OPENAI_BASE_URL")
    if env_url:
        return env_url
    return DEFAULT_LOCAL_LLM_BASE_URL


def main() -> int:
    parser = argparse.ArgumentParser(description="Build contamination-safe frontend live/archive bundle.")
    parser.add_argument("--config", default="src/config/default.yaml")
    parser.add_argument("--data-dir", default="Data_auto")
    parser.add_argument(
        "--strategy",
        choices=["benchmark_gate", "legacy_cot_rf"],
        default="benchmark_gate",
        help="Production default is the validated sentiment-alignment benchmark gate.",
    )
    parser.add_argument(
        "--target-mode",
        choices=["price", "returns"],
        default="returns",
        help="Train on returns or prices. Returns is the clean production default.",
    )
    parser.add_argument("--history-years", type=int, default=5)
    parser.add_argument(
        "--origin-step",
        type=int,
        default=63,
        help="Archive origin spacing in trading rows. 63 is roughly quarterly.",
    )
    parser.add_argument("--actual-history-days", type=int, default=365)
    parser.add_argument("--val-windows", type=int, default=252)
    parser.add_argument("--min-train-windows", type=int, default=756)
    parser.add_argument("--min-val-windows", type=int, default=126)
    parser.add_argument("--max-origins", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument("--publish-dir", default=None)
    parser.add_argument("--llm-model", default=None, help="Legacy alias for --llm4b-model.")
    parser.add_argument("--llm-base-url", default=None, help="Legacy alias for --llm4b-base-url.")
    parser.add_argument("--llm4b-model", default="qwen3-vl-4b-gpu")
    parser.add_argument("--llm4b-base-url", default=None)
    parser.add_argument("--llm35b-model", default="qwen3.5-35b-a3b-ud-q4-k-xl")
    parser.add_argument("--llm35b-base-url", default="http://192.168.1.140:9881/v1")
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(int(args.seed))

    overrides = {
        "target": {"mode": args.target_mode},
        "frontend_live_bundle": {
            "history_years": int(args.history_years),
            "origin_step": int(args.origin_step),
            "actual_history_days": int(args.actual_history_days),
            "val_windows": int(args.val_windows),
            "min_train_windows": int(args.min_train_windows),
            "min_val_windows": int(args.min_val_windows),
        },
        "output": {
            "generate_paper": False,
            "write_project_paper": False,
        },
    }
    config = load_config(args.config, overrides=overrides)
    run_dir = config.setup_run_dir(PROJECT_ROOT / config.output.get("runs_dir", "runs"))
    setup_logging(log_dir=run_dir, level=20, log_file=True)
    config.save()
    config.save_reproducibility_info()

    llm_api_key = _resolve_api_key(args.llm_api_key)
    llm4b_model = args.llm_model or args.llm4b_model
    llm4b_base_url = _resolve_base_url(args.llm4b_base_url or args.llm_base_url)
    publish_dir = Path(args.publish_dir).resolve() if args.publish_dir else None

    if args.strategy == "benchmark_gate":
        if args.skip_llm:
            raise SystemExit("--skip-llm is not supported for --strategy benchmark_gate.")
        result = generate_regime_gate_bundle(
            config_raw=config.raw,
            data_dir=Path(args.data_dir).resolve(),
            run_dir=run_dir,
            publish_dir=publish_dir,
            llm_api_key=llm_api_key,
            llm4b_model=llm4b_model,
            llm4b_base_url=llm4b_base_url,
            llm35b_model=args.llm35b_model,
            llm35b_base_url=args.llm35b_base_url,
            max_origins=args.max_origins,
        )
    else:
        result = generate_frontend_live_bundle(
            config_raw=config.raw,
            data_dir=Path(args.data_dir).resolve(),
            run_dir=run_dir,
            publish_dir=publish_dir,
            llm_enabled=not args.skip_llm,
            llm_model=llm4b_model,
            llm_api_key=llm_api_key,
            llm_base_url=llm4b_base_url,
            max_origins=args.max_origins,
        )

    summary_models = result.get("archive_summary", {}).get("models", [])
    print("\n=== Frontend Live Bundle Complete ===")
    print(f"Run directory: {run_dir}")
    print(f"Bundle directory: {result['output_dir']}")
    print(f"Strategy: {args.strategy}")
    for row in summary_models:
        print(
            f"- {row['display_name']} ({row['model']}): "
            f"path_mse={row['path_mse']:.6f}, n_origins={row['n_origins']}"
        )
    print(
        f"Daily ticker as of {result['current_price']['as_of_date']}: "
        f"{result['current_price']['price']:.4f}"
    )
    if args.publish_dir:
        print(f"Published bundle: {Path(args.publish_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
