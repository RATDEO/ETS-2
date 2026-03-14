#!/usr/bin/env python3
"""Targeted UK ETS DLinear search following the pivot report recommendations."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tune_post_2023_tsm as base
from tune_post_2023_tsm import Candidate

from src.config import load_config
from src.utils import set_seed


UK_CANDIDATES: list[Candidate] = [
    Candidate(
        name="dlinear_targetonly_s20_l10_k25_ind",
        tsm_type="dlinear",
        target_mode="returns",
        seq_len=20,
        label_len=10,
        batch_size=64,
        learning_rate=1e-3,
        max_epochs=40,
        patience=8,
        kernel_size=25,
        dlinear_individual=True,
        weight_decay=1e-2,
        dlinear_channel_mixer="target_only",
    ),
    Candidate(
        name="dlinear_resmix_s20_l10_k25_ind",
        tsm_type="dlinear",
        target_mode="returns",
        seq_len=20,
        label_len=10,
        batch_size=64,
        learning_rate=1e-3,
        max_epochs=40,
        patience=8,
        kernel_size=25,
        dlinear_individual=True,
        weight_decay=1e-2,
        dlinear_channel_mixer="residual_linear",
    ),
    Candidate(
        name="dlinear_resmix_s20_l10_k5_shared",
        tsm_type="dlinear",
        target_mode="returns",
        seq_len=20,
        label_len=10,
        batch_size=64,
        learning_rate=1e-3,
        max_epochs=40,
        patience=8,
        kernel_size=5,
        dlinear_individual=False,
        weight_decay=1e-2,
        dlinear_channel_mixer="residual_linear",
    ),
    Candidate(
        name="dlinear_resmix_s15_l5_k5_shared",
        tsm_type="dlinear",
        target_mode="returns",
        seq_len=15,
        label_len=5,
        batch_size=64,
        learning_rate=1e-3,
        max_epochs=40,
        patience=8,
        kernel_size=5,
        dlinear_individual=False,
        weight_decay=1e-2,
        dlinear_channel_mixer="residual_linear",
    ),
    Candidate(
        name="dlinear_resmix_s10_l5_k5_shared",
        tsm_type="dlinear",
        target_mode="returns",
        seq_len=10,
        label_len=5,
        batch_size=64,
        learning_rate=1e-3,
        max_epochs=40,
        patience=8,
        kernel_size=5,
        dlinear_individual=False,
        weight_decay=1e-2,
        dlinear_channel_mixer="residual_linear",
    ),
    Candidate(
        name="dlinear_resmix_s10_l5_k3_shared_lr5e4",
        tsm_type="dlinear",
        target_mode="returns",
        seq_len=10,
        label_len=5,
        batch_size=64,
        learning_rate=5e-4,
        max_epochs=40,
        patience=5,
        kernel_size=3,
        dlinear_individual=False,
        weight_decay=5e-2,
        dlinear_channel_mixer="residual_linear",
    ),
    Candidate(
        name="dlinear_resmix_s20_l10_k5_shared_lr1e4_wd01",
        tsm_type="dlinear",
        target_mode="returns",
        seq_len=20,
        label_len=10,
        batch_size=64,
        learning_rate=1e-4,
        max_epochs=40,
        patience=5,
        kernel_size=5,
        dlinear_individual=False,
        weight_decay=1e-1,
        dlinear_channel_mixer="residual_linear",
    ),
    Candidate(
        name="dlinear_resmix_s20_l10_k9_shared_lr5e4",
        tsm_type="dlinear",
        target_mode="returns",
        seq_len=20,
        label_len=10,
        batch_size=64,
        learning_rate=5e-4,
        max_epochs=40,
        patience=5,
        kernel_size=9,
        dlinear_individual=False,
        weight_decay=5e-2,
        dlinear_channel_mixer="residual_linear",
    ),
    Candidate(
        name="dlinear_resmix_resid_s20_l10_k25_ind_a01",
        tsm_type="dlinear",
        target_mode="returns",
        seq_len=20,
        label_len=10,
        batch_size=64,
        learning_rate=1e-3,
        max_epochs=40,
        patience=8,
        kernel_size=25,
        dlinear_individual=True,
        weight_decay=1e-2,
        dlinear_channel_mixer="residual_linear",
        use_residual_wrapper=True,
        residual_alpha=0.1,
    ),
    Candidate(
        name="dlinear_resmix_resid_s20_l10_k5_shared_a01",
        tsm_type="dlinear",
        target_mode="returns",
        seq_len=20,
        label_len=10,
        batch_size=64,
        learning_rate=1e-3,
        max_epochs=40,
        patience=8,
        kernel_size=5,
        dlinear_individual=False,
        weight_decay=1e-2,
        dlinear_channel_mixer="residual_linear",
        use_residual_wrapper=True,
        residual_alpha=0.1,
    ),
    Candidate(
        name="dlinear_targetonly_s15_l5_k5_shared",
        tsm_type="dlinear",
        target_mode="returns",
        seq_len=15,
        label_len=5,
        batch_size=64,
        learning_rate=1e-3,
        max_epochs=40,
        patience=8,
        kernel_size=5,
        dlinear_individual=False,
        weight_decay=1e-2,
        dlinear_channel_mixer="target_only",
    ),
    Candidate(
        name="dlinear_targetonly_s10_l5_k3_shared_lr5e4",
        tsm_type="dlinear",
        target_mode="returns",
        seq_len=10,
        label_len=5,
        batch_size=64,
        learning_rate=5e-4,
        max_epochs=40,
        patience=5,
        kernel_size=3,
        dlinear_individual=False,
        weight_decay=5e-2,
        dlinear_channel_mixer="target_only",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Targeted UK ETS DLinear search.")
    parser.add_argument("--config", type=Path, default=Path("uk_ets/config/uk_ets_tuned.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("uk_ets/Data_auto_uk"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(int(args.seed))
    config = load_config(args.config)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = ROOT / "reports" / "uk_ets_dlinear_tune" / datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_bundle = base._build_loaders(config.raw, args.data_dir, UK_CANDIDATES[0])
    with (output_dir / "metadata.json").open("w") as fh:
        json.dump(
            {
                "config": str(args.config),
                "data_dir": str(args.data_dir),
                "seed": int(args.seed),
                "n_candidates": len(UK_CANDIDATES),
                "panel_rows": int(len(baseline_bundle["panel"])),
                "split_sizes": {
                    split: int(len(baseline_bundle["splits"][split]["dates"]))
                    for split in ("train", "val", "test")
                },
                "note": "DLinear ignores dropout/d_model/n_heads/e_layers/d_ff, so this search varies effective DLinear/training knobs plus the target-vs-exogenous channel mixer.",
            },
            fh,
            indent=2,
        )
    base_rows = base._baseline_rows(baseline_bundle)
    with (output_dir / "baseline_reference.csv").open("w") as fh:
        fh.write("name,test_path_mse\n")
        for row in base_rows:
            fh.write(f"{row['name']},{row['test_path_mse']}\n")

    rows: list[dict] = []
    for candidate in UK_CANDIDATES:
        set_seed(int(args.seed))
        bundle = base._build_loaders(config.raw, args.data_dir, candidate)
        row = base._fit_and_score(candidate, bundle)
        rows.append(row)
        base.pd.DataFrame(rows).sort_values(["val_path_mse", "test_path_mse"]).to_csv(
            output_dir / "tsm_search_results.partial.csv",
            index=False,
        )

    out = base.pd.DataFrame(rows).sort_values(["val_path_mse", "test_path_mse"]).reset_index(drop=True)
    out.to_csv(output_dir / "tsm_search_results.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
