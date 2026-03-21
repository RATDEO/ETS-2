#!/usr/bin/env python3
"""Benchmark legacy /Data source groups on W0-W4 for the current UK ETS stack."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.run_experiment import run_experiment
from uk_ets.scripts.benchmark_effective_retrieval_w1_w4_huber_energy import (
    _baseline_policy_overrides as wide_policy_overrides,
)


CONFIG_PATH = "uk_ets/config/uk_ets_llm_4b_current_default.yaml"
CURRENT_DATA_DIR = PROJECT_ROOT / "uk_ets" / "Data_auto_uk"
LEGACY_DATA_DIR = PROJECT_ROOT / "Data"
MERGED_ROOT = PROJECT_ROOT / "tmp" / "legacy_source_merge"


@dataclass(frozen=True)
class WindowSpec:
    name: str
    train_end: str
    val_end: str
    test_end: str


@dataclass(frozen=True)
class Candidate:
    name: str
    objective: str
    expectation: str
    data_variant: str
    overrides: dict[str, Any]


WINDOWS = [
    WindowSpec("W0", "2024-06-30", "2025-06-30", "2026-03-04"),
    WindowSpec("W1", "2023-10-27", "2024-10-26", "2025-06-30"),
    WindowSpec("W2", "2023-02-22", "2024-02-22", "2024-10-26"),
    WindowSpec("W3", "2022-06-20", "2023-06-20", "2024-02-22"),
    WindowSpec("W4", "2021-10-16", "2022-10-16", "2023-06-20"),
]


BASE_ENERGY16 = [
    "target_range_pct",
    "target_volume",
    "y_vol_20d",
    "y_ma_5d",
    "y_momentum_20d",
    "is_auction_day",
    "uk_icap_secondary_print_day",
    "coal_brent_ratio",
    "coal_brent_ratio_z20",
    "uk_power_gas_vol_ratio_20d",
    "uk_gas_hdd18_surprise_interaction",
    "uk_hdd18_7d_ma",
    "uka_brent_ratio",
    "uka_brent_ratio_z20",
    "uk_gas_vol_20d",
    "uk_temp_mean_c",
]

INDICES5 = ["idx_keua", "idx_krbn", "idx_grn", "idx_kcca", "idx_kset"]
VSTOXX3 = ["vstoxx", "vstoxx_ma_20d", "vstoxx_high_vol"]
ICAP3 = ["uk_icap_secondary", "uk_icap_primary", "uk_icap_primary_secondary_spread_pct"]
AUCTION2 = ["auction_volume", "auction_price_lag_20"]

LEGACY_COMPACT20 = [
    "target_range_pct",
    "target_volume",
    "y_vol_20d",
    "y_ma_5d",
    "y_momentum_20d",
    "is_auction_day",
    "uk_icap_secondary_print_day",
    "uk_icap_secondary",
    "uk_icap_primary_secondary_spread_pct",
    "auction_volume",
    "coal_brent_ratio",
    "coal_brent_ratio_z20",
    "uk_power_gas_vol_ratio_20d",
    "uk_gas_hdd18_surprise_interaction",
    "uk_hdd18_7d_ma",
    "uka_brent_ratio",
    "uk_gas_vol_20d",
    "uk_temp_mean_c",
    "idx_keua",
    "vstoxx",
]


def _merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _prepare_merged_dirs() -> dict[str, Path]:
    MERGED_ROOT.mkdir(parents=True, exist_ok=True)
    dirs: dict[str, Path] = {}
    variants = {
        "current": {},
        "legacy_indices": {"carbon-market-indices": LEGACY_DATA_DIR / "carbon-market-indices"},
        "legacy_auctions": {
            "emission-spot-primary-market-auction": LEGACY_DATA_DIR / "emission-spot-primary-market-auction"
        },
        "legacy_all": {
            "carbon-market-indices": LEGACY_DATA_DIR / "carbon-market-indices",
            "emission-spot-primary-market-auction": LEGACY_DATA_DIR / "emission-spot-primary-market-auction",
            "volatility-proxy": LEGACY_DATA_DIR / "volatility-proxy",
            "usd.xml": LEGACY_DATA_DIR / "usd.xml",
        },
    }

    for name, overlays in variants.items():
        root = MERGED_ROOT / name
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)

        for child in CURRENT_DATA_DIR.iterdir():
            target = root / child.name
            os.symlink(child, target, target_is_directory=child.is_dir())

        for rel_name, src in overlays.items():
            target = root / rel_name
            if target.exists() or target.is_symlink():
                if target.is_dir() and not target.is_symlink():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            os.symlink(src, target, target_is_directory=src.is_dir())

        dirs[name] = root
    return dirs


def _base_anchor() -> dict[str, Any]:
    return {
        "time_series": {
            "seq_len": 20,
            "label_len": 10,
        },
        "model": {
            "learning_rate": 0.001,
            "max_epochs": 40,
            "early_stopping_patience": 8,
            "dropout": 0.5,
            "weight_decay": 0.02,
            "dlinear_individual": False,
            "loss_type": "huber",
            "loss_huber_beta": 0.5,
        },
        "features": {
            "include_brent": True,
            "include_coal": True,
            "include_uk_gas": True,
            "include_uk_power": True,
            "include_uk_weather": True,
            "weather_feature_pack_path": "weather-proxy/uk_weather_daily_feature_pack.csv",
            "weather_lag_days": 1,
            "include_auctions": True,
            "auction_features": ["price", "is_auction_day"],
            "auction_lags": [1, 5, 20],
            "include_auction_proxy_pack": True,
            "auction_proxy_pack_path": "auction-proxy-features/uk_icap_primary_secondary_feature_pack.csv",
            "auction_proxy_fill_limit": 10,
            "auction_proxy_days_since_cap": 30,
            "include_indices": False,
            "indices_list": [],
            "include_icap_secondary": False,
            "include_vstoxx": False,
            "max_exogenous_features_model": len(BASE_ENERGY16),
            "preferred_feature_order": BASE_ENERGY16,
        },
    }


def _candidate_definitions() -> list[Candidate]:
    base = _base_anchor()
    candidates = [
        Candidate(
            "baseline_energy16",
            "Current improved UK-specific base without legacy-only source groups.",
            "Anchor candidate.",
            "current",
            base,
        ),
        Candidate(
            "legacy_indices5",
            "Add legacy carbon-market indices, including KSET, and expose them to the base model.",
            "Possible small help on W3/W4 if global carbon beta matters.",
            "legacy_indices",
            _merge(
                base,
                {
                    "features": {
                        "include_indices": True,
                        "indices_list": ["KEUA", "KRBN", "GRN", "KCCA", "KSET"],
                        "max_exogenous_features_model": len(BASE_ENERGY16 + INDICES5),
                        "preferred_feature_order": BASE_ENERGY16 + INDICES5,
                    }
                },
            ),
        ),
        Candidate(
            "legacy_vstoxx",
            "Turn VSTOXX back on as a legacy volatility proxy.",
            "Possible small help if hard windows are volatility-regime driven.",
            "legacy_all",
            _merge(
                base,
                {
                    "features": {
                        "include_vstoxx": True,
                        "max_exogenous_features_model": len(BASE_ENERGY16 + VSTOXX3),
                        "preferred_feature_order": BASE_ENERGY16 + VSTOXX3,
                    }
                },
            ),
        ),
        Candidate(
            "legacy_icap_secondary",
            "Expose raw ICAP secondary and spread features back into the base.",
            "Possible help in auction/microstructure-heavy windows.",
            "current",
            _merge(
                base,
                {
                    "features": {
                        "include_icap_secondary": True,
                        "max_exogenous_features_model": len(BASE_ENERGY16 + ICAP3),
                        "preferred_feature_order": BASE_ENERGY16 + ICAP3,
                    }
                },
            ),
        ),
        Candidate(
            "legacy_auction_volume",
            "Overlay the old auction files and reintroduce auction volume/lag features.",
            "Possible help in W4 if early auction regime is better represented by legacy files.",
            "legacy_auctions",
            _merge(
                base,
                {
                    "features": {
                        "auction_features": ["price", "volume", "is_auction_day"],
                        "max_exogenous_features_model": len(BASE_ENERGY16 + AUCTION2),
                        "preferred_feature_order": BASE_ENERGY16 + AUCTION2,
                    }
                },
            ),
        ),
        Candidate(
            "legacy_market_compact20",
            "Compact mixed legacy stack: one index, VSTOXX, richer ICAP, and auction volume on top of the improved base.",
            "Best chance of a robust mixed-source gain without blowing up dimensionality.",
            "legacy_all",
            _merge(
                base,
                {
                    "features": {
                        "include_indices": True,
                        "indices_list": ["KEUA", "KRBN", "GRN", "KCCA", "KSET"],
                        "include_icap_secondary": True,
                        "include_vstoxx": True,
                        "auction_features": ["price", "volume", "is_auction_day"],
                        "max_exogenous_features_model": len(LEGACY_COMPACT20),
                        "preferred_feature_order": LEGACY_COMPACT20,
                    }
                },
            ),
        ),
        Candidate(
            "legacy_market_stack_all",
            "Throw the main legacy market groups in at once with a wider feature budget.",
            "Highest variance candidate. Could help if source omission is the core problem, could also dilute badly.",
            "legacy_all",
            _merge(
                base,
                {
                    "features": {
                        "include_indices": True,
                        "indices_list": ["KEUA", "KRBN", "GRN", "KCCA", "KSET"],
                        "include_icap_secondary": True,
                        "include_vstoxx": True,
                        "auction_features": ["price", "volume", "is_auction_day"],
                        "max_exogenous_features_model": len(BASE_ENERGY16 + INDICES5 + VSTOXX3 + ICAP3 + AUCTION2),
                        "preferred_feature_order": BASE_ENERGY16 + INDICES5 + VSTOXX3 + ICAP3 + AUCTION2,
                    }
                },
            ),
        ),
    ]
    return candidates


def _run_candidate(
    window: WindowSpec,
    candidate: Candidate,
    data_dirs: dict[str, Path],
    llm_enabled: bool,
) -> Path:
    extra = {}
    if llm_enabled:
        extra = wide_policy_overrides()
    else:
        extra = {"llm": {"methods": []}}
    overrides = _merge(
        _merge(candidate.overrides, extra),
        {
            "split": {
                "train_end": window.train_end,
                "val_end": window.val_end,
                "test_end": window.test_end,
            }
        },
    )
    rel_data_dir = str(data_dirs[candidate.data_variant].relative_to(PROJECT_ROOT))
    return Path(run_experiment(config_path=CONFIG_PATH, overrides=overrides, data_dir=rel_data_dir)).resolve()


def _extract_base_metrics(run_dir: Path) -> dict[str, float]:
    path_df = pd.read_csv(run_dir / "results" / "path_metrics.csv")
    horizon_df = pd.read_csv(run_dir / "results" / "metrics_by_horizon.csv")
    row = path_df[path_df["model"] == "tsm"].iloc[0]
    by_h = horizon_df[horizon_df["model"] == "tsm"].set_index("horizon")
    return {
        "path_mse": float(row["mse_path"]),
        "h1": float(by_h.loc[1, "mse"]),
        "h5": float(by_h.loc[5, "mse"]),
        "h20": float(by_h.loc[20, "mse"]),
        "h30": float(by_h.loc[30, "mse"]),
    }


def _extract_live_metrics(run_dir: Path) -> dict[str, float]:
    path_df = pd.read_csv(run_dir / "results" / "path_metrics.csv")
    horizon_df = pd.read_csv(run_dir / "results" / "metrics_by_horizon.csv")
    tsm_row = path_df[path_df["model"] == "tsm_llm_subset"].iloc[0]
    llm_row = path_df[path_df["model"] == "TSM+LLM-COT-RF-HDELTA"].iloc[0]
    tsm_h = horizon_df[horizon_df["model"] == "tsm_llm_subset"].set_index("horizon")
    llm_h = horizon_df[horizon_df["model"] == "TSM+LLM-COT-RF-HDELTA"].set_index("horizon")
    return {
        "tsm_path_mse": float(tsm_row["mse_path"]),
        "llm_path_mse": float(llm_row["mse_path"]),
        "tsm_h20": float(tsm_h.loc[20, "mse"]),
        "llm_h20": float(llm_h.loc[20, "mse"]),
        "tsm_h30": float(tsm_h.loc[30, "mse"]),
        "llm_h30": float(llm_h.loc[30, "mse"]),
    }


def _write_plan(out_dir: Path, candidates: list[Candidate]) -> None:
    lines = [
        "# W0-W4 Legacy Source Benchmark Plan",
        "",
        "## Objective",
        "- Start from the corrected UK-specific live stack and improved Huber base.",
        "- Reintroduce legacy `/Data` source groups from the older pipeline in controlled bundles.",
        "- Test whether any of those legacy source groups improve current `W0-W4` performance.",
        "",
        "## Method",
        "- Build merged data roots that keep `uk_ets/Data_auto_uk` as the base and overlay legacy directories only where needed.",
        "- Run a broad base-only sweep first.",
        "- Then run the current best live refiner on the best few source bundles.",
        "",
        "## Candidates",
    ]
    for c in candidates:
        lines.append(f"- `{c.name}` ({c.data_variant}): {c.objective}")
        lines.append(f"  Expected outcome: {c.expectation}")
    (out_dir / "plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summarize_base(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("candidate")[["path_mse", "h20", "h30"]]
        .mean()
        .sort_values("path_mse")
        .reset_index()
    )


def _summarize_live(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["path_improvement_pct"] = 1.0 - (df["llm_path_mse"] / df["tsm_path_mse"])
    df["h20_improvement_pct"] = 1.0 - (df["llm_h20"] / df["tsm_h20"])
    df["h30_improvement_pct"] = 1.0 - (df["llm_h30"] / df["tsm_h30"])
    return (
        df.groupby("candidate")[["tsm_path_mse", "llm_path_mse", "path_improvement_pct", "h20_improvement_pct", "h30_improvement_pct"]]
        .mean()
        .sort_values("llm_path_mse")
        .reset_index()
    )


def _write_summary(out_dir: Path, base_df: pd.DataFrame, live_df: pd.DataFrame | None) -> None:
    lines = [
        "# W0-W4 Legacy Source Benchmark",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Base Sweep",
        "",
        base_df.to_markdown(index=False),
        "",
        "## Base Mean By Candidate",
        "",
        _summarize_base(base_df).to_markdown(index=False),
    ]
    if live_df is not None and not live_df.empty:
        lines.extend(
            [
                "",
                "## Live Shortlist",
                "",
                live_df.to_markdown(index=False),
                "",
                "## Live Mean By Candidate",
                "",
                _summarize_live(live_df).to_markdown(index=False),
            ]
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-top-k", type=int, default=3)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "reports" / "uk_ets_legacy_sources_w0_w4" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = _candidate_definitions()
    _write_plan(out_dir, candidates)
    data_dirs = _prepare_merged_dirs()
    (out_dir / "data_variants.json").write_text(
        json.dumps({k: str(v) for k, v in data_dirs.items()}, indent=2) + "\n",
        encoding="utf-8",
    )

    base_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        for window in WINDOWS:
            run_dir = _run_candidate(window, candidate, data_dirs, llm_enabled=False)
            metrics = _extract_base_metrics(run_dir)
            row = {
                "candidate": candidate.name,
                "data_variant": candidate.data_variant,
                "window": window.name,
                "train_end": window.train_end,
                "val_end": window.val_end,
                "test_end": window.test_end,
                "run_dir": str(run_dir),
                **metrics,
            }
            base_rows.append(row)
            pd.DataFrame(base_rows).to_csv(out_dir / "base_results.csv", index=False)

    base_df = pd.DataFrame(base_rows)
    base_mean = _summarize_base(base_df)
    shortlist = base_mean.head(args.live_top_k)["candidate"].tolist()

    live_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate.name not in shortlist:
            continue
        for window in WINDOWS:
            run_dir = _run_candidate(window, candidate, data_dirs, llm_enabled=True)
            metrics = _extract_live_metrics(run_dir)
            row = {
                "candidate": candidate.name,
                "data_variant": candidate.data_variant,
                "window": window.name,
                "train_end": window.train_end,
                "val_end": window.val_end,
                "test_end": window.test_end,
                "run_dir": str(run_dir),
                **metrics,
            }
            live_rows.append(row)
            pd.DataFrame(live_rows).to_csv(out_dir / "live_results.csv", index=False)

    live_df = pd.DataFrame(live_rows) if live_rows else None
    _write_summary(out_dir, base_df, live_df)


if __name__ == "__main__":
    main()
