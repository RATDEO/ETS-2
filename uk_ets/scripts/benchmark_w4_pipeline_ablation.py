#!/usr/bin/env python3
"""Isolate the W4 pipeline jump by ablating weather, auction proxy, and training regime."""

from __future__ import annotations

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


CONFIG_PATH = "uk_ets/config/uk_ets_llm_4b_current_default.yaml"
DATA_DIR = "uk_ets/Data_auto_uk"
W4_SPLIT = {
    "train_end": "2021-10-16",
    "val_end": "2022-10-16",
    "test_end": "2023-06-20",
}

CURRENT16 = [
    "target_range_pct",
    "target_volume",
    "is_auction_day",
    "uk_icap_primary_print_day",
    "uk_icap_secondary_print_day",
    "y_vol_20d",
    "y_ma_5d",
    "y_momentum_5d",
    "y_momentum_20d",
    "uk_power_return",
    "uk_gas_return",
    "brent_return",
    "coal_return",
    "uk_hdd18",
    "uk_hdd18_7d_ma",
    "uk_temp_mean_c",
]

ENERGY16 = [
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


@dataclass(frozen=True)
class Candidate:
    name: str
    objective: str
    overrides: dict[str, Any]


def _merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _feature_cfg(order: list[str], include_weather: bool, include_proxy: bool) -> dict[str, Any]:
    return {
        "include_brent": True,
        "include_coal": True,
        "include_uk_gas": True,
        "include_uk_power": True,
        "include_uk_weather": include_weather,
        "weather_feature_pack_path": "weather-proxy/uk_weather_daily_feature_pack.csv",
        "weather_lag_days": 1,
        "include_indices": False,
        "indices_list": [],
        "include_auctions": True,
        "auction_features": ["price", "is_auction_day"],
        "auction_lags": [1, 5, 20],
        "include_auction_proxy_pack": include_proxy,
        "auction_proxy_pack_path": "auction-proxy-features/uk_icap_primary_secondary_feature_pack.csv",
        "auction_proxy_fill_limit": 10,
        "auction_proxy_days_since_cap": 30,
        "include_icap_secondary": False,
        "include_vstoxx": False,
        "returns": True,
        "rolling_vol_window": 20,
        "rolling_mean_windows": [5, 20, 60],
        "max_exogenous_features_model": len(order),
        "preferred_feature_order": order,
    }


def _old_model() -> dict[str, Any]:
    return {
        "tsm_type": "dlinear",
        "batch_size": 64,
        "learning_rate": 0.003,
        "max_epochs": 80,
        "early_stopping_patience": 12,
        "dropout": 0.3,
        "weight_decay": 0.01,
        "dlinear_individual": False,
        "kernel_size": 3,
        "dlinear_channel_mixer": "residual_linear",
    }


def _new_model() -> dict[str, Any]:
    return {
        "tsm_type": "dlinear",
        "batch_size": 64,
        "learning_rate": 0.001,
        "max_epochs": 40,
        "early_stopping_patience": 8,
        "dropout": 0.5,
        "weight_decay": 0.02,
        "dlinear_individual": False,
        "kernel_size": 3,
        "dlinear_channel_mixer": "residual_linear",
        "loss_type": "huber",
        "loss_huber_beta": 0.5,
    }


def _candidate_defs() -> list[Candidate]:
    old_current16_no_weather = [f for f in CURRENT16 if f not in {"uk_hdd18", "uk_hdd18_7d_ma", "uk_temp_mean_c"}]
    old_current16_no_proxy = [f for f in CURRENT16 if f not in {"uk_icap_primary_print_day", "uk_icap_secondary_print_day"}]
    old_current16_no_both = [
        f
        for f in CURRENT16
        if f
        not in {
            "uk_hdd18",
            "uk_hdd18_7d_ma",
            "uk_temp_mean_c",
            "uk_icap_primary_print_day",
            "uk_icap_secondary_print_day",
        }
    ]

    return [
        Candidate(
            "old_train_fullpacks_current16",
            "Recreate the old training regime on the corrected UK data root with weather and proxy packs enabled.",
            {
                "time_series": {"seq_len": 20, "label_len": 10},
                "model": _old_model(),
                "features": _feature_cfg(CURRENT16, include_weather=True, include_proxy=True),
                "llm": {"methods": []},
            },
        ),
        Candidate(
            "old_train_no_weather_current13",
            "Old training regime on corrected UK data, but with weather disabled.",
            {
                "time_series": {"seq_len": 20, "label_len": 10},
                "model": _old_model(),
                "features": _feature_cfg(old_current16_no_weather, include_weather=False, include_proxy=True),
                "llm": {"methods": []},
            },
        ),
        Candidate(
            "old_train_no_proxy_current14",
            "Old training regime on corrected UK data, but with the UK auction proxy pack disabled.",
            {
                "time_series": {"seq_len": 20, "label_len": 10},
                "model": _old_model(),
                "features": _feature_cfg(old_current16_no_proxy, include_weather=True, include_proxy=False),
                "llm": {"methods": []},
            },
        ),
        Candidate(
            "old_train_no_weather_proxy_current11",
            "Old training regime on corrected UK data, with both weather and proxy pack disabled.",
            {
                "time_series": {"seq_len": 20, "label_len": 10},
                "model": _old_model(),
                "features": _feature_cfg(old_current16_no_both, include_weather=False, include_proxy=False),
                "llm": {"methods": []},
            },
        ),
        Candidate(
            "new_train_no_weather_current13",
            "New regularized Huber training on corrected UK data, with weather disabled.",
            {
                "time_series": {"seq_len": 20, "label_len": 10},
                "model": _new_model(),
                "features": _feature_cfg(old_current16_no_weather, include_weather=False, include_proxy=True),
                "llm": {"methods": []},
            },
        ),
        Candidate(
            "new_train_no_proxy_current14",
            "New regularized Huber training on corrected UK data, with proxy pack disabled.",
            {
                "time_series": {"seq_len": 20, "label_len": 10},
                "model": _new_model(),
                "features": _feature_cfg(old_current16_no_proxy, include_weather=True, include_proxy=False),
                "llm": {"methods": []},
            },
        ),
        Candidate(
            "new_train_no_weather_proxy_current11",
            "New regularized Huber training on corrected UK data, with both weather and proxy pack disabled.",
            {
                "time_series": {"seq_len": 20, "label_len": 10},
                "model": _new_model(),
                "features": _feature_cfg(old_current16_no_both, include_weather=False, include_proxy=False),
                "llm": {"methods": []},
            },
        ),
    ]


def _run(candidate: Candidate) -> Path:
    overrides = _merge(candidate.overrides, {"split": dict(W4_SPLIT)})
    return Path(run_experiment(config_path=CONFIG_PATH, overrides=overrides, data_dir=DATA_DIR)).resolve()


def _extract_metrics(run_dir: Path) -> dict[str, float]:
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


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "reports" / "uk_ets_w4_pipeline_ablation" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "candidate": "old_reference_generic_data",
            "source": "imported",
            "objective": "Original repeated-holdout W4 reference from the old generic Data tree.",
            "run_dir": "/Users/davidwilkinson/Desktop/ETS 2/runs/20260316_225915_798bce",
            "path_mse": 58.9789924621582,
            "h1": 4.714263916015625,
            "h5": 24.53902244567871,
            "h20": 75.92779541015625,
            "h30": 101.1852798461914,
        },
        {
            "candidate": "new_reference_current16",
            "source": "imported",
            "objective": "Corrected UK-data-root W4 with new training and current16 feature slate.",
            "run_dir": "/Users/davidwilkinson/Desktop/ETS 2/runs/20260319_200110_362947",
            "path_mse": 102.83332824707033,
            "h1": 11.390755653381348,
            "h5": 26.95508575439453,
            "h20": 127.08478546142578,
            "h30": 218.5743865966797,
        },
        {
            "candidate": "new_reference_energy16",
            "source": "imported",
            "objective": "Corrected UK-data-root W4 with new training and engineered energy16 feature slate.",
            "run_dir": "/Users/davidwilkinson/Desktop/ETS 2/runs/20260319_200137_e26524",
            "path_mse": 101.73527526855467,
            "h1": 11.40880012512207,
            "h5": 27.140268325805664,
            "h20": 124.65435028076172,
            "h30": 216.0431365966797,
        },
    ]

    candidates = _candidate_defs()
    for candidate in candidates:
        run_dir = _run(candidate)
        rows.append(
            {
                "candidate": candidate.name,
                "source": "rerun",
                "objective": candidate.objective,
                "run_dir": str(run_dir),
                **_extract_metrics(run_dir),
            }
        )
        pd.DataFrame(rows).to_csv(out_dir / "results.csv", index=False)

    df = pd.DataFrame(rows)
    lines = [
        "# W4 Pipeline Ablation",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Goal",
        "- Isolate why the old `W4` number (`~58.98`) is so much lower than the corrected UK-specific `W4` numbers (`~101-103`).",
        "- Hold the split fixed at `2021-10-16 / 2022-10-16 / 2023-06-20`.",
        "- Toggle weather, UK auction proxy pack, and training regime separately on the corrected UK data root.",
        "",
        "## Results",
        "",
        df.to_markdown(index=False),
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
