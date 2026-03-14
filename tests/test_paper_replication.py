from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.paper_replication import aggregate_fold_rows, build_outer_folds
from scripts.run_paper_replication_v2 import (
    _candidate_llm_config,
    _prepare_example_pool,
    _resolve_requested_names,
    _select_example_indices,
)
from src.eval.paper_replication import PaperReplicationCandidate


def test_build_outer_folds_respects_cap_and_order() -> None:
    folds = build_outer_folds(10, n_folds=3, max_samples_per_fold=2)
    assert len(folds) == 3
    assert [fold.tolist() for fold in folds] == [[0, 3], [4, 6], [7, 9]]


def test_aggregate_fold_rows_sorts_by_best_mean_path_mse() -> None:
    rows = pd.DataFrame(
        [
            {"candidate": "b", "fold": 1, "mse_path": 2.0, "paper_d10_acc": 0.4},
            {"candidate": "b", "fold": 2, "mse_path": 3.0, "paper_d10_acc": 0.5},
            {"candidate": "a", "fold": 1, "mse_path": 1.0, "paper_d10_acc": 0.6},
            {"candidate": "a", "fold": 2, "mse_path": 1.5, "paper_d10_acc": 0.7},
        ]
    )
    summary = aggregate_fold_rows(rows)
    assert summary["candidate"].tolist() == ["a", "b"]
    assert np.isclose(summary.iloc[0]["mse_path_mean"], 1.25)
    assert np.isclose(summary.iloc[1]["mse_path_mean"], 2.5)


def test_candidate_llm_config_applies_hdelta_overrides() -> None:
    candidate = PaperReplicationCandidate(
        name="paper_hdelta_guarded",
        method="TSM+LLM-COT-SENT-RF-HDELTA",
        sentiment_path="data/news/daily_sentiment.csv",
        hdelta_max_adjustment_pct=1.0,
        hdelta_key_horizons=(1, 10, 20, 30),
        hdelta_freeze_horizons=(1,),
        hdelta_sentiment_secondary=True,
        retain_context=False,
        strict_json_prompt=True,
    )
    cfg = _candidate_llm_config(
        base_llm_cfg={"cot_rf": {}, "sentiment": {}},
        candidate=candidate,
        api_key="deo",
        base_url_override=None,
    )
    assert cfg["methods"] == ["TSM+LLM-COT-SENT-RF-HDELTA"]
    assert cfg["cot_rf"]["retain_context"] is False
    assert cfg["cot_rf"]["strict_json_prompt"] is True
    assert cfg["hdelta"]["key_horizons"] == [1, 10, 20, 30]
    assert cfg["hdelta"]["max_adjustment_pct"] == 1.0
    assert cfg["hdelta"]["freeze_horizons"] == [1]
    assert cfg["hdelta"]["sentiment_secondary"] is True


def test_candidate_llm_config_applies_hprice_overrides() -> None:
    candidate = PaperReplicationCandidate(
        name="paper_hprice_guarded",
        method="TSM+LLM-COT-SENT-RF-HPRICE",
        sentiment_path="data/news/daily_sentiment.csv",
        hprice_max_adjustment_pct=1.0,
        hprice_key_horizons=(1, 10, 20, 30),
        hprice_freeze_horizons=(1,),
        hprice_sentiment_secondary=True,
        retain_context=False,
        strict_json_prompt=True,
    )
    cfg = _candidate_llm_config(
        base_llm_cfg={"cot_rf": {}, "sentiment": {}},
        candidate=candidate,
        api_key="deo",
        base_url_override=None,
    )
    assert cfg["methods"] == ["TSM+LLM-COT-SENT-RF-HPRICE"]
    assert cfg["cot_rf"]["retain_context"] is False
    assert cfg["cot_rf"]["strict_json_prompt"] is True
    assert cfg["hprice"]["key_horizons"] == [1, 10, 20, 30]
    assert cfg["hprice"]["max_adjustment_pct"] == 1.0
    assert cfg["hprice"]["freeze_horizons"] == [1]
    assert cfg["hprice"]["sentiment_secondary"] is True


def test_resolve_requested_names_uses_default_profile() -> None:
    requested, profile = _resolve_requested_names(
        rep_cfg={
            "default_profile": "simplified_4b_hprice",
            "profiles": {"simplified_4b_hprice": ["paper_event90_proxy_hprice_guarded"]},
        },
        profile_name=None,
        candidate_names_arg=None,
    )
    assert requested == {"paper_event90_proxy_hprice_guarded"}
    assert profile == "simplified_4b_hprice"


def test_resolve_requested_names_prefers_explicit_candidate_names() -> None:
    requested, profile = _resolve_requested_names(
        rep_cfg={
            "default_profile": "simplified_4b_hprice",
            "profiles": {"simplified_4b_hprice": ["paper_event90_proxy_hprice_guarded"]},
        },
        profile_name="simplified_4b_hprice",
        candidate_names_arg="paper_baseline,paper_event90_proxy",
    )
    assert requested == {"paper_baseline", "paper_event90_proxy"}
    assert profile is None


def test_resolve_requested_names_uses_named_profile() -> None:
    requested, profile = _resolve_requested_names(
        rep_cfg={
            "default_profile": "simplified_4b_hprice",
            "profiles": {
                "simplified_4b_hprice": ["paper_event90_proxy_hprice_guarded"],
                "compare_fullpath_4b_official_ctx": ["paper_event90_proxy", "paper_event90_proxy_official_ctx"],
            },
        },
        profile_name="compare_fullpath_4b_official_ctx",
        candidate_names_arg=None,
    )
    assert requested == {"paper_event90_proxy", "paper_event90_proxy_official_ctx"}
    assert profile == "compare_fullpath_4b_official_ctx"


def test_prepare_example_pool_event_similarity_includes_retrieval_features() -> None:
    pool_dates = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[D]")
    pool_histories = np.array(
        [
            [70.0, 71.0, 72.0],
            [72.0, 71.0, 70.0],
        ],
        dtype=float,
    )
    pool_forecasts = np.array([[73.0, 74.0], [69.0, 68.0]], dtype=float)
    pool_truth = np.array([[73.5, 74.5], [68.5, 67.5]], dtype=float)
    retrieval_feature_map = {
        pd.Timestamp("2024-01-01"): np.array([1.0, 0.0], dtype=float),
        pd.Timestamp("2024-01-02"): np.array([0.0, 2.0], dtype=float),
    }
    prepared = _prepare_example_pool(
        pool_dates=pool_dates,
        pool_histories=pool_histories,
        pool_forecasts=pool_forecasts,
        pool_truth=pool_truth,
        selection_mode="event_similarity",
        feature_window=3,
        retrieval_feature_map=retrieval_feature_map,
        retrieval_feature_width=2,
    )
    assert prepared["event_similarity_features"].shape == (2, 5)
    assert prepared["retrieval_feature_width"] == 2


def test_select_example_indices_event_similarity_prefers_matching_event_vector() -> None:
    pool_dates = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[D]")
    pool_histories = np.array(
        [
            [70.0, 70.0, 70.0],
            [70.0, 70.0, 70.0],
        ],
        dtype=float,
    )
    pool_forecasts = np.array([[73.0, 74.0], [69.0, 68.0]], dtype=float)
    pool_truth = np.array([[73.0, 74.0], [69.0, 68.0]], dtype=float)
    retrieval_feature_map = {
        pd.Timestamp("2024-01-01"): np.array([2.0, 0.0], dtype=float),
        pd.Timestamp("2024-01-02"): np.array([-2.0, 0.0], dtype=float),
    }
    prepared = _prepare_example_pool(
        pool_dates=pool_dates,
        pool_histories=pool_histories,
        pool_forecasts=pool_forecasts,
        pool_truth=pool_truth,
        selection_mode="event_similarity",
        feature_window=3,
        retrieval_feature_map=retrieval_feature_map,
        retrieval_feature_width=2,
    )
    picked = _select_example_indices(
        prepared,
        reference_date=np.datetime64("2024-01-03"),
        reference_history=np.array([70.0, 70.0, 70.0], dtype=float),
        selection_mode="event_similarity",
        k_examples=1,
        feature_window=3,
        lookback_days=365,
        reference_retrieval_features=np.array([1.8, 0.0], dtype=float),
    )
    assert picked.tolist() == [0]
