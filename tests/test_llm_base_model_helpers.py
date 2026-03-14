from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.run_experiment import (
    _profile_regime_tag,
    apply_rule_gate,
    build_exogenous_summary,
    build_aux_teacher_case_summary,
    apply_delta_calibration,
    build_llm_refiner_config,
    build_retrieval_feature_vector,
    build_rule_gate_candidate_masks,
    build_rule_gate_feature_frame,
    evaluate_rule_gate_candidates,
    fit_delta_calibration_scales,
    infer_llm_market_name,
    llm_method_requires_base_forecast,
    llm_result_name,
    normalize_llm_base_model_name,
    parse_name_list,
    resolve_exogenous_feature_names,
    select_balanced_long_horizon_indices,
    select_counterexample_index,
    select_error_stratified_indices,
    select_skill_tag_recent_high_error_indices,
    select_similarity_error_hybrid_indices,
    select_top_score_indices,
    select_utility_mmr_indices,
    select_utility_score_indices,
    summarize_hindsight_feedback,
)


def test_normalize_llm_base_model_name_maps_common_aliases():
    assert normalize_llm_base_model_name(None) == "tsm"
    assert normalize_llm_base_model_name("ridge") == "linear_ridge"
    assert normalize_llm_base_model_name("Linear-Ridge") == "linear_ridge"
    assert normalize_llm_base_model_name("seasonal-naive") == "seasonal_naive"


def test_llm_method_requires_base_forecast_only_for_tsm_prefixed_methods():
    assert llm_method_requires_base_forecast("TSM+LLM-COT-RF") is True
    assert llm_method_requires_base_forecast("TSM+NEWS-DRIFT") is True
    assert llm_method_requires_base_forecast("CoT-RF") is False
    assert llm_method_requires_base_forecast("DP") is False


def test_llm_result_name_relabels_base_forecast_methods_for_non_tsm_models():
    assert llm_result_name("TSM+LLM-COT-RF", "tsm") == "TSM+LLM-COT-RF"
    assert llm_result_name("TSM+LLM-COT-RF", "linear_ridge") == "linear_ridge+LLM-COT-RF"
    assert llm_result_name("TSM+NEWS-DRIFT", "linear_ridge") == "linear_ridge+NEWS-DRIFT"
    assert llm_result_name("CoT-RF", "linear_ridge") == "CoT-RF"


def test_infer_llm_market_name_maps_known_instruments():
    assert infer_llm_market_name({"instrument": "UKA_FUTURES"}) == "UK ETS carbon allowance market"
    assert infer_llm_market_name({"instrument": "EUA_FUTURES"}) == "EU ETS carbon allowance market"
    assert infer_llm_market_name({"instrument": "ABC"}) == "ABC carbon allowance market"


def test_parse_name_list_handles_strings_and_sequences():
    assert parse_name_list("a, b ,c") == ["a", "b", "c"]
    assert parse_name_list(["a", None, " b "]) == ["a", "b"]


def test_resolve_exogenous_feature_names_prefers_uk_specific_features():
    feature_cols = [
        "y",
        "auction_volume",
        "target_volume",
        "uk_icap_primary_print_day",
        "y_vol_20d",
    ]
    selected = resolve_exogenous_feature_names(
        feature_cols,
        target_col="y",
        max_features=3,
    )
    assert selected == [
        "target_volume",
        "uk_icap_primary_print_day",
        "y_vol_20d",
    ]


def test_build_retrieval_feature_vector_encodes_numeric_and_binary_state():
    feature_cols = ["y", "target_volume", "is_auction_day"]
    window = np.array(
        [
            [70.0, 100.0, 0.0],
            [71.0, 105.0, 0.0],
            [72.0, 110.0, 1.0],
            [73.0, 120.0, 1.0],
        ],
        dtype=float,
    )
    vector = build_retrieval_feature_vector(
        window,
        feature_cols,
        target_col="y",
        include_features=["target_volume", "is_auction_day"],
        max_features=2,
        feature_window=4,
    )
    assert vector.shape == (4,)
    assert np.allclose(vector[:2], np.array([120.0, 20.0]))
    assert np.allclose(vector[2:], np.array([1.0, 0.5]))


def test_build_exogenous_summary_can_focus_on_requested_features():
    feature_cols = ["y", "target_range_pct", "target_volume", "is_auction_day"]
    window = np.array(
        [
            [70.0, 1.0, 100.0, 0.0],
            [71.0, 1.2, 105.0, 0.0],
            [72.0, 1.3, 115.0, 1.0],
            [73.0, 1.1, 120.0, 1.0],
        ],
        dtype=float,
    )
    summary = build_exogenous_summary(
        window,
        feature_cols,
        target_col="y",
        max_features=2,
        include_features=["is_auction_day", "target_volume"],
    )
    assert list(summary.keys()) == ["is_auction_day", "target_volume"]


def test_fit_delta_calibration_scales_shared_only_changes_selected_horizons():
    y_true = np.array([[10.0, 10.0, 12.0, 12.0]], dtype=float)
    base_pred = np.array([[10.0, 10.0, 10.0, 10.0]], dtype=float)
    llm_pred = np.array([[10.0, 10.0, 14.0, 14.0]], dtype=float)
    scales = fit_delta_calibration_scales(
        y_true,
        base_pred,
        llm_pred,
        horizons=[1, 2, 3, 4],
        target_horizons=[3, 4],
        min_scale=0.0,
        max_scale=1.0,
        shared=True,
    )
    assert scales[1] == 1.0
    assert scales[2] == 1.0
    assert np.isclose(scales[3], 0.5)
    assert np.isclose(scales[4], 0.5)


def test_apply_delta_calibration_scales_selected_horizons_only():
    base_pred = np.array([[10.0, 10.0, 10.0, 10.0]], dtype=float)
    llm_pred = np.array([[10.0, 11.0, 14.0, 16.0]], dtype=float)
    calibrated = apply_delta_calibration(
        base_pred,
        llm_pred,
        scales={3: 0.5, 4: 0.25},
        pred_len=4,
    )
    assert np.allclose(calibrated[0], np.array([10.0, 11.0, 12.0, 11.5]))


def test_select_top_score_indices_prefers_higher_scores_then_recency():
    candidate_indices = np.array([0, 1, 2, 3], dtype=int)
    pool_dates = np.array(
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        dtype="datetime64[D]",
    )
    scores = np.array([1.0, 3.0, 3.0, 2.0], dtype=float)
    selected = select_top_score_indices(candidate_indices, pool_dates, scores, 2)
    assert selected.tolist() == [1, 2]


def test_select_balanced_long_horizon_indices_covers_h20_and_h30():
    candidate_indices = np.array([0, 1, 2, 3, 4], dtype=int)
    pool_dates = np.array(
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        dtype="datetime64[D]",
    )
    h20_scores = np.array([9.0, 8.0, 1.0, 0.5, 0.1], dtype=float)
    h30_scores = np.array([0.1, 0.5, 8.5, 9.5, 1.0], dtype=float)
    selected = select_balanced_long_horizon_indices(
        candidate_indices,
        pool_dates,
        h20_scores,
        h30_scores,
        4,
    )
    assert selected.tolist() == [0, 1, 2, 3]


def test_select_skill_tag_recent_high_error_indices_filters_by_tag_overlap():
    candidate_indices = np.array([0, 1, 2, 3], dtype=int)
    pool_dates = np.array(
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        dtype="datetime64[D]",
    )
    pool_tags = np.array(
        [
            "trend5=pos|trend20=pos|drift20=pos|drift30=pos|vol=normal",
            "trend5=pos|trend20=pos|drift20=neg|drift30=neg|vol=normal",
            "trend5=pos|trend20=pos|drift20=pos|drift30=pos|vol=normal",
            "trend5=neg|trend20=neg|drift20=neg|drift30=neg|vol=elevated",
        ],
        dtype=object,
    )
    pool_scores = np.array([1.0, 5.0, 3.0, 4.0], dtype=float)
    reference_tag = "trend5=pos|trend20=pos|drift20=pos|drift30=pos|vol=normal"
    selected = select_skill_tag_recent_high_error_indices(
        candidate_indices,
        pool_dates,
        pool_tags,
        pool_scores,
        reference_tag,
        2,
        min_overlap=4,
    )
    assert selected.tolist() == [0, 2]


def test_select_counterexample_index_picks_similar_low_metric_case():
    candidate_indices = np.array([0, 1, 2, 3], dtype=int)
    pool_case_profiles = np.array(
        [
            [10.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [10.2, 0.1, 0.0, 1.0, 0.0, 0.0, 0.0],
            [50.0, 5.0, 4.0, 4.0, 5.0, 6.0, 7.0],
            [9.9, -0.1, 0.0, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    reference_profile = np.array([10.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=float)
    pool_metric = np.array([10.0, 0.2, 15.0, 0.1], dtype=float)
    picked = select_counterexample_index(
        candidate_indices,
        pool_case_profiles,
        reference_profile,
        pool_metric,
        exclude_indices=[0],
        quantile=0.5,
    )
    assert picked == 1


def test_summarize_hindsight_feedback_detects_long_horizon_overshoot():
    feedback = summarize_hindsight_feedback({5: 0.2, 20: 1.5, 30: 1.0})
    assert "overshot" in feedback.lower()
    assert "h20/h30" in feedback


def test_summarize_hindsight_feedback_supports_compact_tags():
    feedback = summarize_hindsight_feedback({5: -0.2, 20: 1.5, 30: 1.0}, mode="compact_tags")
    assert "long_tail=overshoot" in feedback
    assert "h5=undershoot" in feedback
    assert "action=down" in feedback


def test_summarize_hindsight_feedback_supports_long_horizon_only_mode():
    feedback = summarize_hindsight_feedback({5: -1.2, 20: -1.5, 30: -1.0}, mode="long_horizon_only")
    assert "undershot" in feedback.lower()
    assert "h20/h30" in feedback


def test_build_aux_teacher_case_summary_reports_deltas_and_gain():
    summary = build_aux_teacher_case_summary(
        base_forecast=np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0,
                                110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0,
                                120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0, 129.0]),
        aux_forecast=np.array([100.0, 101.0, 102.0, 103.0, 103.0, 105.0, 106.0, 107.0, 108.0, 109.0,
                               110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 117.0,
                               120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0, 126.0]),
        teacher_name="linear_ridge",
        truth=np.array([100.0] * 30),
        horizons=[5, 20, 30],
    )
    assert "linear ridge_delta_pct" in summary
    assert "linear ridge_gain_long" in summary

def test_select_error_stratified_indices_prefers_diverse_error_modes():
    candidate_indices = np.array([0, 1, 2, 3, 4], dtype=int)
    pool_dates = np.array([
        "2024-01-01",
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
    ], dtype="datetime64[D]")
    pool_error_profiles = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0, 0.0],
        [-5.0, 0.0, 0.0, 0.0],
        [0.0, 5.0, 0.0, 0.0],
        [0.0, -5.0, 0.0, 0.0],
    ], dtype=float)
    selected = select_error_stratified_indices(
        candidate_indices=candidate_indices,
        pool_dates=pool_dates,
        pool_error_profiles=pool_error_profiles,
        k_examples=3,
    )
    assert selected.tolist() == [0, 3, 4]


def test_select_similarity_error_hybrid_indices_keeps_nearest_and_diverse_examples():
    candidate_indices = np.array([0, 1, 2, 3, 4, 5], dtype=int)
    pool_dates = np.array(
        [
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-06",
        ],
        dtype="datetime64[D]",
    )
    pool_case_profiles = np.array(
        [
            [10.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [11.0, 0.1, 0.0, 1.0, 0.0, 0.0, 0.0],
            [50.0, 5.0, 4.0, 4.0, 5.0, 6.0, 7.0],
            [9.8, -0.1, 0.0, 1.1, 0.0, 0.0, 0.0],
            [60.0, -5.0, -4.0, 4.0, -5.0, -6.0, -7.0],
            [12.0, 0.2, 0.0, 0.9, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    pool_error_profiles = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0, 0.0],
            [-4.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, -3.0, 0.0, 0.0],
            [0.0, 0.0, 2.5, 0.0],
        ],
        dtype=float,
    )
    reference_profile = np.array([10.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=float)
    selected = select_similarity_error_hybrid_indices(
        candidate_indices=candidate_indices,
        pool_dates=pool_dates,
        pool_case_profiles=pool_case_profiles,
        pool_error_profiles=pool_error_profiles,
        reference_profile=reference_profile,
        k_examples=4,
        n_similarity=2,
    )
    assert selected.tolist() == [0, 1, 3, 4]


def test_select_utility_score_indices_favors_recent_similar_hard_cases():
    candidate_indices = np.array([0, 1, 2, 3, 4, 5], dtype=int)
    pool_dates = np.array(
        [
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-06",
        ],
        dtype="datetime64[D]",
    )
    pool_case_profiles = np.array(
        [
            [10.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [10.2, 0.1, 0.0, 1.0, 0.1, 0.1, 0.1],
            [10.3, 0.2, 0.1, 1.0, 0.2, 0.2, 0.2],
            [14.0, 4.0, 4.0, 3.0, 5.0, 5.0, 5.0],
            [10.4, 0.2, 0.1, 0.9, 0.1, 0.2, 0.2],
            [15.0, -4.0, -4.0, 3.2, -5.0, -5.0, -5.0],
        ],
        dtype=float,
    )
    pool_path_mse = np.array([0.5, 1.0, 1.2, 2.5, 2.0, 2.7], dtype=float)
    reference_profile = np.array([10.3, 0.2, 0.1, 1.0, 0.1, 0.2, 0.2], dtype=float)
    selected = select_utility_score_indices(
        candidate_indices=candidate_indices,
        pool_dates=pool_dates,
        pool_case_profiles=pool_case_profiles,
        pool_path_mse=pool_path_mse,
        reference_profile=reference_profile,
        k_examples=3,
    )
    assert selected.tolist() == [1, 2, 4]


def test_select_utility_mmr_indices_reduces_redundant_neighbors():
    candidate_indices = np.array([0, 1, 2, 3, 4], dtype=int)
    pool_dates = np.array(
        [
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
        ],
        dtype="datetime64[D]",
    )
    pool_case_profiles = np.array(
        [
            [10.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [10.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [10.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [13.0, 3.0, 2.0, 2.5, 3.0, 4.0, 4.5],
            [8.0, -3.0, -2.0, 2.5, -3.0, -4.0, -4.5],
        ],
        dtype=float,
    )
    pool_path_mse = np.array([1.0, 1.1, 1.2, 2.4, 2.5], dtype=float)
    reference_profile = np.array([10.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=float)
    selected = select_utility_mmr_indices(
        candidate_indices=candidate_indices,
        pool_dates=pool_dates,
        pool_case_profiles=pool_case_profiles,
        pool_path_mse=pool_path_mse,
        reference_profile=reference_profile,
        k_examples=3,
        mmr_lambda=0.65,
    )
    assert selected.tolist() == [2, 3, 4]


def test_profile_regime_tag_formats_compact_retrieval_label():
    tag = _profile_regime_tag(np.array([70.0, 1.0, 2.0, 1.2, 0.5, 1.0, 1.5], dtype=float))
    assert "trend5=pos" in tag
    assert "drift30=pos" in tag


def test_build_llm_refiner_config_merges_top_level_hdelta():
    config = SimpleNamespace(
        llm={"model": "qwen3-vl-4b-gpu", "cot_rf": {"strict_json_prompt": True}},
        target={"instrument": "UKA_FUTURES", "currency": "GBP"},
        raw={"hdelta": {"key_horizons": [1, 5, 20, 30], "case_conditioned": True}},
        delta=None,
        hprice=None,
        norm_delta=None,
        news_drift=None,
    )
    merged = build_llm_refiner_config(config)
    assert merged["currency"] == "GBP"
    assert merged["market_name"] == "UK ETS carbon allowance market"
    assert merged["hdelta"]["case_conditioned"] is True


def test_build_rule_gate_feature_frame_extracts_adjustments_and_guidance():
    base_pred = np.array(
        [
            [100.0, 101.0, 102.0, 103.0, 104.0] + [104.0] * 25,
            [100.0, 101.0, 102.0, 103.0, 104.0] + [104.0] * 25,
        ],
        dtype=float,
    )
    llm_pred = np.array(
        [
            [100.0, 101.0, 102.0, 103.0, 104.52] + [103.48] * 15 + [104.0] * 10,
            [100.0, 101.0, 102.0, 103.0, 103.48] + [104.0] * 25,
        ],
        dtype=float,
    )
    metadata = [
        {
            "success": True,
            "matched_teaching_dates": ["2025-01-01", "2025-01-03"],
            "dynamic_frozen_horizons": [1],
            "structured_horizon_guidance": {
                5: {"mode": "adjust", "confidence": "high", "magnitude": "small"},
                20: {"mode": "adjust", "confidence": "medium", "magnitude": "small"},
                30: {"mode": "freeze", "confidence": "medium", "magnitude": "zero"},
            },
        },
        {
            "success": True,
            "matched_teaching_dates": ["2025-01-05"],
            "dynamic_frozen_horizons": [1, 20],
            "structured_horizon_guidance": {
                5: {"mode": "adjust", "confidence": "high", "magnitude": "small"},
                20: {"mode": "adjust", "confidence": "low", "magnitude": "small"},
                30: {"mode": "adjust", "confidence": "low", "magnitude": "small"},
            },
        },
    ]

    feature_df = build_rule_gate_feature_frame(
        base_pred=base_pred,
        llm_pred=llm_pred,
        metadata=metadata,
        key_horizons=[1, 5, 20, 30],
    )

    assert feature_df.loc[0, "matched_teaching_count"] == 2
    assert feature_df.loc[0, "dynamic_frozen_count"] == 1
    assert feature_df.loc[0, "h5_sign"] == 1
    assert feature_df.loc[0, "h20_confidence_rank"] == 1
    assert feature_df.loc[0, "h30_mode"] == "freeze"
    assert feature_df.loc[1, "matched_teaching_count"] == 1
    assert feature_df.loc[1, "h5_sign"] == -1


def test_evaluate_rule_gate_candidates_selects_conflict_rejector_when_it_helps():
    base_pred = np.array(
        [
            [100.0, 100.0, 100.0, 100.0, 100.0] + [100.0] * 25,
            [100.0, 100.0, 100.0, 100.0, 100.0] + [100.0] * 25,
            [100.0, 100.0, 100.0, 100.0, 100.0] + [100.0] * 25,
        ],
        dtype=float,
    )
    llm_pred = base_pred.copy()
    llm_pred[0, 4] = 100.5
    llm_pred[0, 19] = 99.4
    llm_pred[0, 29] = 99.2
    llm_pred[1, 4] = 100.4
    llm_pred[1, 19] = 100.3
    llm_pred[1, 29] = 100.2
    llm_pred[2, 4] = 99.8
    llm_pred[2, 19] = 100.0
    llm_pred[2, 29] = 100.0

    y_true = base_pred.copy()
    y_true[0, 4] = 100.0
    y_true[0, 19] = 100.0
    y_true[0, 29] = 100.0
    y_true[1, 4] = 100.4
    y_true[1, 19] = 100.3
    y_true[1, 29] = 100.2
    y_true[2, 4] = 99.8
    y_true[2, 19] = 100.0
    y_true[2, 29] = 100.0

    metadata = [
        {
            "success": True,
            "matched_teaching_dates": ["2025-01-01", "2025-01-02"],
            "structured_horizon_guidance": {
                5: {"mode": "adjust", "confidence": "high", "magnitude": "small"},
                20: {"mode": "adjust", "confidence": "medium", "magnitude": "small"},
                30: {"mode": "adjust", "confidence": "medium", "magnitude": "small"},
            },
        },
        {
            "success": True,
            "matched_teaching_dates": ["2025-01-05", "2025-01-06", "2025-01-07"],
            "structured_horizon_guidance": {
                5: {"mode": "adjust", "confidence": "high", "magnitude": "small"},
                20: {"mode": "adjust", "confidence": "high", "magnitude": "small"},
                30: {"mode": "adjust", "confidence": "high", "magnitude": "small"},
            },
        },
        {
            "success": True,
            "matched_teaching_dates": ["2025-01-09", "2025-01-10"],
            "structured_horizon_guidance": {
                5: {"mode": "adjust", "confidence": "high", "magnitude": "small"},
                20: {"mode": "freeze", "confidence": "high", "magnitude": "zero"},
                30: {"mode": "freeze", "confidence": "high", "magnitude": "zero"},
            },
        },
    ]

    feature_df = build_rule_gate_feature_frame(base_pred, llm_pred, metadata, key_horizons=[1, 5, 20, 30])
    masks = build_rule_gate_candidate_masks(
        feature_df,
        candidates=["none", "conflict_any", "h30_negative"],
        low_match_threshold=2,
        abs_h30_thresholds_pct=[0.25],
    )
    assert masks["conflict_any"].tolist() == [True, False, False]

    summary_df, gated_predictions = evaluate_rule_gate_candidates(
        y_true=y_true,
        base_pred=base_pred,
        llm_pred=llm_pred,
        feature_df=feature_df,
        candidates=["none", "conflict_any", "h30_negative"],
        low_match_threshold=2,
        abs_h30_thresholds_pct=[0.25],
        horizons=[1, 5, 20, 30],
    )

    best_candidate = str(summary_df.loc[summary_df["mse_path"].idxmin(), "candidate"])
    assert best_candidate == "conflict_any"
    assert summary_df.loc[summary_df["candidate"] == "conflict_any", "flagged_count"].iloc[0] == 1

    gated = apply_rule_gate(base_pred, llm_pred, masks["conflict_any"])
    assert np.allclose(gated_predictions["conflict_any"], gated)
    assert np.allclose(gated[0], base_pred[0])
    assert np.allclose(gated[1], llm_pred[1])
