from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.prompts import (
    CoTRFHorizonDeltaApplyTemplate,
    CoTRFHorizonDeltaStructuredReflectionStrictJSONTemplate,
    CoTSentRFReflectionTemplate,
)


def test_cot_sent_rf_reflection_template_accepts_exogenous_summary() -> None:
    template = CoTSentRFReflectionTemplate()
    prompt = template.format(
        examples=[
            {
                "history": np.array([70.0, 71.0, 72.0]),
                "forecast": np.array([73.0, 74.0]),
                "truth": np.array([72.5, 73.5]),
                "sentiment_history": np.array([0.0, 0.33, 0.67]),
                "date": "2026-02-26",
                "exogenous_summary": {"official_event_balance": "2 events/30d, net=-1.00"},
            }
        ],
        pred_len=2,
        history_points=3,
        sentiment_points=3,
        exogenous_summary={"official_recent_events": "2026-02-25 auction_supply: Heavy auction result"},
    )
    assert "Current Official / Exogenous Context" in prompt
    assert "official_recent_events" in prompt
    assert "official_event_balance" in prompt


def test_hdelta_structured_reflection_template_supports_reasoning_styles_and_memory() -> None:
    template = CoTRFHorizonDeltaStructuredReflectionStrictJSONTemplate()
    prompt = template.format(
        examples=[
            {
                "history": np.array([70.0, 71.0, 72.0, 73.0, 74.0]),
                "forecast": np.array([74.0] * 30),
                "truth": np.array([73.8] * 30),
                "date": "2026-02-26",
                "anchor_error_pct": {1: 0.0, 5: 0.4, 20: -0.2, 30: 0.0},
                "case_summary": {"change_5d_pct": "+1.0", "vol_20d": "0.8"},
            }
        ],
        pred_len=30,
        history_points=5,
        key_horizons=[1, 5, 20, 30],
        reasoning_style="least_to_most",
        reflection_memory=["Avoid unsupported h30 drift."],
    )
    assert "Persistent UK Error Memory" in prompt
    assert "Avoid unsupported h30 drift." in prompt
    assert "Decide h1 first" in prompt


def test_hdelta_apply_template_supports_program_of_thought_style() -> None:
    template = CoTRFHorizonDeltaApplyTemplate()
    prompt = template.format(
        history=np.array([70.0, 71.0, 72.0, 73.0, 74.0]),
        dates=["2026-02-20", "2026-02-21", "2026-02-24", "2026-02-25", "2026-02-26"],
        tsm_forecast=np.array([74.1] * 30),
        rules_text="- h5: positive adjustment; confidence=high; magnitude=small; reason=undershoot",
        pred_len=30,
        key_horizons=[1, 5, 20, 30],
        frozen_horizons=[1],
        current_case_summary={"change_5d_pct": "+1.0"},
        structured_horizon_guidance={
            1: {"mode": "freeze", "preferred_sign": "zero", "confidence": "high", "magnitude": "zero", "reason": "aligned"},
            5: {"mode": "adjust", "preferred_sign": "positive", "confidence": "high", "magnitude": "small", "reason": "undershoot"},
            20: {"mode": "freeze", "preferred_sign": "zero", "confidence": "medium", "magnitude": "zero", "reason": "mixed"},
            30: {"mode": "freeze", "preferred_sign": "zero", "confidence": "medium", "magnitude": "zero", "reason": "mixed"},
        },
        apply_style="program_of_thought",
        reflection_memory=["Do not create new h30 drift."],
    )
    assert "Persistent UK Error Memory" in prompt
    assert "Do not create new h30 drift." in prompt
    assert "internally compute for each horizon" in prompt


def test_hdelta_structured_reflection_template_supports_structural_reasoning_styles() -> None:
    template = CoTRFHorizonDeltaStructuredReflectionStrictJSONTemplate()
    prompt = template.format(
        examples=[
            {
                "history": np.array([70.0, 71.0, 72.0, 73.0, 74.0]),
                "forecast": np.array([74.0] * 30),
                "truth": np.array([73.8] * 30),
                "date": "2026-02-26",
                "anchor_error_pct": {1: 0.0, 5: 0.4, 20: -0.2, 30: 0.0},
                "case_summary": {"change_5d_pct": "+1.0", "vol_20d": "0.8"},
            }
        ],
        pred_len=30,
        history_points=5,
        key_horizons=[1, 5, 20, 30],
        reasoning_style="rarr_attribution",
    )
    assert "Ground every horizon decision in explicit matched-example evidence" in prompt
    assert "cite example rank" in prompt


def test_hdelta_structured_reflection_template_supports_followup_styles() -> None:
    template = CoTRFHorizonDeltaStructuredReflectionStrictJSONTemplate()
    prompt = template.format(
        examples=[
            {
                "history": np.array([70.0, 71.0, 72.0, 73.0, 74.0]),
                "forecast": np.array([74.0] * 30),
                "truth": np.array([73.8] * 30),
                "date": "2026-02-26",
                "anchor_error_pct": {1: 0.0, 5: 0.4, 20: -0.2, 30: 0.0},
                "case_summary": {"change_5d_pct": "+1.0", "vol_20d": "0.8"},
            }
        ],
        pred_len=30,
        history_points=5,
        key_horizons=[1, 5, 20, 30],
        reasoning_style="analogical_step_back",
    )
    assert "most analogous" in prompt
    assert "higher-level regime" in prompt


def test_hdelta_apply_template_supports_verifier_and_citation_styles() -> None:
    template = CoTRFHorizonDeltaApplyTemplate()
    common_kwargs = {
        "history": np.array([70.0, 71.0, 72.0, 73.0, 74.0]),
        "dates": ["2026-02-20", "2026-02-21", "2026-02-24", "2026-02-25", "2026-02-26"],
        "tsm_forecast": np.array([74.1] * 30),
        "rules_text": "- h20: negative adjustment; confidence=high; magnitude=small; reason=Example 1 shows overshoot",
        "pred_len": 30,
        "key_horizons": [1, 5, 20, 30],
        "frozen_horizons": [1],
        "current_case_summary": {"change_5d_pct": "+1.0"},
        "structured_horizon_guidance": {
            1: {"mode": "freeze", "preferred_sign": "zero", "confidence": "high", "magnitude": "zero", "reason": "aligned"},
            5: {"mode": "freeze", "preferred_sign": "zero", "confidence": "medium", "magnitude": "zero", "reason": "mixed"},
            20: {"mode": "adjust", "preferred_sign": "negative", "confidence": "high", "magnitude": "small", "reason": "Example 1 shows overshoot"},
            30: {"mode": "freeze", "preferred_sign": "zero", "confidence": "medium", "magnitude": "zero", "reason": "mixed"},
        },
    }
    verifier_prompt = template.format(**common_kwargs, apply_style="verifier_program")
    citation_prompt = template.format(**common_kwargs, apply_style="citation_bounded")
    assert "verify it against the structured guidance" in verifier_prompt
    assert "reason is evidence-grounded" in citation_prompt
