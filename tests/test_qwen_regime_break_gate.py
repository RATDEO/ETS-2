from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_qwen_regime_break_gate_v1 import (
    apply_regime_gate,
    build_regime_prompt,
    recent_headline_records,
)


def test_recent_headlines_respect_origin_cutoff() -> None:
    headlines = pd.DataFrame(
        {
            "seendate": ["2025-01-01", "2025-01-10", "2025-01-11"],
            "title": ["known old", "known current", "future leak"],
            "llm_score": [-1.0, 1.0, 1.0],
            "llm_importance": [5.0, 7.0, 10.0],
        }
    )
    records = recent_headline_records(headlines, pd.Timestamp("2025-01-10"), 30, 10)
    assert [record["title"] for record in records] == ["known old", "known current"]


def test_prompt_contains_no_unprovided_outcomes() -> None:
    prompt = build_regime_prompt(
        pd.Timestamp("2025-01-10"),
        {"spot_price": 40.0},
        np.linspace(40.0, 42.0, 30),
        {"evt_news_count_7d_sum": 3.0},
        [{"date": "2025-01-09", "title": "Known policy news", "importance": 8.0}],
    )
    assert "/no_think" in prompt
    assert "2025-01-09" in prompt
    assert "realized_price" not in prompt
    assert "y_true" not in prompt


def test_regime_gate_only_abstains_on_trusted_conflicts() -> None:
    base = np.full((4, 30), 100.0)
    refined = base.copy()
    refined[0, -1] = 95.0
    refined[1, -1] = 105.0
    refined[2, -1] = 95.0
    refined[3, -1] = 95.0
    directions = np.array(["above_base", "above_base", "near_base", "above_base"], dtype=object)
    confidences = np.array(["high", "high", "medium", "low"], dtype=object)
    gated, rejected = apply_regime_gate(base, refined, directions, confidences, "medium")
    assert rejected.tolist() == [True, False, True, False]
    assert np.allclose(gated[0], base[0])
    assert np.allclose(gated[1], refined[1])
    assert np.allclose(gated[3], refined[3])


def test_regime_gate_honors_external_activation_mask() -> None:
    base = np.full((2, 30), 100.0)
    refined = base.copy()
    refined[:, -1] = 95.0
    gated, rejected = apply_regime_gate(
        base,
        refined,
        np.array(["above_base", "above_base"], dtype=object),
        np.array(["high", "high"], dtype=object),
        "medium",
        activation_mask=np.array([False, True]),
    )
    assert rejected.tolist() == [False, True]
    assert np.allclose(gated[0], refined[0])
    assert np.allclose(gated[1], base[1])
