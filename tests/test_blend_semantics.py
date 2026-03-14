from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.refine import (
    blend_mode_from_config,
    method_supports_internal_blend,
    method_uses_internal_blend,
)
from src.run_experiment import restrict_to_recent_tail, select_eval_indices


def test_cot_sent_rf_ignores_internal_blend_config():
    cfg = {"blend": {"mode": "horizon_weighted", "min_weight": 0.15, "max_weight": 0.6}}
    assert blend_mode_from_config(cfg) == "horizon_weighted"
    assert method_supports_internal_blend("TSM+LLM-COT-SENT-RF") is False
    assert method_uses_internal_blend("TSM+LLM-COT-SENT-RF", cfg) is False


def test_delta_method_uses_internal_blend_unless_disabled():
    assert method_supports_internal_blend("TSM+LLM-COT-SENT-RF-DELTA") is True
    assert method_uses_internal_blend("TSM+LLM-COT-SENT-RF-DELTA", {}) is True
    assert method_uses_internal_blend(
        "TSM+LLM-COT-SENT-RF-DELTA",
        {"blend": {"mode": "none"}},
    ) is False


def test_restrict_to_recent_tail_by_fraction():
    idx = [0, 1, 2, 3, 4, 5, 6, 7]
    assert restrict_to_recent_tail(idx, tail_fraction=0.25).tolist() == [6, 7]
    assert restrict_to_recent_tail(idx, tail_fraction=0.5).tolist() == [4, 5, 6, 7]


def test_restrict_to_recent_tail_respects_min_samples():
    idx = [10, 11, 12, 13, 14]
    assert restrict_to_recent_tail(idx, tail_fraction=0.2, min_samples=3).tolist() == [12, 13, 14]


def test_recent_tail_scope_can_feed_latest_eval_slice():
    idx = list(range(20))
    tail = restrict_to_recent_tail(idx, tail_fraction=0.1, min_samples=4)
    selected = select_eval_indices(tail, max_samples=4, strategy="first")
    assert selected.tolist() == [16, 17, 18, 19]
