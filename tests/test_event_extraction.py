from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.news.event_extraction import _extract_json, build_event_extraction_prompt


def test_build_event_extraction_prompt_mentions_required_keys() -> None:
    prompt = build_event_extraction_prompt("Title", "Summary", "Body")
    assert "event_type" in prompt
    assert "affected_channel" in prompt
    assert "direction" in prompt


def test_extract_json_coerces_invalid_values() -> None:
    payload = _extract_json('{"is_relevant": true, "event_type": "bad", "affected_channel": "bad", "direction": "up", "intensity": 9, "expected_horizon": "tomorrow", "novelty": "x", "policy_stage": "y", "confidence": 3, "summary": "ok"}')
    assert payload["is_relevant"] is True
    assert payload["event_type"] == "other"
    assert payload["affected_channel"] == "other"
    assert payload["direction"] == "unclear"
    assert payload["intensity"] == 3
    assert payload["confidence"] == 1.0
