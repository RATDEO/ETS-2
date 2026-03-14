from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.validate_regime_gates_archive import (
    HPRICE_4B_MODEL,
    PAPER_MODEL,
    _candidate_definitions,
    _resolve_llm_cfg,
)


def _args(**overrides: object) -> SimpleNamespace:
    base = dict(
        llm4b_base_url="http://192.168.1.140:9877/v1",
        llm4b_model="qwen3-vl-4b-gpu",
        llm35b_base_url="http://192.168.1.140:9881/v1",
        llm35b_model="qwen3.5-35b-a3b-ud-q4-k-xl",
        paper_base_url=None,
        paper_model=None,
        paper_mode="fullpath",
        divergent_base_url=None,
        divergent_model=None,
        divergent_mode="raw",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_candidate_definitions_default_to_fullpath_paper_role() -> None:
    candidates = _candidate_definitions(_args())
    assert candidates[0].display_name == PAPER_MODEL
    assert candidates[0].method == "TSM+LLM-COT-SENT-RF"
    assert candidates[0].sentiment_path == "data/news/daily_sentiment.csv"


def test_candidate_definitions_support_hprice_paper_role() -> None:
    candidates = _candidate_definitions(_args(paper_mode="hprice"))
    assert candidates[0].display_name == HPRICE_4B_MODEL
    assert candidates[0].method == "TSM+LLM-COT-SENT-RF-HPRICE"
    assert (
        candidates[0].sentiment_path
        == "data/news/daily_sentiment_qwen_votes3_daily3_gate_policy_shipping_event90_proxy.csv"
    )


def test_resolve_llm_cfg_applies_hprice_settings() -> None:
    candidate = _candidate_definitions(_args(paper_mode="hprice"))[0]
    cfg = _resolve_llm_cfg(base_cfg={"cot_rf": {}, "sentiment": {}}, candidate=candidate, api_key="deo", timeout_seconds=60.0)
    assert cfg["methods"] == ["TSM+LLM-COT-SENT-RF-HPRICE"]
    assert cfg["sentiment"]["path"] == candidate.sentiment_path
    assert cfg["cot_rf"]["retain_context"] is False
    assert cfg["cot_rf"]["strict_json_prompt"] is True
    assert cfg["hprice"]["key_horizons"] == [1, 10, 20, 30]
    assert cfg["hprice"]["max_adjustment_pct"] == 1.0
    assert cfg["hprice"]["freeze_horizons"] == [1]
    assert cfg["hprice"]["sentiment_secondary"] is True
