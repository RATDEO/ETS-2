from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.refine import (  # noqa: E402
    _extract_llm_observability,
    _merge_llm_observability,
)


def test_extract_llm_observability_reads_openai_compatible_usage():
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=120,
            completion_tokens=30,
            total_tokens=150,
        )
    )

    observation = _extract_llm_observability(
        response,
        latency_seconds=1.25,
        base_url="http://127.0.0.1:9881/v1",
    )

    assert observation == {
        "request_count": 1,
        "prompt_tokens": 120,
        "completion_tokens": 30,
        "total_tokens": 150,
        "latency_seconds": 1.25,
        "cache_hit": False,
        "cache_hit_count": 0,
        "base_url": "http://127.0.0.1:9881/v1",
    }


def test_merge_llm_observability_sums_multi_round_tool_calls():
    total = {}
    first = {
        "request_count": 1,
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "total_tokens": 110,
        "latency_seconds": 0.5,
        "cache_hit_count": 0,
        "base_url": "http://model/v1",
    }
    second = {
        "request_count": 1,
        "prompt_tokens": 140,
        "completion_tokens": 20,
        "total_tokens": 160,
        "latency_seconds": 0.75,
        "cache_hit_count": 0,
        "base_url": "http://model/v1",
    }

    _merge_llm_observability(total, first)
    _merge_llm_observability(total, second)

    assert total == {
        "request_count": 2,
        "prompt_tokens": 240,
        "completion_tokens": 30,
        "total_tokens": 270,
        "latency_seconds": 1.25,
        "cache_hit_count": 0,
        "base_url": "http://model/v1",
    }
