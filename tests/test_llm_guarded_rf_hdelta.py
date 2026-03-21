from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.refine import (
    CASE_RETRIEVAL_TOOL_NAME,
    COUNTEREXAMPLE_TOOL_NAME,
    DELTA_VERIFIER_TOOL_NAME,
    LLMRefiner,
    MARKET_MICROSTRUCTURE_TOOL_NAME,
    NUMERIC_ANALYSIS_TOOL_NAME,
    aggregate_horizon_adjustments,
    aggregate_structured_hdelta_guidance,
    apply_discrete_hdelta_actions,
    apply_structured_hdelta_coherence_guards,
    build_hdelta_retrieval_tag,
    build_hdelta_delta_verification_payload,
    build_market_microstructure_payload,
    build_numeric_analysis_payload,
    build_structured_case_retrieval_payload,
    derive_hdelta_freeze_counterexample,
    derive_hdelta_case_controls,
    enforce_structured_hdelta_adjustments,
    hdelta_guidance_to_rules_text,
    parse_horizon_deltas,
    parse_hdelta_reflection_guidance,
    select_structured_reflect_sample_budget,
)


def test_cot_rf_hdelta_routes_reflection_to_override_and_freezes_h1(monkeypatch):
    calls = []

    def fake_call(
        self,
        prompt,
        system_message,
        response_format=None,
        model_override=None,
        api_key_override=None,
        base_url_override=None,
        **kwargs,
    ):
        calls.append(
            {
                "prompt": prompt,
                "model_override": model_override,
                "api_key_override": api_key_override,
                "base_url_override": base_url_override,
                "response_format": response_format,
            }
        )
        if len(calls) == 1:
            return (
                '{"rules": ['
                '"Keep day 1 unchanged unless there is overwhelming evidence.", '
                '"Use small corrections only.", '
                '"Favor mid-horizon fixes over full-path rewrites.", '
                '"Avoid compounding a directional bias across all horizons."'
                "]}"
            )
        return '{"adjustments": {"h1": 0.7, "h5": 1.5, "h20": -0.5, "h30": 0.25}}'

    monkeypatch.setattr(LLMRefiner, "_call_llm", fake_call)

    refiner = LLMRefiner(
        {
            "provider": "openai",
            "model": "apply-model",
            "api_key": "deo",
            "base_url": "http://apply.local/v1",
            "currency": "GBP",
            "cot_rf": {
                "retain_context": False,
                "strict_json_prompt": True,
                "strict_json_response_format": True,
                "reflect_model": "reflect-model",
                "reflect_api_key": "reflect-key",
                "reflect_base_url": "http://reflect.local/v1",
            },
            "hdelta": {
                "key_horizons": [1, 5, 20, 30],
                "freeze_horizons": [1],
                "max_adjustment_pct": 2.0,
            },
        }
    )

    history = np.linspace(50.0, 55.0, 30)
    tsm_forecast = np.full(30, 60.0)
    dates = [f"2026-02-{day:02d}" for day in range(1, 31)]
    teaching_examples = [
        {
            "history": np.linspace(48.0, 54.0, 30),
            "forecast": np.full(30, 58.0),
            "truth": np.full(30, 57.0),
            "date": "2025-12-01",
        }
    ]

    forecast, metadata = refiner.refine(
        method="TSM+LLM-COT-RF-HDELTA",
        history=history,
        dates=dates,
        tsm_forecast=tsm_forecast,
        pred_len=30,
        teaching_examples=teaching_examples,
    )

    assert forecast is not None
    assert np.isclose(forecast[0], 60.0)
    assert np.isclose(forecast[4], 60.0 * 1.015)
    assert calls[0]["model_override"] == "reflect-model"
    assert calls[0]["api_key_override"] == "reflect-key"
    assert calls[0]["base_url_override"] == "http://reflect.local/v1"
    assert calls[1]["model_override"] is None
    assert metadata["reflect_model"] == "reflect-model"


def test_cot_rf_hdelta_retries_without_response_format_when_endpoint_rejects_it(monkeypatch):
    calls = []

    def fake_call(
        self,
        prompt,
        system_message,
        response_format=None,
        model_override=None,
        api_key_override=None,
        base_url_override=None,
        **kwargs,
    ):
        calls.append(response_format)
        if len(calls) == 1:
            raise RuntimeError("Failed to parse input at pos 1679")
        if len(calls) == 2:
            return (
                '{"rules": ['
                '"Keep day 1 unchanged.", '
                '"Use bounded changes only.", '
                '"Prefer medium-horizon corrections.", '
                '"Avoid adding a new long-horizon drift."'
                "]}"
            )
        return '{"adjustments": {"h1": 0.0, "h5": 0.5, "h20": 0.0, "h30": 0.0}}'

    monkeypatch.setattr(LLMRefiner, "_call_llm", fake_call)

    refiner = LLMRefiner(
        {
            "provider": "openai",
            "model": "apply-model",
            "api_key": "deo",
            "base_url": "http://apply.local/v1",
            "cot_rf": {
                "retain_context": False,
                "strict_json_prompt": True,
                "strict_json_response_format": True,
                "reflect_model": "reflect-model",
            },
            "hdelta": {
                "key_horizons": [1, 5, 20, 30],
                "freeze_horizons": [1],
                "max_adjustment_pct": 1.0,
            },
        }
    )

    forecast, metadata = refiner.refine(
        method="TSM+LLM-COT-RF-HDELTA",
        history=np.linspace(50.0, 55.0, 30),
        dates=[f"2026-02-{day:02d}" for day in range(1, 31)],
        tsm_forecast=np.full(30, 60.0),
        pred_len=30,
        teaching_examples=[
            {
                "history": np.linspace(48.0, 54.0, 30),
                "forecast": np.full(30, 58.0),
                "truth": np.full(30, 57.0),
                "date": "2025-12-01",
            }
        ],
    )

    assert forecast is not None
    assert calls[0] is not None
    assert calls[1] is None
    assert metadata["response_format_fallback"] is True


def test_derive_hdelta_case_controls_freezes_conflicted_horizons_and_scales_bounds():
    history = np.linspace(50.0, 55.0, 30)
    forecast = np.full(30, 60.0)
    teaching_examples = [
        {
            "history": np.linspace(48.0, 54.0, 30),
            "forecast": np.full(30, 58.0),
            "truth": np.array([57.7] * 30),
            "date": "2025-12-01",
        },
        {
            "history": np.linspace(48.2, 54.2, 30),
            "forecast": np.full(30, 58.0),
            "truth": np.array([57.7] * 30),
            "date": "2025-12-02",
        },
        {
            "history": np.linspace(70.0, 65.0, 30),
            "forecast": np.full(30, 63.0),
            "truth": np.array([64.5] * 30),
            "date": "2025-11-15",
        },
    ]
    teaching_examples[0]["truth"][4] = 58.4
    teaching_examples[1]["truth"][4] = 57.6

    controls = derive_hdelta_case_controls(
        history=history,
        forecast=forecast,
        teaching_examples=teaching_examples,
        key_horizons=[1, 5, 20, 30],
        config={
            "max_adjustment_pct": 0.8,
            "case_match_top_k": 2,
            "case_min_examples": 2,
            "case_min_sign_agreement": 0.67,
            "case_min_mean_abs_error_pct": 0.20,
            "case_bound_scale": 0.75,
            "case_min_bound_pct": 0.10,
        },
        currency="GBP",
    )

    assert controls["matched_dates"] == ["2025-12-01", "2025-12-02"]
    assert 5 in controls["dynamic_freeze_horizons"]
    assert 1 not in controls["dynamic_freeze_horizons"]
    assert controls["per_horizon_max_adjustment_pct"][5] == 0.0
    assert 0.10 <= controls["per_horizon_max_adjustment_pct"][1] <= 0.8
    assert any("actionable" in line for line in controls["horizon_guidance_summary"])
    assert any("freeze to 0.0" in line for line in controls["horizon_guidance_summary"])


def test_parse_hdelta_reflection_guidance_and_rules_text():
    parsed = parse_hdelta_reflection_guidance(
        '{"horizons": {'
        '"h1": {"mode": "freeze", "preferred_sign": "zero", "confidence": "high", "magnitude": "zero", "reason": "Short horizon already aligned."}, '
        '"h5": {"mode": "adjust", "preferred_sign": "positive", "confidence": "high", "magnitude": "small", "reason": "Matched examples undershot around h5."}, '
        '"h20": {"mode": "adjust", "preferred_sign": "positive", "confidence": "medium", "magnitude": "small", "reason": "Mid-horizon bias stays slightly negative."}, '
        '"h30": {"mode": "freeze", "preferred_sign": "zero", "confidence": "medium", "magnitude": "zero", "reason": "Long-horizon evidence is mixed."}'
        "}}",
        key_horizons=[1, 5, 20, 30],
    )

    assert parsed is not None
    assert parsed[5]["preferred_sign"] == "positive"
    assert parsed[30]["mode"] == "freeze"

    rules_text = hdelta_guidance_to_rules_text(parsed)
    assert "h5: positive adjustment" in rules_text
    assert "h30: keep unchanged" in rules_text


def test_build_numeric_analysis_payload_reports_horizon_bounds_and_drifts():
    history = np.array([50.0, 51.0, 52.0, 53.0, 54.0, 55.0], dtype=float)
    forecast = np.array([55.5, 56.0, 57.0, 58.0, 59.0], dtype=float)
    payload = build_numeric_analysis_payload(
        history=history,
        tsm_forecast=forecast,
        key_horizons=[1, 5],
        per_horizon_max_adjustment_pct={1: 0.0, 5: 1.5},
        structured_horizon_guidance={
            1: {"mode": "freeze", "preferred_sign": "zero"},
            5: {"mode": "adjust", "preferred_sign": "positive", "confidence": "medium", "magnitude": "small"},
        },
        current_case_summary={"change_5d_pct": "+10.0"},
        currency="GBP",
    )

    assert payload["currency"] == "GBP"
    assert np.isclose(payload["current_price"], 55.0)
    assert np.isclose(payload["change_5d_pct"], (55.0 / 51.0 - 1.0) * 100.0)
    assert payload["horizons"]["h1"]["max_adjustment_pct"] == 0.0
    assert payload["horizons"]["h1"]["structured_mode"] == "freeze"
    assert payload["horizons"]["h5"]["structured_sign"] == "positive"
    assert np.isclose(payload["horizons"]["h5"]["lower_adjusted_price"], 59.0 * 0.985)
    assert np.isclose(payload["horizons"]["h5"]["upper_adjusted_price"], 59.0 * 1.015)


def test_build_structured_case_retrieval_payload_returns_compact_non_leaking_matches():
    controls = {
        "current_case_summary": {"retrieval_tag": "trend20=pos|vol=normal"},
        "matched_examples": [
            {
                "date": "2025-12-01",
                "match_rank": 1,
                "match_distance": 0.42,
                "anchor_error_pct": {1: 0.0, 5: -0.6, 20: -1.2, 30: -1.4},
                "case_summary": {
                    "selection_role": "support",
                    "retrieval_tag": "trend20=pos|vol=normal",
                    "hindsight_feedback": "Base forecast was too low at h20/h30.",
                },
            }
        ],
        "horizon_guidance_summary": [
            "h5: actionable, matched examples show mean_err=-0.60% with sign_agreement=1.00; prefer a small positive adjustment within +/-0.50%.",
            "h20: actionable, matched examples show mean_err=-1.20% with sign_agreement=1.00; prefer a small positive adjustment within +/-0.75%.",
        ],
        "dynamic_freeze_horizons": [1],
        "per_horizon_max_adjustment_pct": {1: 0.0, 5: 0.5, 20: 0.75, 30: 0.8},
    }

    payload = build_structured_case_retrieval_payload(
        controls,
        key_horizons=[1, 5, 20, 30],
        requested_horizons=[5, 20, 30],
        max_examples=1,
    )

    assert payload["current_case_summary"]["retrieval_tag"] == "trend20=pos|vol=normal"
    assert payload["matched_examples"][0]["date"] == "2025-12-01"
    assert payload["matched_examples"][0]["anchor_error_pct"] == {
        "h5": -0.6,
        "h20": -1.2,
        "h30": -1.4,
    }
    assert payload["dynamic_freeze_horizons"] == ["h1"]
    assert payload["per_horizon_max_adjustment_pct"]["h20"] == 0.75
    assert any(item["horizon"] == "h5" for item in payload["horizon_guidance"])


def test_build_hdelta_delta_verification_payload_clips_and_zeroes_by_guidance():
    payload = build_hdelta_delta_verification_payload(
        {"h1": 0.2, "h5": 1.2, "h20": -0.4, "h30": 0.6},
        key_horizons=[1, 5, 20, 30],
        per_horizon_max_adjustment_pct={1: 0.0, 5: 0.5, 20: 0.8, 30: 0.5},
        structured_horizon_guidance={
            1: {"mode": "freeze", "preferred_sign": "zero", "confidence": "high", "magnitude": "zero"},
            5: {"mode": "adjust", "preferred_sign": "positive", "confidence": "medium", "magnitude": "small"},
            20: {"mode": "adjust", "preferred_sign": "positive", "confidence": "medium", "magnitude": "small"},
            30: {"mode": "adjust", "preferred_sign": "positive", "confidence": "high", "magnitude": "small"},
        },
        frozen_horizons=[1],
        config={
            "structured_enforce_sign": True,
            "structured_cap_by_guidance": False,
            "structured_coherence_guards": True,
            "structured_zero_negative_long_when_h5_positive": True,
            "structured_zero_h20_negative_when_h30_zero": True,
            "structured_zero_mixed_long_signs": True,
            "structured_prefer_positive_long_conflicts": True,
        },
    )

    assert payload["verified_adjustments"]["h1"] == 0.0
    assert payload["verified_adjustments"]["h5"] == 0.5
    assert payload["verified_adjustments"]["h20"] == 0.0
    assert payload["verified_adjustments"]["h30"] == 0.5
    assert payload["per_horizon_diagnostics"]["h5"]["status"] == "clipped_to_bound"
    assert payload["per_horizon_diagnostics"]["h20"]["status"] == "zeroed_by_guidance"


def test_build_market_microstructure_payload_parses_summary_fields():
    payload = build_market_microstructure_payload(
        {
            "is_auction_day": "last=1, mean20=0.15",
            "uk_icap_primary_secondary_spread_pct": "last=0.250, 5d=2.00%, 20d=-1.00%, vol20=0.150",
            "target_volume": "last=1250.0, mean20=1100.0",
            "y_ma_5d": "last=54.20",
        },
        current_case_summary={"retrieval_tag": "auction=1|spread=wide"},
    )

    assert payload["current_case_summary"]["retrieval_tag"] == "auction=1|spread=wide"
    assert set(payload["market_features"].keys()) == {
        "is_auction_day",
        "uk_icap_primary_secondary_spread_pct",
        "target_volume",
    }
    assert payload["market_features"]["is_auction_day"]["last"] == 1.0
    assert payload["market_features"]["is_auction_day"]["mean20"] == 0.15
    assert payload["market_features"]["uk_icap_primary_secondary_spread_pct"]["change_5d_pct"] == 2.0
    assert payload["market_features"]["uk_icap_primary_secondary_spread_pct"]["change_20d_pct"] == -1.0
    assert payload["market_features"]["uk_icap_primary_secondary_spread_pct"]["vol20"] == 0.15


def test_derive_hdelta_freeze_counterexample_returns_similar_low_error_case():
    history = np.linspace(50.0, 55.0, 30)
    forecast = np.linspace(55.0, 58.0, 30)
    teaching_examples = [
        {
            "date": "2025-12-01",
            "history": np.linspace(49.8, 54.8, 30),
            "forecast": np.linspace(54.9, 57.9, 30),
            "truth": np.linspace(54.92, 57.92, 30),
            "retrieval_tag": "recent_high_error",
            "case_summary": {
                "selection_role": "support",
                "hindsight_feedback": "Base forecast already close; avoid over-correcting long horizons.",
            },
        },
        {
            "date": "2025-11-10",
            "history": np.linspace(60.0, 57.0, 30),
            "forecast": np.linspace(58.0, 55.0, 30),
            "truth": np.linspace(56.0, 53.0, 30),
            "retrieval_tag": "recent_high_error",
            "case_summary": {
                "selection_role": "support",
                "hindsight_feedback": "Forecast was directionally wrong.",
            },
        },
        {
            "date": "2025-10-20",
            "history": np.linspace(50.5, 55.5, 30),
            "forecast": np.linspace(55.1, 58.1, 30),
            "truth": np.linspace(55.6, 58.6, 30),
            "retrieval_tag": "recent_high_error",
            "case_summary": {
                "selection_role": "support",
                "hindsight_feedback": "Needed a medium positive correction.",
            },
        },
    ]

    payload = derive_hdelta_freeze_counterexample(
        history=history,
        forecast=forecast,
        teaching_examples=teaching_examples,
        key_horizons=[1, 5, 20, 30],
        quantile=0.4,
    )

    assert payload["date"] == "2025-12-01"
    assert payload["retrieval_tag"] == "recent_high_error"
    assert "shrink unsupported adjustments toward 0.0" in payload["summary"]
    assert payload["match_distance"] >= 0.0
    assert payload["path_mse"] < 0.01
    assert "hindsight_feedback" in payload


def test_call_llm_messages_with_tools_executes_local_function_tool(monkeypatch):
    class FakeToolCall:
        def __init__(self, call_id: str, name: str, arguments: str):
            self.id = call_id
            self.type = "function"
            self.function = SimpleNamespace(name=name, arguments=arguments)

    class FakeClient:
        def __init__(self):
            self.calls = []
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                message = SimpleNamespace(
                    content="",
                    reasoning_content=None,
                    tool_calls=[FakeToolCall("call_1", NUMERIC_ANALYSIS_TOOL_NAME, '{"requested_horizons":[5,20]}')],
                )
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
                    model="stub-model",
                )
            message = SimpleNamespace(
                content='{"adjustments": {"h1": 0.0, "h5": 0.4, "h20": 0.6, "h30": 0.0}}',
                reasoning_content=None,
                tool_calls=None,
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason="stop")],
                model="stub-model",
            )

    fake_client = FakeClient()

    refiner = LLMRefiner({"provider": "openai", "model": "stub-model", "api_key": "deo", "base_url": "http://stub.local/v1"})
    monkeypatch.setattr(refiner, "_get_client", lambda **kwargs: fake_client)

    tool_payload = {"ok": True, "horizons": {"h5": {}, "h20": {}}}
    seen_calls = []

    def tool_executor(name, arguments):
        seen_calls.append((name, arguments))
        return tool_payload

    content = refiner._call_llm_messages_with_tools(
        [{"role": "user", "content": "Use the numeric tool before answering."}],
        tools=[{"type": "function", "function": {"name": NUMERIC_ANALYSIS_TOOL_NAME, "parameters": {"type": "object"}}}],
        tool_executor=tool_executor,
        system_message="Return JSON only.",
        tool_choice="required",
    )

    assert '"adjustments"' in content
    assert seen_calls == [(NUMERIC_ANALYSIS_TOOL_NAME, {"requested_horizons": [5, 20]})]
    assert refiner._last_llm_response["tool_calls"] == [NUMERIC_ANALYSIS_TOOL_NAME]
    assert refiner._last_llm_response["tool_invocations"] == 1
    second_messages = fake_client.calls[1]["messages"]
    assert any(msg.get("role") == "tool" for msg in second_messages)


def test_call_llm_messages_with_tools_can_require_multiple_tools(monkeypatch):
    class FakeToolCall:
        def __init__(self, call_id: str, name: str, arguments: str):
            self.id = call_id
            self.type = "function"
            self.function = SimpleNamespace(name=name, arguments=arguments)

    class FakeClient:
        def __init__(self):
            self.calls = []
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                message = SimpleNamespace(
                    content="",
                    reasoning_content=None,
                    tool_calls=[FakeToolCall("call_1", CASE_RETRIEVAL_TOOL_NAME, '{"max_examples":2}')],
                )
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
                    model="stub-model",
                )
            if len(self.calls) == 2:
                message = SimpleNamespace(
                    content='{"adjustments":{"h1":0.0}}',
                    reasoning_content=None,
                    tool_calls=None,
                )
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=message, finish_reason="stop")],
                    model="stub-model",
                )
            message = SimpleNamespace(
                content="",
                reasoning_content=None,
                tool_calls=[FakeToolCall("call_2", DELTA_VERIFIER_TOOL_NAME, '{"proposed_adjustments":{"h1":0.0}}')],
            )
            if len(self.calls) == 3:
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
                    model="stub-model",
                )
            message = SimpleNamespace(
                content='{"adjustments": {"h1": 0.0, "h5": 0.1, "h20": 0.2, "h30": 0.2}}',
                reasoning_content=None,
                tool_calls=None,
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason="stop")],
                model="stub-model",
            )

    fake_client = FakeClient()
    refiner = LLMRefiner({"provider": "openai", "model": "stub-model", "api_key": "deo", "base_url": "http://stub.local/v1"})
    monkeypatch.setattr(refiner, "_get_client", lambda **kwargs: fake_client)

    tool_results = []

    def tool_executor(name, arguments):
        tool_results.append((name, arguments))
        if name == CASE_RETRIEVAL_TOOL_NAME:
            return {"matched_examples": []}
        if name == DELTA_VERIFIER_TOOL_NAME:
            return {"verified_adjustments": {"h1": 0.0, "h5": 0.1, "h20": 0.2, "h30": 0.2}}
        raise AssertionError(f"unexpected tool {name}")

    content = refiner._call_llm_messages_with_tools(
        [{"role": "user", "content": "Use all tools before the final answer."}],
        tools=[
            {"type": "function", "function": {"name": CASE_RETRIEVAL_TOOL_NAME, "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": DELTA_VERIFIER_TOOL_NAME, "parameters": {"type": "object"}}},
        ],
        tool_executor=tool_executor,
        required_tool_names=[CASE_RETRIEVAL_TOOL_NAME, DELTA_VERIFIER_TOOL_NAME],
        tool_choice="required",
        max_tool_rounds=4,
    )

    assert '"adjustments"' in content
    assert [name for name, _ in tool_results] == [CASE_RETRIEVAL_TOOL_NAME, DELTA_VERIFIER_TOOL_NAME]
    assert refiner._last_llm_response["tool_calls"] == [
        CASE_RETRIEVAL_TOOL_NAME,
        DELTA_VERIFIER_TOOL_NAME,
    ]
    assert refiner._last_llm_response["missing_required_tool_calls"] == []


def test_cot_rf_hdelta_numeric_tool_routes_apply_through_tool_loop(monkeypatch):
    reflect_calls = []
    tool_calls = []

    def fake_reflect(
        self,
        prompt,
        system_message,
        response_format=None,
        model_override=None,
        api_key_override=None,
        base_url_override=None,
        **kwargs,
    ):
        reflect_calls.append(prompt)
        return (
            '{"horizons": {'
            '"h1": {"mode": "freeze", "preferred_sign": "zero", "confidence": "high", "magnitude": "zero", "reason": "Short horizon aligned."}, '
            '"h5": {"mode": "adjust", "preferred_sign": "positive", "confidence": "medium", "magnitude": "small", "reason": "Examples undershot around h5."}, '
            '"h20": {"mode": "adjust", "preferred_sign": "positive", "confidence": "medium", "magnitude": "small", "reason": "Medium horizon bias stays negative."}, '
            '"h30": {"mode": "freeze", "preferred_sign": "zero", "confidence": "medium", "magnitude": "zero", "reason": "Long horizon mixed."}'
            "}}"
        )

    def fake_tool_apply(
        self,
        messages,
        *,
        tools,
        tool_executor,
        system_message="",
        tool_choice="auto",
        **kwargs,
    ):
        tool_calls.append(
            {
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
                "system_message": system_message,
            }
        )
        payload = tool_executor(NUMERIC_ANALYSIS_TOOL_NAME, {"requested_horizons": [5, 20, 30]})
        assert "horizons" in payload
        self._last_llm_response = {
            "content": '{"adjustments": {"h1": 0.0, "h5": 0.5, "h20": 0.75, "h30": 0.0}}',
            "reasoning_content": "",
            "finish_reason": "stop",
            "response_model": "stub-tool-model",
            "used_reasoning_fallback": False,
            "tool_calls": [NUMERIC_ANALYSIS_TOOL_NAME],
            "tool_invocations": 1,
        }
        return self._last_llm_response["content"]

    monkeypatch.setattr(LLMRefiner, "_call_llm", fake_reflect)
    monkeypatch.setattr(LLMRefiner, "_call_llm_messages_with_tools", fake_tool_apply)

    refiner = LLMRefiner(
        {
            "provider": "openai",
            "model": "apply-model",
            "api_key": "deo",
            "base_url": "http://apply.local/v1",
            "cot_rf": {
                "retain_context": False,
                "strict_json_prompt": True,
                "strict_json_response_format": True,
                "structured_horizon_reflection": True,
                "enable_numeric_tool": True,
                "force_numeric_tool": True,
            },
            "hdelta": {
                "key_horizons": [1, 5, 20, 30],
                "freeze_horizons": [1],
                "max_adjustment_pct": 1.0,
            },
        }
    )

    forecast, metadata = refiner.refine(
        method="TSM+LLM-COT-RF-HDELTA",
        history=np.linspace(50.0, 55.0, 30),
        dates=[f"2026-02-{day:02d}" for day in range(1, 31)],
        tsm_forecast=np.full(30, 60.0),
        pred_len=30,
        teaching_examples=[
            {
                "history": np.linspace(48.0, 54.0, 30),
                "forecast": np.full(30, 58.0),
                "truth": np.full(30, 57.0),
                "date": "2025-12-01",
            }
        ],
    )

    assert forecast is not None
    assert np.isclose(forecast[0], 60.0)
    assert tool_calls
    assert tool_calls[0]["tool_choice"] == "required"
    assert tool_calls[0]["tools"][0]["function"]["name"] == NUMERIC_ANALYSIS_TOOL_NAME
    assert NUMERIC_ANALYSIS_TOOL_NAME in tool_calls[0]["messages"][0]["content"]
    assert metadata["numeric_tool_enabled"] is True
    assert metadata["numeric_tool_calls"] == [NUMERIC_ANALYSIS_TOOL_NAME]


def test_cot_rf_hdelta_case_retrieval_and_verifier_tools_route_apply_through_tool_loop(monkeypatch):
    tool_calls = []

    def fake_reflect(
        self,
        prompt,
        system_message,
        response_format=None,
        model_override=None,
        api_key_override=None,
        base_url_override=None,
        **kwargs,
    ):
        return (
            '{"horizons": {'
            '"h1": {"mode": "freeze", "preferred_sign": "zero", "confidence": "high", "magnitude": "zero", "reason": "Short horizon aligned."}, '
            '"h5": {"mode": "adjust", "preferred_sign": "positive", "confidence": "medium", "magnitude": "small", "reason": "Examples undershot around h5."}, '
            '"h20": {"mode": "adjust", "preferred_sign": "positive", "confidence": "medium", "magnitude": "small", "reason": "Medium horizon bias stays negative."}, '
            '"h30": {"mode": "adjust", "preferred_sign": "positive", "confidence": "medium", "magnitude": "small", "reason": "Long horizon still slightly low."}'
            "}}"
        )

    def fake_tool_apply(
        self,
        messages,
        *,
        tools,
        tool_executor,
        required_tool_names=None,
        tool_choice="auto",
        **kwargs,
    ):
        tool_calls.append(
            {
                "messages": messages,
                "tools": tools,
                "required_tool_names": required_tool_names,
                "tool_choice": tool_choice,
            }
        )
        retrieval_payload = tool_executor(CASE_RETRIEVAL_TOOL_NAME, {"requested_horizons": [5, 20, 30]})
        verifier_payload = tool_executor(
            DELTA_VERIFIER_TOOL_NAME,
            {"proposed_adjustments": {"h1": 0.0, "h5": 0.4, "h20": 0.5, "h30": 0.4}},
        )
        assert "matched_examples" in retrieval_payload
        assert "verified_adjustments" in verifier_payload
        self._last_llm_response = {
            "content": '{"adjustments": {"h1": 0.0, "h5": 0.4, "h20": 0.5, "h30": 0.4}}',
            "reasoning_content": "",
            "finish_reason": "stop",
            "response_model": "stub-tool-model",
            "used_reasoning_fallback": False,
            "tool_calls": [CASE_RETRIEVAL_TOOL_NAME, DELTA_VERIFIER_TOOL_NAME],
            "tool_invocations": 2,
            "missing_required_tool_calls": [],
        }
        return self._last_llm_response["content"]

    monkeypatch.setattr(LLMRefiner, "_call_llm", fake_reflect)
    monkeypatch.setattr(LLMRefiner, "_call_llm_messages_with_tools", fake_tool_apply)

    refiner = LLMRefiner(
        {
            "provider": "openai",
            "model": "apply-model",
            "api_key": "deo",
            "base_url": "http://apply.local/v1",
            "cot_rf": {
                "retain_context": False,
                "strict_json_prompt": True,
                "strict_json_response_format": True,
                "structured_horizon_reflection": True,
                "enable_case_retrieval_tool": True,
                "force_case_retrieval_tool": True,
                "enable_delta_verifier_tool": True,
                "force_delta_verifier_tool": True,
            },
            "hdelta": {
                "key_horizons": [1, 5, 20, 30],
                "freeze_horizons": [1],
                "max_adjustment_pct": 1.0,
            },
        }
    )

    forecast, metadata = refiner.refine(
        method="TSM+LLM-COT-RF-HDELTA",
        history=np.linspace(50.0, 55.0, 30),
        dates=[f"2026-02-{day:02d}" for day in range(1, 31)],
        tsm_forecast=np.full(30, 60.0),
        pred_len=30,
        teaching_examples=[
            {
                "history": np.linspace(48.0, 54.0, 30),
                "forecast": np.full(30, 58.0),
                "truth": np.full(30, 57.0),
                "date": "2025-12-01",
            }
        ],
    )

    assert forecast is not None
    assert tool_calls
    assert tool_calls[0]["tool_choice"] == "required"
    assert tool_calls[0]["required_tool_names"] == [
        CASE_RETRIEVAL_TOOL_NAME,
        DELTA_VERIFIER_TOOL_NAME,
    ]
    assert metadata["case_retrieval_tool_enabled"] is True
    assert metadata["delta_verifier_tool_enabled"] is True
    assert metadata["case_retrieval_tool_invocations"] == 1
    assert metadata["delta_verifier_tool_invocations"] == 1


def test_cot_rf_hdelta_market_counterexample_and_verifier_tools_route_apply_through_tool_loop(monkeypatch):
    tool_calls = []

    def fake_reflect(
        self,
        prompt,
        system_message,
        response_format=None,
        model_override=None,
        api_key_override=None,
        base_url_override=None,
        **kwargs,
    ):
        return (
            '{"horizons": {'
            '"h1": {"mode": "freeze", "preferred_sign": "zero", "confidence": "high", "magnitude": "zero", "reason": "Short horizon aligned."}, '
            '"h5": {"mode": "adjust", "preferred_sign": "positive", "confidence": "medium", "magnitude": "small", "reason": "Auction-day pressure can lift h5."}, '
            '"h20": {"mode": "adjust", "preferred_sign": "positive", "confidence": "medium", "magnitude": "small", "reason": "Mid horizon still slightly low."}, '
            '"h30": {"mode": "adjust", "preferred_sign": "positive", "confidence": "medium", "magnitude": "small", "reason": "Long horizon still slightly low."}'
            "}}"
        )

    def fake_tool_apply(
        self,
        messages,
        *,
        tools,
        tool_executor,
        required_tool_names=None,
        tool_choice="auto",
        **kwargs,
    ):
        tool_calls.append(
            {
                "messages": messages,
                "tools": tools,
                "required_tool_names": required_tool_names,
                "tool_choice": tool_choice,
            }
        )
        market_payload = tool_executor(MARKET_MICROSTRUCTURE_TOOL_NAME, {"requested_features": ["is_auction_day"]})
        counterexample_payload = tool_executor(COUNTEREXAMPLE_TOOL_NAME, {})
        verifier_payload = tool_executor(
            DELTA_VERIFIER_TOOL_NAME,
            {"proposed_adjustments": {"h1": 0.0, "h5": 0.3, "h20": 0.5, "h30": 0.4}},
        )
        assert "market_features" in market_payload
        assert "summary" in counterexample_payload
        assert "verified_adjustments" in verifier_payload
        self._last_llm_response = {
            "content": '{"adjustments": {"h1": 0.0, "h5": 0.3, "h20": 0.5, "h30": 0.4}}',
            "reasoning_content": "",
            "finish_reason": "stop",
            "response_model": "stub-tool-model",
            "used_reasoning_fallback": False,
            "tool_calls": [
                MARKET_MICROSTRUCTURE_TOOL_NAME,
                COUNTEREXAMPLE_TOOL_NAME,
                DELTA_VERIFIER_TOOL_NAME,
            ],
            "tool_invocations": 3,
            "missing_required_tool_calls": [],
        }
        return self._last_llm_response["content"]

    monkeypatch.setattr(LLMRefiner, "_call_llm", fake_reflect)
    monkeypatch.setattr(LLMRefiner, "_call_llm_messages_with_tools", fake_tool_apply)

    refiner = LLMRefiner(
        {
            "provider": "openai",
            "model": "apply-model",
            "api_key": "deo",
            "base_url": "http://apply.local/v1",
            "cot_rf": {
                "retain_context": False,
                "strict_json_prompt": True,
                "strict_json_response_format": True,
                "structured_horizon_reflection": True,
                "enable_market_microstructure_tool": True,
                "force_market_microstructure_tool": True,
                "enable_freeze_counterexample_tool": True,
                "force_freeze_counterexample_tool": True,
                "enable_delta_verifier_tool": True,
                "force_delta_verifier_tool": True,
            },
            "hdelta": {
                "key_horizons": [1, 5, 20, 30],
                "freeze_horizons": [1],
                "max_adjustment_pct": 1.0,
            },
        }
    )

    forecast, metadata = refiner.refine(
        method="TSM+LLM-COT-RF-HDELTA",
        history=np.linspace(50.0, 55.0, 30),
        dates=[f"2026-02-{day:02d}" for day in range(1, 31)],
        tsm_forecast=np.full(30, 60.0),
        pred_len=30,
        exogenous_summary={
            "is_auction_day": "last=1, mean20=0.10",
            "uk_icap_primary_secondary_spread_pct": "last=0.25, 5d=1.50%, 20d=0.50%, vol20=0.10",
            "target_range_pct": "last=1.20, mean20=0.90",
        },
        teaching_examples=[
            {
                "history": np.linspace(48.0, 54.0, 30),
                "forecast": np.full(30, 58.0),
                "truth": np.full(30, 57.8),
                "date": "2025-12-01",
                "retrieval_tag": "recent_high_error",
                "case_summary": {
                    "selection_role": "support",
                    "hindsight_feedback": "Base forecast already close; shrink unsupported long-horizon changes.",
                },
            },
            {
                "history": np.linspace(47.5, 53.5, 30),
                "forecast": np.full(30, 57.5),
                "truth": np.full(30, 57.45),
                "date": "2025-11-18",
                "retrieval_tag": "recent_high_error",
                "case_summary": {
                    "selection_role": "support",
                    "hindsight_feedback": "Minimal correction needed.",
                },
            },
        ],
    )

    assert forecast is not None
    assert tool_calls
    assert tool_calls[0]["tool_choice"] == "required"
    assert tool_calls[0]["required_tool_names"] == [
        MARKET_MICROSTRUCTURE_TOOL_NAME,
        COUNTEREXAMPLE_TOOL_NAME,
        DELTA_VERIFIER_TOOL_NAME,
    ]
    assert metadata["market_microstructure_tool_enabled"] is True
    assert metadata["freeze_counterexample_tool_enabled"] is True
    assert metadata["delta_verifier_tool_enabled"] is True
    assert metadata["market_microstructure_tool_invocations"] == 1
    assert metadata["freeze_counterexample_tool_invocations"] == 1
    assert metadata["delta_verifier_tool_invocations"] == 1


def test_refine_batch_can_resume_from_checkpoint(tmp_path, monkeypatch):
    calls = []

    def fake_refine(
        self,
        method,
        history,
        dates,
        tsm_forecast=None,
        pred_len=30,
        exogenous_summary=None,
        price_base=None,
        teaching_examples=None,
        sentiment_history=None,
    ):
        calls.append(dates[-1])
        value = float(len(calls))
        return np.full(pred_len, value, dtype=float), {
            "success": True,
            "window_end": dates[-1],
        }

    monkeypatch.setattr(LLMRefiner, "refine", fake_refine)

    refiner = LLMRefiner({"provider": "openai", "model": "stub"})
    histories = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=float,
    )
    dates = [
        ["2026-03-01", "2026-03-02", "2026-03-03"],
        ["2026-03-04", "2026-03-05", "2026-03-06"],
    ]
    tsm = np.ones((2, 4), dtype=float)

    forecasts_first, meta_first = refiner.refine_batch(
        method="TSM+LLM-COT-RF-HDELTA",
        histories=histories,
        date_arrays=dates,
        tsm_forecasts=tsm,
        pred_len=4,
        checkpoint_dir=tmp_path / "ckpt_a",
        sample_keys=["200", "201"],
    )

    assert len(calls) == 2
    assert (tmp_path / "ckpt_a" / "200.json").exists()
    assert (tmp_path / "ckpt_a" / "201.json").exists()
    assert all(not meta.get("resumed_from_checkpoint", False) for meta in meta_first)

    forecasts_second, meta_second = refiner.refine_batch(
        method="TSM+LLM-COT-RF-HDELTA",
        histories=histories,
        date_arrays=dates,
        tsm_forecasts=tsm,
        pred_len=4,
        checkpoint_dir=tmp_path / "ckpt_b",
        resume_checkpoint_dir=tmp_path / "ckpt_a",
        sample_keys=["200", "201"],
    )

    assert len(calls) == 2
    assert np.allclose(forecasts_first, forecasts_second)
    assert all(meta.get("resumed_from_checkpoint", False) for meta in meta_second)
    assert (tmp_path / "ckpt_b" / "200.json").exists()
    assert (tmp_path / "ckpt_b" / "201.json").exists()


def test_parse_hdelta_reflection_guidance_from_reasoning_trace():
    parsed = parse_hdelta_reflection_guidance(
        """
Thinking Process:
1. Infer the regime.
2. Keep h1 frozen.
3. Return strict JSON.

Final output:
{"horizons": {
  "h1": {"mode": "freeze", "preferred_sign": "zero", "confidence": "high", "magnitude": "zero", "reason": "Short horizon aligned."},
  "h5": {"mode": "adjust", "preferred_sign": "positive", "confidence": "medium", "magnitude": "small", "reason": "Examples undershot at h5."},
  "h20": {"mode": "adjust", "preferred_sign": "positive", "confidence": "medium", "magnitude": "small", "reason": "Examples stay mildly under-biased at h20."},
  "h30": {"mode": "freeze", "preferred_sign": "zero", "confidence": "medium", "magnitude": "zero", "reason": "Long-horizon evidence is mixed."}
}}
""",
        key_horizons=[1, 5, 20, 30],
    )

    assert parsed is not None
    assert parsed[1]["mode"] == "freeze"
    assert parsed[5]["preferred_sign"] == "positive"


def test_parse_hdelta_reflection_guidance_from_reasoning_bullets_without_json():
    parsed = parse_hdelta_reflection_guidance(
        """
Thinking Process:
* h1: adjust. positive. medium confidence. small magnitude. Reason: preliminary idea.
* h1: freeze. zero. high confidence. zero magnitude. Reason: Recent volatility dominates the weak short-horizon bias.
* h5: adjust. positive. medium confidence. small magnitude. Reason: Structural underestimation bias appears around h5.
* h20: adjust. positive. high confidence. medium magnitude. Reason: Historical examples show consistent forecast undershoot at h20.
* h30: adjust. positive. high confidence. medium magnitude. Reason: Historical examples show consistent forecast undershoot at h30.
""",
        key_horizons=[1, 5, 20, 30],
    )

    assert parsed is not None
    assert parsed[1]["mode"] == "freeze"
    assert parsed[1]["confidence"] == "high"
    assert parsed[20]["magnitude"] == "medium"


def test_parse_horizon_deltas_from_reasoning_trace():
    parsed = parse_horizon_deltas(
        """
Thinking Process:
Use small, bounded corrections.
Final answer:
{"adjustments": {"h1": 0.0, "h5": 0.2, "h20": 0.5, "h30": 0.4}}
"""
    )

    assert parsed == {1: 0.0, 5: 0.2, 20: 0.5, 30: 0.4}


def test_parse_horizon_deltas_prefers_tail_values_over_quoted_example_json():
    parsed = parse_horizon_deltas(
        """
Thinking Process:
Example format:
{"adjustments": {"h1": 0.3, "h5": -0.5, "h20": 0.8, "h30": -0.2}}

Chosen values:
h1: 0.0
h5: 0.5
h20: 0.6
h30: 0.6
"""
    )

    assert parsed == {1: 0.0, 5: 0.5, 20: 0.6, 30: 0.6}


def test_response_format_policy_can_disable_specific_reasoning_endpoint():
    refiner = LLMRefiner(
        {
            "provider": "openai",
            "model": "qwen3.5-35b-a3b-ud-q4-k-xl",
            "base_url": "http://192.168.1.140:9881/v1",
            "disable_response_format_base_urls": ["9881"],
        }
    )

    allowed = refiner._response_format_allowed(
        {"type": "json_schema"},
        model_override="qwen3.5-35b-a3b-ud-q4-k-xl",
        base_url_override="http://192.168.1.140:9881/v1",
    )

    assert allowed is None


def test_aggregate_structured_hdelta_guidance_freezes_conflicted_horizon():
    guidance = aggregate_structured_hdelta_guidance(
        [
            {
                1: {"mode": "freeze", "preferred_sign": "zero", "confidence": "high", "magnitude": "zero", "reason": "Aligned."},
                5: {"mode": "adjust", "preferred_sign": "positive", "confidence": "high", "magnitude": "small", "reason": "Example 1 undershot."},
                20: {"mode": "adjust", "preferred_sign": "positive", "confidence": "medium", "magnitude": "small", "reason": "Example 1 and 2 undershot."},
                30: {"mode": "adjust", "preferred_sign": "positive", "confidence": "medium", "magnitude": "small", "reason": "Example 1 supports a lift."},
            },
            {
                1: {"mode": "freeze", "preferred_sign": "zero", "confidence": "high", "magnitude": "zero", "reason": "Aligned."},
                5: {"mode": "adjust", "preferred_sign": "positive", "confidence": "medium", "magnitude": "tiny", "reason": "Example 2 undershot."},
                20: {"mode": "adjust", "preferred_sign": "positive", "confidence": "medium", "magnitude": "tiny", "reason": "Example 2 undershot."},
                30: {"mode": "freeze", "preferred_sign": "zero", "confidence": "low", "magnitude": "zero", "reason": "Long horizon mixed."},
            },
            {
                1: {"mode": "freeze", "preferred_sign": "zero", "confidence": "high", "magnitude": "zero", "reason": "Aligned."},
                5: {"mode": "adjust", "preferred_sign": "positive", "confidence": "medium", "magnitude": "tiny", "reason": "Example 3 undershot."},
                20: {"mode": "adjust", "preferred_sign": "positive", "confidence": "medium", "magnitude": "tiny", "reason": "Example 3 undershot."},
                30: {"mode": "adjust", "preferred_sign": "negative", "confidence": "medium", "magnitude": "small", "reason": "Example 3 overshot long."},
            },
        ],
        key_horizons=[1, 5, 20, 30],
        mode="conservative_majority",
    )

    assert guidance is not None
    assert guidance[5]["mode"] == "adjust"
    assert guidance[5]["preferred_sign"] == "positive"
    assert guidance[5]["confidence"] == "medium"
    assert guidance[30]["mode"] == "freeze"
    assert guidance[30]["preferred_sign"] == "zero"


def test_cot_rf_hdelta_reflect_samples_aggregate_guidance(monkeypatch):
    calls = []

    reflect_payloads = [
        '{"horizons": {"h1": {"mode": "freeze", "preferred_sign": "zero", "confidence": "high", "magnitude": "zero", "reason": "Aligned."}, "h5": {"mode": "adjust", "preferred_sign": "positive", "confidence": "high", "magnitude": "small", "reason": "Example 1 undershot."}, "h20": {"mode": "adjust", "preferred_sign": "positive", "confidence": "medium", "magnitude": "small", "reason": "Example 1 undershot."}, "h30": {"mode": "adjust", "preferred_sign": "positive", "confidence": "medium", "magnitude": "small", "reason": "Example 1 supports a lift."}}}',
        '{"horizons": {"h1": {"mode": "freeze", "preferred_sign": "zero", "confidence": "high", "magnitude": "zero", "reason": "Aligned."}, "h5": {"mode": "adjust", "preferred_sign": "positive", "confidence": "medium", "magnitude": "tiny", "reason": "Example 2 undershot."}, "h20": {"mode": "adjust", "preferred_sign": "positive", "confidence": "medium", "magnitude": "tiny", "reason": "Example 2 undershot."}, "h30": {"mode": "freeze", "preferred_sign": "zero", "confidence": "low", "magnitude": "zero", "reason": "Long horizon mixed."}}}',
        '{"horizons": {"h1": {"mode": "freeze", "preferred_sign": "zero", "confidence": "high", "magnitude": "zero", "reason": "Aligned."}, "h5": {"mode": "adjust", "preferred_sign": "positive", "confidence": "medium", "magnitude": "tiny", "reason": "Example 3 undershot."}, "h20": {"mode": "adjust", "preferred_sign": "positive", "confidence": "medium", "magnitude": "tiny", "reason": "Example 3 undershot."}, "h30": {"mode": "adjust", "preferred_sign": "negative", "confidence": "medium", "magnitude": "small", "reason": "Example 3 overshot long."}}}',
    ]

    def fake_call(
        self,
        prompt,
        system_message,
        response_format=None,
        model_override=None,
        api_key_override=None,
        base_url_override=None,
        temperature_override=None,
        cache_key_suffix=None,
        **kwargs,
    ):
        calls.append({"prompt": prompt, "temperature_override": temperature_override, "cache_key_suffix": cache_key_suffix})
        if len(calls) <= 3:
            return reflect_payloads[len(calls) - 1]
        return '{"adjustments": {"h1": 0.0, "h5": 0.4, "h20": 0.3, "h30": 0.2}}'

    monkeypatch.setattr(LLMRefiner, "_call_llm", fake_call)

    refiner = LLMRefiner(
        {
            "provider": "openai",
            "model": "apply-model",
            "api_key": "deo",
            "base_url": "http://apply.local/v1",
            "cot_rf": {
                "retain_context": False,
                "strict_json_prompt": True,
                "strict_json_response_format": True,
                "structured_horizon_reflection": True,
                "reflect_samples": 3,
                "reflect_aggregation": "conservative_majority",
                "reflect_temperature": 0.2,
            },
            "hdelta": {
                "key_horizons": [1, 5, 20, 30],
                "freeze_horizons": [1],
                "max_adjustment_pct": 1.0,
            },
        }
    )

    forecast, metadata = refiner.refine(
        method="TSM+LLM-COT-RF-HDELTA",
        history=np.linspace(50.0, 55.0, 30),
        dates=[f"2026-02-{day:02d}" for day in range(1, 31)],
        tsm_forecast=np.full(30, 60.0),
        pred_len=30,
        teaching_examples=[
            {
                "history": np.linspace(48.0, 54.0, 30),
                "forecast": np.full(30, 58.0),
                "truth": np.full(30, 57.0),
                "date": "2025-12-01",
            }
        ],
    )

    assert forecast is not None
    assert metadata["reflect_samples"] == 3
    assert metadata["reflect_valid_samples"] == 3
    assert metadata["structured_horizon_guidance"][30]["mode"] == "freeze"
    assert calls[0]["temperature_override"] == 0.2
    assert calls[1]["cache_key_suffix"] == "reflect_sample_1"
    assert np.isclose(forecast[29], 60.0)


def test_select_structured_reflect_sample_budget_routes_high_uncertainty_cases():
    samples, routed, diagnostics = select_structured_reflect_sample_budget(
        cot_rf_cfg={
            "reflect_samples": 1,
            "reflect_uncertainty_routing": True,
            "reflect_samples_high_uncertainty": 3,
            "reflect_samples_low_uncertainty": 1,
            "reflect_uncertainty_distance_threshold": 1.2,
            "reflect_uncertainty_min_dynamic_freezes": 2,
        },
        hdelta_case_controls={
            "dynamic_freeze_horizons": [1, 20, 30],
            "per_horizon_max_adjustment_pct": {1: 0.0, 5: 0.15, 20: 0.0, 30: 0.0},
            "matched_examples": [
                {"match_distance": 1.6},
                {"match_distance": 1.4},
                {"match_distance": 1.5},
            ],
        },
        frozen_horizons=[1],
    )

    assert samples == 3
    assert routed is True
    assert diagnostics["dynamic_freeze_count"] == 2.0
    assert diagnostics["mean_match_distance"] > 1.2


def test_select_structured_reflect_sample_budget_uses_sign_disagreement():
    samples, routed, diagnostics = select_structured_reflect_sample_budget(
        cot_rf_cfg={
            "reflect_samples": 1,
            "reflect_uncertainty_routing": True,
            "reflect_samples_high_uncertainty": 3,
            "reflect_samples_low_uncertainty": 1,
            "reflect_uncertainty_distance_threshold": 9.0,
            "reflect_uncertainty_distance_std_threshold": 9.0,
            "reflect_uncertainty_min_sign_agreement": 0.75,
            "reflect_uncertainty_sign_horizons": [20, 30],
        },
        hdelta_case_controls={
            "dynamic_freeze_horizons": [],
            "per_horizon_max_adjustment_pct": {1: 0.0, 5: 0.30, 20: 0.30, 30: 0.30},
            "matched_examples": [
                {"match_distance": 0.5, "anchor_error_pct": {20: 1.0, 30: 1.0}},
                {"match_distance": 0.6, "anchor_error_pct": {20: -1.0, 30: 1.0}},
                {"match_distance": 0.4, "anchor_error_pct": {20: 1.0, 30: -1.0}},
            ],
        },
        frozen_horizons=[1],
    )

    assert samples == 3
    assert routed is True
    assert diagnostics["min_sign_agreement"] < 0.75


def test_derive_hdelta_case_controls_adds_retrieval_tags_when_enabled():
    controls = derive_hdelta_case_controls(
        history=np.linspace(50.0, 55.0, 30),
        forecast=np.full(30, 60.0),
        teaching_examples=[
            {
                "history": np.linspace(48.0, 54.0, 30),
                "forecast": np.full(30, 58.0),
                "truth": np.full(30, 57.0),
                "date": "2025-12-01",
            }
        ],
        key_horizons=[1, 5, 20, 30],
        config={
            "case_match_top_k": 1,
            "case_min_examples": 1,
            "case_min_sign_agreement": 0.0,
            "case_min_mean_abs_error_pct": 0.0,
            "include_retrieval_tag": True,
        },
        currency="GBP",
    )

    assert "retrieval_tag" in controls["current_case_summary"]
    assert controls["matched_examples"][0]["case_summary"]["retrieval_tag"] == build_hdelta_retrieval_tag(
        np.linspace(48.0, 54.0, 30),
        np.full(30, 58.0),
    )


def test_cot_rf_hdelta_structured_reflection_routes_guidance_into_apply(monkeypatch):
    calls = []

    def fake_call(
        self,
        prompt,
        system_message,
        response_format=None,
        model_override=None,
        api_key_override=None,
        base_url_override=None,
        **kwargs,
    ):
        calls.append(
            {
                "prompt": prompt,
                "response_format": response_format,
            }
        )
        if len(calls) == 1:
            return (
                '{"horizons": {'
                '"h1": {"mode": "freeze", "preferred_sign": "zero", "confidence": "high", "magnitude": "zero", "reason": "Short horizon already aligned."}, '
                '"h5": {"mode": "adjust", "preferred_sign": "positive", "confidence": "high", "magnitude": "small", "reason": "Matched examples undershot around h5."}, '
                '"h20": {"mode": "adjust", "preferred_sign": "positive", "confidence": "medium", "magnitude": "small", "reason": "Mid-horizon bias stays slightly negative."}, '
                '"h30": {"mode": "freeze", "preferred_sign": "zero", "confidence": "medium", "magnitude": "zero", "reason": "Long-horizon evidence is mixed."}'
                "}}"
            )
        return '{"adjustments": {"h1": 0.7, "h5": 0.5, "h20": 0.4, "h30": 0.1}}'

    monkeypatch.setattr(LLMRefiner, "_call_llm", fake_call)

    refiner = LLMRefiner(
        {
            "provider": "openai",
            "model": "apply-model",
            "cot_rf": {
                "strict_json_prompt": True,
                "strict_json_response_format": True,
                "structured_horizon_reflection": True,
            },
            "hdelta": {
                "key_horizons": [1, 5, 20, 30],
                "freeze_horizons": [1],
                "max_adjustment_pct": 1.0,
                "case_conditioned": True,
                "case_match_top_k": 1,
                "case_min_examples": 1,
                "case_min_sign_agreement": 0.0,
                "case_min_mean_abs_error_pct": 0.0,
            },
        }
    )

    forecast, metadata = refiner.refine(
        method="TSM+LLM-COT-RF-HDELTA",
        history=np.linspace(50.0, 55.0, 30),
        dates=[f"2026-02-{day:02d}" for day in range(1, 31)],
        tsm_forecast=np.full(30, 60.0),
        pred_len=30,
        teaching_examples=[
            {
                "history": np.linspace(48.0, 54.0, 30),
                "forecast": np.full(30, 58.0),
                "truth": np.full(30, 57.0),
                "date": "2025-12-01",
            }
        ],
    )

    assert forecast is not None
    assert calls[0]["response_format"]["json_schema"]["name"] == "cot_sent_rf_hdelta_structured_reflection"
    assert "Structured Horizon Decisions" in calls[1]["prompt"]
    assert "mode=adjust; sign=positive" in calls[1]["prompt"]
    assert metadata["structured_horizon_reflection"] is True
    assert metadata["structured_horizon_guidance"][5]["preferred_sign"] == "positive"


def test_derive_hdelta_case_controls_supports_horizon_overrides():
    history = np.linspace(50.0, 55.0, 30)
    forecast = np.full(30, 60.0)
    teaching_examples = [
        {
            "history": np.linspace(48.0, 54.0, 30),
            "forecast": np.full(30, 58.0),
            "truth": np.array([57.7] * 30),
            "date": "2025-12-01",
        },
        {
            "history": np.linspace(48.2, 54.2, 30),
            "forecast": np.full(30, 58.0),
            "truth": np.array([57.7] * 30),
            "date": "2025-12-02",
        },
    ]

    controls = derive_hdelta_case_controls(
        history=history,
        forecast=forecast,
        teaching_examples=teaching_examples,
        key_horizons=[1, 5, 20, 30],
        config={
            "max_adjustment_pct": 0.8,
            "case_match_top_k": 2,
            "case_min_examples": 2,
            "case_min_sign_agreement": 0.67,
            "case_min_mean_abs_error_pct": 0.05,
            "case_bound_scale": 0.75,
            "case_min_bound_pct": 0.10,
            "horizon_overrides": {
                "h30": {
                    "force_freeze": True,
                }
            },
        },
        currency="GBP",
    )

    assert 30 in controls["dynamic_freeze_horizons"]
    assert controls["per_horizon_max_adjustment_pct"][30] == 0.0
    assert any("forced by horizon-specific configuration" in line for line in controls["horizon_guidance_summary"])


def test_derive_hdelta_case_controls_supports_horizon_specific_matching():
    history = np.linspace(50.0, 55.0, 30)
    forecast = np.linspace(56.0, 60.0, 30)
    teaching_examples = [
        {
            "history": np.linspace(49.5, 54.5, 30),
            "forecast": np.linspace(55.8, 59.8, 30),
            "truth": np.linspace(55.6, 61.2, 30),
            "date": "2025-12-01",
        },
        {
            "history": np.linspace(42.0, 44.0, 30),
            "forecast": np.linspace(45.0, 48.0, 30),
            "truth": np.linspace(45.2, 47.7, 30),
            "date": "2025-11-15",
        },
        {
            "history": np.linspace(50.2, 55.1, 30),
            "forecast": np.linspace(56.2, 59.9, 30),
            "truth": np.linspace(56.4, 61.6, 30),
            "date": "2025-12-03",
        },
    ]

    controls = derive_hdelta_case_controls(
        history=history,
        forecast=forecast,
        teaching_examples=teaching_examples,
        key_horizons=[1, 5, 20, 30],
        config={
            "max_adjustment_pct": 1.0,
            "case_match_top_k": 1,
            "horizon_specific_matching": True,
            "horizon_match_top_k": 1,
            "horizon_match_union_limit": 3,
            "case_min_examples": 1,
            "case_min_sign_agreement": 0.0,
            "case_min_mean_abs_error_pct": 0.0,
        },
        currency="GBP",
    )

    assert len(controls["matched_dates"]) >= 1
    assert any(line.startswith("h1 matches:") for line in controls["matched_examples_summary"])
    assert any(line.startswith("h30 matches:") for line in controls["matched_examples_summary"])
    assert any("horizon-specific matches" in line for line in controls["horizon_guidance_summary"])


def test_enforce_structured_hdelta_adjustments_applies_sign_confidence_and_scale():
    adjusted = enforce_structured_hdelta_adjustments(
        adjustments_pct={1: -0.1, 5: 0.9, 20: 0.4, 30: 0.6},
        structured_guidance={
            1: {"mode": "freeze", "preferred_sign": "zero", "confidence": "high", "magnitude": "zero", "reason": "freeze"},
            5: {"mode": "adjust", "preferred_sign": "positive", "confidence": "high", "magnitude": "small", "reason": "positive h5"},
            20: {"mode": "adjust", "preferred_sign": "positive", "confidence": "low", "magnitude": "small", "reason": "weak h20"},
            30: {"mode": "adjust", "preferred_sign": "negative", "confidence": "medium", "magnitude": "tiny", "reason": "tiny negative h30"},
        },
        per_horizon_max={1: 0.5, 5: 1.0, 20: 1.0, 30: 1.0},
        config={
            "structured_enforce_sign": True,
            "structured_cap_by_guidance": True,
            "structured_min_confidence_by_horizon": {"h20": "medium"},
            "structured_magnitude_scale": {"tiny": 0.2, "small": 0.5},
            "structured_confidence_scale": {"medium": 0.8, "high": 1.0},
        },
    )

    assert adjusted[1] == 0.0
    assert 0.0 <= adjusted[5] <= 0.5
    assert adjusted[20] == 0.0
    assert adjusted[30] == 0.0


def test_apply_structured_hdelta_coherence_guards_targets_conflicts():
    guarded = apply_structured_hdelta_coherence_guards(
        adjustments_pct={5: 0.2, 20: -0.3, 30: 0.1},
        config={
            "structured_coherence_guards": True,
            "structured_zero_negative_long_when_h5_positive": True,
            "structured_zero_h20_negative_when_h30_zero": True,
            "structured_zero_mixed_long_signs": True,
            "structured_prefer_positive_long_conflicts": True,
        },
    )

    assert guarded[20] == 0.0
    assert guarded[30] == 0.1


def test_apply_discrete_hdelta_actions_snaps_to_signed_menu():
    snapped = apply_discrete_hdelta_actions(
        adjustments_pct={5: 0.19, 20: -0.41, 30: 0.0},
        structured_guidance={
            5: {"confidence": "medium"},
            20: {"confidence": "high"},
            30: {"confidence": "high"},
        },
        per_horizon_max={5: 1.0, 20: 1.0, 30: 1.0},
        config={
            "discrete_actions_enabled": True,
            "discrete_action_fractions_by_horizon": {
                "h5": [0.0, 0.2],
                "h20": [0.0, 0.25, 0.5],
                "h30": [0.0, 0.2, 0.4],
            },
            "discrete_activation_fraction_by_horizon": {"h5": 0.1, "h20": 0.1, "h30": 0.1},
        },
    )

    assert snapped[5] == 0.2
    assert snapped[20] == -0.5
    assert snapped[30] == 0.0


def test_apply_discrete_hdelta_actions_respects_confidence_floor():
    snapped = apply_discrete_hdelta_actions(
        adjustments_pct={20: 0.4},
        structured_guidance={20: {"confidence": "low"}},
        per_horizon_max={20: 1.0},
        config={
            "discrete_actions_enabled": True,
            "discrete_action_fractions": [0.0, 0.25, 0.5],
            "discrete_activation_fraction_by_horizon": {"h20": 0.1},
            "discrete_min_confidence_by_horizon": {"h20": "medium"},
        },
    )

    assert snapped[20] == 0.0


def test_aggregate_horizon_adjustments_median():
    aggregated = aggregate_horizon_adjustments(
        [
            {1: 0.0, 5: 0.2, 20: 0.3, 30: 0.1},
            {1: 0.0, 5: 0.4, 20: 0.1, 30: 0.0},
            {1: 0.0, 5: 0.6, 20: 0.2, 30: -0.1},
        ],
        key_horizons=[1, 5, 20, 30],
        mode="median",
    )
    assert aggregated == {1: 0.0, 5: 0.4, 20: 0.2, 30: 0.0}


def test_cot_rf_hdelta_supports_apply_self_consistency(monkeypatch):
    calls = []

    def fake_call(
        self,
        prompt,
        system_message,
        response_format=None,
        model_override=None,
        api_key_override=None,
        base_url_override=None,
        temperature_override=None,
        cache_key_suffix=None,
        **kwargs,
    ):
        calls.append(
            {
                "prompt": prompt,
                "temperature_override": temperature_override,
                "cache_key_suffix": cache_key_suffix,
            }
        )
        if len(calls) == 1:
            return (
                '{"horizons": {'
                '"h1": {"mode": "freeze", "preferred_sign": "zero", "confidence": "high", "magnitude": "zero", "reason": "aligned"}, '
                '"h5": {"mode": "adjust", "preferred_sign": "positive", "confidence": "high", "magnitude": "small", "reason": "undershoot"}, '
                '"h20": {"mode": "adjust", "preferred_sign": "positive", "confidence": "medium", "magnitude": "small", "reason": "mild lag"}, '
                '"h30": {"mode": "freeze", "preferred_sign": "zero", "confidence": "medium", "magnitude": "zero", "reason": "mixed"}'
                "}}"
            )
        responses = [
            '{"adjustments": {"h1": 0.0, "h5": 0.2, "h20": 0.3, "h30": 0.0}}',
            '{"adjustments": {"h1": 0.0, "h5": 0.4, "h20": 0.1, "h30": 0.0}}',
            '{"adjustments": {"h1": 0.0, "h5": 0.6, "h20": 0.2, "h30": 0.0}}',
        ]
        return responses[len(calls) - 2]

    monkeypatch.setattr(LLMRefiner, "_call_llm", fake_call)

    refiner = LLMRefiner(
        {
            "provider": "openai",
            "model": "apply-model",
            "temperature": 0.0,
            "cot_rf": {
                "strict_json_prompt": True,
                "strict_json_response_format": True,
                "structured_horizon_reflection": True,
                "apply_samples": 3,
                "apply_aggregation": "median",
                "apply_temperature": 0.2,
            },
            "hdelta": {
                "key_horizons": [1, 5, 20, 30],
                "freeze_horizons": [1],
                "max_adjustment_pct": 1.0,
                "case_conditioned": True,
                "case_match_top_k": 1,
                "case_min_examples": 1,
                "case_min_sign_agreement": 0.0,
                "case_min_mean_abs_error_pct": 0.0,
            },
        }
    )

    forecast, metadata = refiner.refine(
        method="TSM+LLM-COT-RF-HDELTA",
        history=np.linspace(50.0, 55.0, 30),
        dates=[f"2026-02-{day:02d}" for day in range(1, 31)],
        tsm_forecast=np.full(30, 60.0),
        pred_len=30,
        teaching_examples=[
            {
                "history": np.linspace(48.0, 54.0, 30),
                "forecast": np.full(30, 58.0),
                "truth": np.full(30, 57.0),
                "date": "2025-12-01",
            }
        ],
    )

    assert forecast is not None
    assert np.isclose(forecast[0], 60.0)
    assert np.isclose(forecast[4], 60.0 * 1.004)
    assert np.isclose(forecast[19], 60.0 * 1.002)
    assert metadata["apply_samples"] == 3
    assert metadata["apply_aggregation"] == "median"
    assert calls[1]["temperature_override"] == 0.2
    assert calls[1]["cache_key_suffix"] == "apply_sample_0"

    guarded = apply_structured_hdelta_coherence_guards(
        adjustments_pct={5: 0.0, 20: -0.2, 30: 0.0},
        config={
            "structured_coherence_guards": True,
            "structured_zero_h20_negative_when_h30_zero": True,
        },
    )

    assert guarded[20] == 0.0
