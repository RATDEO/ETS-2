"""Strict JSON event extraction for official article bodies."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI

try:
    from llm.cache import LogStore, ResponseCache  # type: ignore
except ModuleNotFoundError:
    from src.llm.cache import LogStore, ResponseCache  # type: ignore

logger = logging.getLogger(__name__)

SYSTEM_MESSAGE = (
    "You extract structured market events for EU ETS forecasting. "
    "Return valid JSON only."
)

EVENT_TYPES = [
    "policy_rule_change",
    "auction_supply",
    "verified_emissions",
    "maritime_ets",
    "aviation_ets",
    "cbam",
    "power_gas_coal_system",
    "industrial_demand",
    "macro_risk",
    "geopolitics",
    "weather",
    "other",
]

CHANNELS = [
    "allowance_supply",
    "compliance_demand",
    "power_generation",
    "fuel_switching",
    "industrial_output",
    "macro",
    "other",
]


def build_event_extraction_prompt(title: str, summary: str, body_text: str) -> str:
    return (
        "Extract one structured EU ETS event from the text below.\n"
        "Return JSON with exactly these keys:\n"
        "{\n"
        '  "is_relevant": true|false,\n'
        f'  "event_type": one of {EVENT_TYPES},\n'
        f'  "affected_channel": one of {CHANNELS},\n'
        '  "direction": "bullish"|"bearish"|"mixed"|"neutral"|"unclear",\n'
        '  "intensity": 0|1|2|3,\n'
        '  "expected_horizon": "1_5d"|"5_20d"|"20_60d"|"60d_plus"|"unclear",\n'
        '  "novelty": "new"|"continuation"|"repeat"|"unclear",\n'
        '  "policy_stage": "none"|"consultation"|"proposal"|"vote"|"adoption"|"implementation"|"reporting"|"auction_schedule"|"unclear",\n'
        '  "confidence": 0.0 to 1.0,\n'
        '  "summary": "<<= 40 words>"\n'
        "}\n"
        "Use the article body, not generic climate opinion. Mark is_relevant=false if not plausibly useful for EU ETS price formation.\n"
        f"TITLE: {title}\n"
        f"SUMMARY: {summary}\n"
        f"BODY: {body_text[:4000]}\n"
    )


def _coerce_event_payload(payload: dict[str, Any]) -> dict[str, Any]:
    event_type = str(payload.get("event_type", "other"))
    if event_type not in EVENT_TYPES:
        event_type = "other"
    channel = str(payload.get("affected_channel", "other"))
    if channel not in CHANNELS:
        channel = "other"
    direction = str(payload.get("direction", "unclear"))
    if direction not in {"bullish", "bearish", "mixed", "neutral", "unclear", "scheduled"}:
        direction = "unclear"
    novelty = str(payload.get("novelty", "unclear"))
    if novelty not in {"new", "continuation", "repeat", "unclear"}:
        novelty = "unclear"
    expected_horizon = str(payload.get("expected_horizon", "unclear"))
    if expected_horizon not in {"1_5d", "5_20d", "20_60d", "60d_plus", "unclear"}:
        expected_horizon = "unclear"
    policy_stage = str(payload.get("policy_stage", "unclear"))
    if policy_stage not in {"none", "consultation", "proposal", "vote", "adoption", "implementation", "reporting", "auction_schedule", "unclear"}:
        policy_stage = "unclear"
    try:
        intensity = int(round(float(payload.get("intensity", 0))))
    except Exception:
        intensity = 0
    intensity = max(0, min(3, intensity))
    try:
        confidence = float(payload.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return {
        "is_relevant": bool(payload.get("is_relevant", False)),
        "event_type": event_type,
        "affected_channel": channel,
        "direction": direction,
        "intensity": intensity,
        "expected_horizon": expected_horizon,
        "novelty": novelty,
        "policy_stage": policy_stage,
        "confidence": confidence,
        "summary": str(payload.get("summary", "")).strip()[:280],
    }


def _extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return _coerce_event_payload({})
    try:
        return _coerce_event_payload(json.loads(text))
    except Exception:
        match_start = text.find("{")
        match_end = text.rfind("}")
        if match_start >= 0 and match_end > match_start:
            try:
                return _coerce_event_payload(json.loads(text[match_start : match_end + 1]))
            except Exception:
                pass
    return _coerce_event_payload({})


def extract_event_batch(
    rows: pd.DataFrame,
    model: str,
    base_url: str,
    api_key: str,
    cache_dir: str | Path,
    log_dir: str | Path,
    temperature: float = 0.0,
    max_tokens: int = 300,
    timeout_seconds: float = 60.0,
) -> pd.DataFrame:
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)
    cache = ResponseCache(cache_dir)
    logs = LogStore(log_dir)
    outputs: list[dict[str, Any]] = []

    for row in rows.to_dict(orient="records"):
        prompt = build_event_extraction_prompt(
            title=str(row.get("title", "")),
            summary=str(row.get("summary", "")),
            body_text=str(row.get("body_text", "")),
        )
        context = {"url": str(row.get("url", ""))}
        cached = cache.get(prompt, model=model, temperature=temperature, context=context)
        if cached is None:
            request = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            response = client.chat.completions.create(**request)
            content = response.choices[0].message.content
            if isinstance(content, list):
                text = "\n".join(str(getattr(item, "text", "") or item.get("text", "")) for item in content if item)
            else:
                text = str(content or "")
            cached = {"content": text}
            cache.set(prompt, model=model, temperature=temperature, response=cached, context=context)
        payload = _extract_json(str(cached.get("content", "")))
        logs.log_call(prompt, payload, model=model, temperature=temperature, method="official_event_extract", metadata=context)
        outputs.append({**row, **payload})
    return pd.DataFrame(outputs)
