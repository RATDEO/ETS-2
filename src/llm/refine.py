"""
LLM refinement for time series forecasts.

Implements the LLM-based forecast refinement layer that takes
TSM predictions and historical context to produce refined forecasts.
"""

import numpy as np
import pandas as pd
import json
import re
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from pathlib import Path
import logging
from collections import Counter

from .prompts import get_template, PromptTemplate
from .cache import ResponseCache, LogStore

logger = logging.getLogger(__name__)

INTERNAL_BLEND_METHODS = frozenset(
    {
        "TSM+LLM-COT-SENT-RF-DELTA",
        "TSM+LLM-NORM-DELTA",
    }
)

HDELTA_KEY_HORIZONS = (1, 5, 20, 30)
HPRICE_KEY_HORIZONS = (1, 10, 20, 30)
NUMERIC_ANALYSIS_TOOL_NAME = "get_numeric_analysis"
CASE_RETRIEVAL_TOOL_NAME = "get_structured_case_retrieval"
DELTA_VERIFIER_TOOL_NAME = "verify_hdelta_adjustments"
MARKET_MICROSTRUCTURE_TOOL_NAME = "get_market_microstructure_state"
COUNTEREXAMPLE_TOOL_NAME = "get_freeze_counterexample"


def blend_mode_from_config(config: Optional[dict]) -> str:
    """Return the configured LLM internal-blend mode."""
    cfg = config or {}
    blend_cfg = cfg.get("blend", {}) or {}
    return str(blend_cfg.get("mode", "horizon_weighted")).lower()


def method_supports_internal_blend(method: str) -> bool:
    """Whether a method actually calls the refiner's internal blend path."""
    return method in INTERNAL_BLEND_METHODS


def method_uses_internal_blend(method: str, config: Optional[dict]) -> bool:
    """Whether a method will apply internal blending for the given config."""
    if not method_supports_internal_blend(method):
        return False
    return blend_mode_from_config(config) != "none"


def _extract_openai_message_text(message: object) -> str:
    """
    Extract only assistant final output text from an OpenAI chat message.

    This intentionally ignores any reasoning-specific fields.
    """
    content = getattr(message, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                text = item
            elif isinstance(item, dict):
                text = item.get("text") or ""
            else:
                text = getattr(item, "text", "") or ""
            text = str(text).strip()
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    return str(content).strip()


def _extract_openai_reasoning_text(message: object) -> str:
    """Extract reasoning text from OpenAI-compatible reasoning models."""
    content = getattr(message, "reasoning_content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                text = item
            elif isinstance(item, dict):
                text = item.get("text") or ""
            else:
                text = getattr(item, "text", "") or ""
            text = str(text).strip()
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    return str(content).strip()


def _strip_code_fences(text: str) -> str:
    """Remove a single leading/trailing markdown code fence when present."""
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", cleaned)
        cleaned = re.sub(r"\n```$", "", cleaned)
    return cleaned.strip()


def _extract_balanced_json_snippets(
    text: str,
    open_char: str = "{",
    close_char: str = "}",
) -> List[str]:
    """Extract balanced JSON object or array substrings from noisy model text."""
    raw = str(text or "")
    snippets: List[str] = []
    start: Optional[int] = None
    depth = 0
    in_string = False
    escaped = False

    for idx, char in enumerate(raw):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == open_char:
            if depth == 0:
                start = idx
            depth += 1
            continue
        if char == close_char and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                snippet = raw[start : idx + 1].strip()
                if snippet:
                    snippets.append(snippet)
                start = None
    return snippets


def _json_parse_candidates(
    response_text: str,
    allow_arrays: bool = False,
) -> List[str]:
    """Generate plausible JSON payload candidates from a noisy LLM response."""
    cleaned = _strip_code_fences(response_text)
    candidates: List[str] = []
    seen = set()

    def _add(candidate: str) -> None:
        text = str(candidate or "").strip()
        if not text or text in seen:
            return
        seen.add(text)
        candidates.append(text)

    _add(cleaned)
    for snippet in reversed(_extract_balanced_json_snippets(cleaned, "{", "}")):
        _add(snippet)
    if allow_arrays:
        for snippet in reversed(_extract_balanced_json_snippets(cleaned, "[", "]")):
            _add(snippet)
    return candidates


def parse_json_array(
    response_text: str,
    expected_len: int = 30,
    keys: Sequence[str] = ("yhat",),
    allow_bare_array: bool = True,
) -> Optional[np.ndarray]:
    """
    Parse a numeric array from LLM response JSON.
    
    Args:
        response_text: Raw LLM response
        expected_len: Expected length of array
        keys: JSON keys to search for
        allow_bare_array: Whether to accept a top-level JSON array response
        
    Returns:
        Numpy array or None if parsing fails
    """
    # Normalize common formatting artifacts
    cleaned = _strip_code_fences(response_text)
    cleaned = cleaned.replace("...", "")
    cleaned = re.sub(r",\s*([\]}])", r"\1", cleaned)

    # Fast path: full-response JSON
    try:
        direct = json.loads(cleaned)
        if isinstance(direct, dict):
            for key in keys:
                values = direct.get(key)
                if isinstance(values, list):
                    if len(values) == expected_len:
                        return np.array(values, dtype=float)
                    if len(values) == expected_len - 1:
                        values = list(values) + [values[-1]]
                        return np.array(values, dtype=float)
        if allow_bare_array and isinstance(direct, list):
            if len(direct) == expected_len:
                return np.array(direct, dtype=float)
            if len(direct) == expected_len - 1:
                values = list(direct) + [direct[-1]]
                return np.array(values, dtype=float)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Try to extract keyed JSON snippets from response
    for text in _json_parse_candidates(cleaned, allow_arrays=allow_bare_array):
        try:
            if text.startswith("{"):
                data = json.loads(re.sub(r",\s*([\]}])", r"\1", text))
                if isinstance(data, dict):
                    for key in keys:
                        values = data.get(key)
                        if isinstance(values, list):
                            if len(values) == expected_len:
                                return np.array(values, dtype=float)
                            if len(values) == expected_len - 1:
                                values = list(values) + [values[-1]]
                                return np.array(values, dtype=float)
            if allow_bare_array and text.startswith("["):
                values = json.loads(re.sub(r",\s*([\]}])", r"\1", text))
                if isinstance(values, list):
                    if len(values) == expected_len:
                        return np.array(values, dtype=float)
                    if len(values) == expected_len - 1:
                        values = list(values) + [values[-1]]
                        return np.array(values, dtype=float)

            for key in keys:
                if key in text:
                    numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', text)
                    if len(numbers) >= expected_len:
                        values = [float(n) for n in numbers[:expected_len]]
                        return np.array(values, dtype=float)
                    if len(numbers) == expected_len - 1:
                        values = [float(n) for n in numbers]
                        values.append(values[-1])
                        return np.array(values, dtype=float)
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.debug(f"JSON parse attempt failed: {e}")
            continue
    
    logger.warning(f"Failed to parse forecast from response: {response_text[:200]}...")
    return None


def parse_json_forecast(response_text: str, expected_len: int = 30) -> Optional[np.ndarray]:
    """Parse JSON forecast (yhat) from LLM response."""
    return parse_json_array(
        response_text,
        expected_len,
        keys=("yhat",),
        allow_bare_array=False,
    )


def parse_reflection_rules(response_text: str) -> Optional[str]:
    """
    Parse correction rules from a strict-JSON reflection response.

    Expected shape: {"rules": ["...", "..."]} (or {"rules_text": "..."}).
    Returns a plain-text rules block suitable for downstream apply prompts.
    """
    cleaned = _strip_code_fences(response_text)
    if not cleaned:
        return None

    for cand in _json_parse_candidates(cleaned):
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(obj, dict):
            continue
        rules = obj.get("rules")
        if isinstance(rules, list):
            lines = [str(x).strip() for x in rules if str(x).strip()]
            if lines:
                return "\n".join(f"- {line}" for line in lines)
        rules_text = obj.get("rules_text")
        if isinstance(rules_text, str) and rules_text.strip():
            return rules_text.strip()

    return None


def parse_hdelta_reflection_guidance(
    response_text: str,
    key_horizons: Sequence[int] = HDELTA_KEY_HORIZONS,
) -> Optional[Dict[int, Dict[str, str]]]:
    """
    Parse structured per-horizon guidance from reflection JSON.

    Expected shape:
    {
      "horizons": {
        "h1": {
          "mode": "freeze"|"adjust",
          "preferred_sign": "positive"|"negative"|"zero",
          "confidence": "low"|"medium"|"high",
          "magnitude": "zero"|"tiny"|"small"|"medium",
          "reason": "..."
        },
        ...
      }
    }
    """
    cleaned = _strip_code_fences(response_text)
    if not cleaned:
        return None

    expected = [int(h) for h in key_horizons]
    horizon_keys = {f"h{int(h)}": int(h) for h in expected}
    allowed_mode = {"freeze", "adjust"}
    allowed_sign = {"positive", "negative", "zero"}
    allowed_confidence = {"low", "medium", "high"}
    allowed_magnitude = {"zero", "tiny", "small", "medium"}

    for cand in _json_parse_candidates(cleaned):
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(obj, dict):
            continue
        raw_horizons = obj.get("horizons")
        if not isinstance(raw_horizons, dict):
            continue

        parsed: Dict[int, Dict[str, str]] = {}
        ok = True
        for raw_key, horizon in horizon_keys.items():
            payload = raw_horizons.get(raw_key)
            if not isinstance(payload, dict):
                ok = False
                break
            mode = str(payload.get("mode", "")).strip().lower()
            preferred_sign = str(payload.get("preferred_sign", "")).strip().lower()
            confidence = str(payload.get("confidence", "")).strip().lower()
            magnitude = str(payload.get("magnitude", "")).strip().lower()
            reason = str(payload.get("reason", "")).strip()
            if (
                mode not in allowed_mode
                or preferred_sign not in allowed_sign
                or confidence not in allowed_confidence
                or magnitude not in allowed_magnitude
                or not reason
            ):
                ok = False
                break
            parsed[int(horizon)] = {
                "mode": mode,
                "preferred_sign": preferred_sign,
                "confidence": confidence,
                "magnitude": magnitude,
                "reason": reason,
            }
        if ok and all(int(h) in parsed for h in expected):
            return {int(h): parsed[int(h)] for h in expected}

    line_pattern = re.compile(r"h(?P<h>\d+)\s*:\s*(?P<body>.+)$", re.IGNORECASE)
    parsed_lines: Dict[int, Dict[str, str]] = {}
    for raw_line in cleaned.splitlines():
        line = str(raw_line or "").strip()
        match = line_pattern.search(line)
        if not match:
            continue
        horizon = int(match.group("h"))
        if horizon not in expected:
            continue
        body = match.group("body").strip()
        mode_match = re.search(r"\b(freeze|adjust)\b", body, re.IGNORECASE)
        if not mode_match:
            continue
        mode = mode_match.group(1).lower()
        sign_match = re.search(r"\b(positive|negative|zero)\b", body, re.IGNORECASE)
        confidence_match = re.search(r"\b(low|medium|high)\b(?:\s+confidence)?", body, re.IGNORECASE)
        magnitude_match = re.search(r"\b(zero|tiny|small|medium)\b(?:\s+magnitude)?", body, re.IGNORECASE)
        reason_match = re.search(r"\bReason\s*:\s*(.+)$", body, re.IGNORECASE)
        reason = ""
        if reason_match:
            reason = reason_match.group(1).strip(" .")
        elif "(" in body and ")" in body:
            paren_match = re.search(r"\((.+?)\)", body)
            if paren_match:
                reason = paren_match.group(1).strip(" .")

        preferred_sign = sign_match.group(1).lower() if sign_match else ("zero" if mode == "freeze" else "")
        confidence = confidence_match.group(1).lower() if confidence_match else ""
        magnitude = magnitude_match.group(1).lower() if magnitude_match else ("zero" if mode == "freeze" else "")
        if (
            mode in allowed_mode
            and preferred_sign in allowed_sign
            and confidence in allowed_confidence
            and magnitude in allowed_magnitude
            and reason
        ):
            parsed_lines[horizon] = {
                "mode": mode,
                "preferred_sign": preferred_sign,
                "confidence": confidence,
                "magnitude": magnitude,
                "reason": reason,
            }

    if all(int(h) in parsed_lines for h in expected):
        return {int(h): parsed_lines[int(h)] for h in expected}

    conf_defaults = re.findall(
        r"`?confidence`?\s*[:=]\s*(low|medium|high)",
        cleaned,
        flags=re.IGNORECASE,
    )
    mag_defaults = re.findall(
        r"`?magnitude`?\s*[:=]\s*(zero|tiny|small|medium)",
        cleaned,
        flags=re.IGNORECASE,
    )
    default_confidence = conf_defaults[-1].lower() if conf_defaults else "medium"
    default_magnitude = mag_defaults[-1].lower() if mag_defaults else "small"
    narrative_guidance: Dict[int, Dict[str, str]] = {}

    for horizon in expected:
        matches = list(re.finditer(rf"h{int(horizon)}\s*:", cleaned, flags=re.IGNORECASE))
        if not matches:
            continue
        start = matches[-1].start()
        next_positions = [
            m.start()
            for h2 in expected
            if int(h2) != int(horizon)
            for m in re.finditer(rf"h{int(h2)}\s*:", cleaned[start + 1 :], flags=re.IGNORECASE)
        ]
        end = start + min(next_positions) if next_positions else min(len(cleaned), start + 900)
        segment = cleaned[start:end].strip()
        mode_candidates = re.findall(r"\b(freeze|adjust)\b", segment, flags=re.IGNORECASE)
        sign_candidates = re.findall(r"\b(positive|negative|zero)\b", segment, flags=re.IGNORECASE)
        conf_candidates = re.findall(
            r"\b(low|medium|high)\b(?:\s+confidence)?",
            segment,
            flags=re.IGNORECASE,
        )
        mag_candidates = re.findall(
            r"\b(zero|tiny|small|medium)\b(?:\s+magnitude)?",
            segment,
            flags=re.IGNORECASE,
        )
        mode = mode_candidates[-1].lower() if mode_candidates else ""
        preferred_sign = sign_candidates[-1].lower() if sign_candidates else ""
        confidence = conf_candidates[-1].lower() if conf_candidates else default_confidence
        magnitude = mag_candidates[-1].lower() if mag_candidates else default_magnitude

        ambiguous = "mixed" in segment.lower() or (
            "freeze" in segment.lower() and "adjust" in segment.lower()
        )
        if ambiguous and int(horizon) == expected[0]:
            mode = "freeze"
            preferred_sign = "zero"
            confidence = "low"
            magnitude = "zero"

        if mode == "freeze":
            preferred_sign = "zero"
            magnitude = "zero"
        if not preferred_sign and mode == "adjust":
            preferred_sign = "positive" if "undershoot" in segment.lower() else ""
        if (
            mode not in allowed_mode
            or preferred_sign not in allowed_sign
            or confidence not in allowed_confidence
            or magnitude not in allowed_magnitude
        ):
            continue

        if mode == "freeze":
            reason = "Evidence is mixed at this horizon; keep the base forecast near unchanged."
        elif preferred_sign == "positive":
            reason = "Matched examples suggest the base forecast undershoots at this horizon."
        else:
            reason = "Matched examples suggest the base forecast overshoots at this horizon."

        narrative_guidance[int(horizon)] = {
            "mode": mode,
            "preferred_sign": preferred_sign,
            "confidence": confidence,
            "magnitude": magnitude,
            "reason": reason,
        }

    if all(int(h) in narrative_guidance for h in expected):
        return {int(h): narrative_guidance[int(h)] for h in expected}
    return None


def _downgrade_confidence(value: str) -> str:
    order = ["low", "medium", "high"]
    text = str(value or "").strip().lower()
    try:
        idx = order.index(text)
    except ValueError:
        return "low"
    return order[max(0, idx - 1)]


def aggregate_structured_hdelta_guidance(
    guidance_samples: Sequence[Dict[int, Dict[str, str]]],
    key_horizons: Sequence[int] = HDELTA_KEY_HORIZONS,
    mode: str = "conservative_majority",
) -> Optional[Dict[int, Dict[str, str]]]:
    """Aggregate multiple structured reflection samples into one conservative guidance object."""
    samples = [sample for sample in guidance_samples if sample]
    if not samples:
        return None
    if len(samples) == 1:
        return {int(h): dict(v) for h, v in samples[0].items()}

    agg_mode = str(mode or "conservative_majority").strip().lower()
    expected = [int(h) for h in key_horizons]
    majority_needed = max(1, int(math.floor(len(samples) / 2.0) + 1))
    confidence_order = ["low", "medium", "high"]
    magnitude_order = ["zero", "tiny", "small", "medium"]

    aggregated: Dict[int, Dict[str, str]] = {}
    for horizon in expected:
        payloads = [sample.get(int(horizon)) or {} for sample in samples]
        modes = [str(payload.get("mode", "")).strip().lower() for payload in payloads]
        signs = [str(payload.get("preferred_sign", "")).strip().lower() for payload in payloads]
        confidences = [str(payload.get("confidence", "")).strip().lower() for payload in payloads]
        magnitudes = [str(payload.get("magnitude", "")).strip().lower() for payload in payloads]
        reasons = [str(payload.get("reason", "")).strip() for payload in payloads]

        mode_counts = Counter(m for m in modes if m)
        sign_counts = Counter(s for s in signs if s)
        conf_counts = Counter(c for c in confidences if c)
        mag_counts = Counter(m for m in magnitudes if m)

        dominant_mode, dominant_mode_count = (mode_counts.most_common(1) or [("freeze", 0)])[0]
        dominant_sign, dominant_sign_count = (sign_counts.most_common(1) or [("zero", 0)])[0]

        conflict = (
            dominant_mode_count < majority_needed
            or dominant_sign_count < majority_needed
            or dominant_mode not in {"freeze", "adjust"}
            or dominant_sign not in {"positive", "negative", "zero"}
        )

        if agg_mode == "majority":
            conflict = (
                dominant_mode not in {"freeze", "adjust"}
                or dominant_sign not in {"positive", "negative", "zero"}
            )

        if conflict:
            aggregated[int(horizon)] = {
                "mode": "freeze",
                "preferred_sign": "zero",
                "confidence": "low",
                "magnitude": "zero",
                "reason": "Reflective candidates disagreed; keep near the base forecast.",
            }
            continue

        agreeing = [
            payload
            for payload in payloads
            if str(payload.get("mode", "")).strip().lower() == dominant_mode
            and str(payload.get("preferred_sign", "")).strip().lower() == dominant_sign
        ]

        dominant_conf = (
            conf_counts.most_common(1)[0][0]
            if conf_counts
            else "low"
        )
        dominant_mag = (
            mag_counts.most_common(1)[0][0]
            if mag_counts
            else "zero"
        )

        if agg_mode == "conservative_majority" and dominant_mode_count != len(samples):
            dominant_conf = _downgrade_confidence(dominant_conf)

        if agg_mode == "conservative_majority" and dominant_mode == "adjust":
            agreeing_conf = [
                str(payload.get("confidence", "")).strip().lower()
                for payload in agreeing
                if str(payload.get("confidence", "")).strip()
            ]
            if agreeing_conf:
                dominant_conf = min(
                    agreeing_conf,
                    key=lambda item: confidence_order.index(item)
                    if item in confidence_order
                    else 0,
                )
            agreeing_mag = [
                str(payload.get("magnitude", "")).strip().lower()
                for payload in agreeing
                if str(payload.get("magnitude", "")).strip()
            ]
            if agreeing_mag:
                dominant_mag = min(
                    agreeing_mag,
                    key=lambda item: magnitude_order.index(item)
                    if item in magnitude_order
                    else 0,
                )

        if dominant_mode == "freeze" or dominant_sign == "zero":
            dominant_sign = "zero"
            dominant_mag = "zero"

        chosen_reason = next((reason for reason in reasons if reason), "Aggregated reflective guidance.")
        if agg_mode == "conservative_majority" and dominant_mode_count != len(samples):
            chosen_reason = f"{chosen_reason} Consensus was partial, so the move stays conservative."

        aggregated[int(horizon)] = {
            "mode": dominant_mode,
            "preferred_sign": dominant_sign,
            "confidence": dominant_conf if dominant_conf in confidence_order else "low",
            "magnitude": dominant_mag if dominant_mag in magnitude_order else "zero",
            "reason": chosen_reason,
        }

    if all(int(h) in aggregated for h in expected):
        return {int(h): aggregated[int(h)] for h in expected}
    return None


def select_structured_reflect_sample_budget(
    cot_rf_cfg: Optional[Dict],
    hdelta_case_controls: Optional[Dict],
    frozen_horizons: Optional[Sequence[int]] = None,
) -> Tuple[int, bool, Dict[str, float]]:
    """Choose the reflection sample budget, optionally routing extra deliberation to uncertain cases."""
    cfg = cot_rf_cfg or {}
    base_samples = max(1, int(cfg.get("reflect_samples", 1)))
    if not bool(cfg.get("reflect_uncertainty_routing", False)):
        return base_samples, False, {}

    high_samples = max(base_samples, int(cfg.get("reflect_samples_high_uncertainty", base_samples)))
    low_samples = max(1, int(cfg.get("reflect_samples_low_uncertainty", 1)))
    case_controls = hdelta_case_controls or {}
    frozen = {int(h) for h in (frozen_horizons or [])}
    dynamic_frozen = {
        int(h)
        for h in (case_controls.get("dynamic_freeze_horizons") or [])
        if int(h) not in frozen
    }
    per_horizon_max = {
        int(h): float(v)
        for h, v in (case_controls.get("per_horizon_max_adjustment_pct") or {}).items()
    }
    matched = list(case_controls.get("matched_examples") or [])
    distances = [
        float(ex.get("match_distance"))
        for ex in matched
        if ex.get("match_distance") is not None
    ]
    mean_distance = float(np.mean(distances)) if distances else 0.0
    distance_std = float(np.std(distances)) if distances else 0.0
    low_bound_threshold = float(cfg.get("reflect_uncertainty_low_bound_threshold", 0.20))
    low_bound_count = sum(
        1
        for horizon, bound in per_horizon_max.items()
        if int(horizon) not in frozen and float(bound) <= low_bound_threshold
    )
    min_dynamic_freezes = int(cfg.get("reflect_uncertainty_min_dynamic_freezes", 2))
    min_low_bound_horizons = int(cfg.get("reflect_uncertainty_min_low_bound_horizons", 2))
    distance_threshold = float(cfg.get("reflect_uncertainty_distance_threshold", 1.50))
    distance_std_threshold = float(cfg.get("reflect_uncertainty_distance_std_threshold", 0.45))

    sign_horizons = tuple(int(h) for h in cfg.get("reflect_uncertainty_sign_horizons", [5, 20, 30]))
    min_sign_agreement = float(cfg.get("reflect_uncertainty_min_sign_agreement", 0.60))
    sign_agreement_scores: List[float] = []
    for horizon in sign_horizons:
        signs = []
        for ex in matched:
            err = ((ex.get("anchor_error_pct") or {}).get(int(horizon)))
            if err is None:
                continue
            err_val = float(err)
            if abs(err_val) < 1e-8:
                signs.append(0.0)
            else:
                signs.append(1.0 if err_val > 0.0 else -1.0)
        non_zero = [sign for sign in signs if abs(sign) > 0.0]
        if not non_zero:
            continue
        sign_agreement_scores.append(abs(float(np.mean(non_zero))))
    min_observed_sign_agreement = (
        float(min(sign_agreement_scores)) if sign_agreement_scores else 1.0
    )

    high_uncertainty = (
        len(dynamic_frozen) >= min_dynamic_freezes
        or low_bound_count >= min_low_bound_horizons
        or mean_distance >= distance_threshold
        or distance_std >= distance_std_threshold
        or min_observed_sign_agreement < min_sign_agreement
    )
    diagnostics = {
        "mean_match_distance": mean_distance,
        "std_match_distance": distance_std,
        "dynamic_freeze_count": float(len(dynamic_frozen)),
        "low_bound_count": float(low_bound_count),
        "min_sign_agreement": min_observed_sign_agreement,
    }
    return (high_samples if high_uncertainty else low_samples), high_uncertainty, diagnostics


def hdelta_guidance_to_rules_text(
    structured_guidance: Optional[Dict[int, Dict[str, str]]],
) -> Optional[str]:
    """Convert structured horizon guidance into a compact rules block."""
    if not structured_guidance:
        return None
    lines = []
    for horizon in sorted(int(h) for h in structured_guidance.keys()):
        payload = structured_guidance[int(horizon)]
        mode = str(payload.get("mode", "")).strip().lower()
        preferred_sign = str(payload.get("preferred_sign", "")).strip().lower()
        confidence = str(payload.get("confidence", "")).strip().lower()
        magnitude = str(payload.get("magnitude", "")).strip().lower()
        reason = str(payload.get("reason", "")).strip()
        if mode == "freeze" or preferred_sign == "zero" or magnitude == "zero":
            lines.append(
                f"- h{int(horizon)}: keep unchanged; confidence={confidence}; reason={reason}"
            )
        else:
            lines.append(
                f"- h{int(horizon)}: {preferred_sign} adjustment; confidence={confidence}; "
                f"magnitude={magnitude}; reason={reason}"
            )
    if not lines:
        return None
    return "\n".join(lines)


def parse_horizon_deltas(
    response_text: str,
    key_horizons: Sequence[int] = HDELTA_KEY_HORIZONS,
) -> Optional[Dict[int, float]]:
    """
    Parse percentage adjustments at key horizons from LLM JSON.

    Expected shape:
    {"adjustments": {"h1": 0.3, "h5": -0.5, ...}}
    """
    cleaned = _strip_code_fences(response_text)
    if not cleaned:
        return None

    expected = [int(h) for h in key_horizons]
    aliases = {f"h{h}": h for h in expected}
    aliases.update({str(h): h for h in expected})

    valid_candidates: List[Dict[int, float]] = []
    for cand in _json_parse_candidates(cleaned):
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(obj, dict):
            continue
        payload = obj.get("adjustments", obj)
        if not isinstance(payload, dict):
            continue
        parsed: Dict[int, float] = {}
        ok = True
        for key, horizon in aliases.items():
            if key not in payload:
                continue
            try:
                parsed[horizon] = float(payload[key])
            except (TypeError, ValueError):
                ok = False
                break
        if ok and all(h in parsed for h in expected):
            valid_candidates.append({h: parsed[h] for h in expected})

    # Reasoning traces often mention the chosen horizon deltas near the end after
    # also quoting earlier example JSON. Prefer small late-stage numeric mentions.
    tail = cleaned[-4000:] if len(cleaned) > 4000 else cleaned
    narrative_values: Dict[int, float] = {}
    for match in re.finditer(
        r"\bh(?P<h>\d+)\s*[:=]\s*(?P<value>[-+]?\d+(?:\.\d+)?)\b(?!\s*%)",
        tail,
        flags=re.IGNORECASE,
    ):
        horizon = int(match.group("h"))
        if horizon not in expected:
            continue
        value = float(match.group("value"))
        if abs(value) > 10.0:
            continue
        narrative_values[horizon] = value

    if all(h in narrative_values for h in expected):
        return {h: narrative_values[h] for h in expected}
    if valid_candidates:
        return valid_candidates[-1]
    return None


def parse_horizon_anchor_prices(
    response_text: str,
    key_horizons: Sequence[int] = HPRICE_KEY_HORIZONS,
) -> Optional[Dict[int, float]]:
    """
    Parse absolute anchor prices at key horizons from LLM JSON.

    Expected shape:
    {"anchors": {"h1": 70.5, "h10": 71.2, ...}}
    """
    cleaned = _strip_code_fences(response_text)
    if not cleaned:
        return None

    expected = [int(h) for h in key_horizons]
    aliases = {f"h{h}": h for h in expected}
    aliases.update({str(h): h for h in expected})

    for cand in _json_parse_candidates(cleaned):
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(obj, dict):
            continue
        payload = obj.get("anchors", obj)
        if not isinstance(payload, dict):
            continue
        parsed: Dict[int, float] = {}
        ok = True
        for key, horizon in aliases.items():
            if key not in payload:
                continue
            try:
                parsed[horizon] = float(payload[key])
            except (TypeError, ValueError):
                ok = False
                break
        if ok and all(h in parsed for h in expected):
            return {h: parsed[h] for h in expected}
    return None


def aggregate_horizon_adjustments(
    samples: Sequence[Dict[int, float]],
    key_horizons: Sequence[int],
    mode: str = "median",
) -> Optional[Dict[int, float]]:
    """Aggregate multiple HDELTA proposals into one per-horizon adjustment map."""
    if not samples:
        return None
    horizons = [int(h) for h in key_horizons]
    stacked = np.array(
        [
            [float(sample.get(int(h), 0.0)) for h in horizons]
            for sample in samples
        ],
        dtype=float,
    )
    agg_mode = str(mode or "median").strip().lower()
    if agg_mode == "mean":
        agg = np.mean(stacked, axis=0)
    else:
        agg = np.median(stacked, axis=0)
    return {int(h): float(v) for h, v in zip(horizons, agg, strict=False)}


def _strict_rules_response_format() -> Dict:
    """OpenAI response_format schema for reflection rules JSON."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "cot_sent_rf_rules",
            "schema": {
                "type": "object",
                "properties": {
                    "rules": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 4,
                        "maxItems": 8,
                    }
                },
                "required": ["rules"],
                "additionalProperties": False,
            },
        },
    }


def _strict_yhat_response_format(pred_len: int) -> Dict:
    """OpenAI response_format schema for forecast JSON."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "cot_sent_rf_yhat",
            "schema": {
                "type": "object",
                "properties": {
                    "yhat": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": int(pred_len),
                        "maxItems": int(pred_len),
                    }
                },
                "required": ["yhat"],
                "additionalProperties": False,
            },
        },
    }


def _strict_hdelta_response_format(key_horizons: Sequence[int]) -> Dict:
    """OpenAI response_format schema for horizon-adjustment JSON."""
    properties = {f"h{int(h)}": {"type": "number"} for h in key_horizons}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "cot_sent_rf_hdelta",
            "schema": {
                "type": "object",
                "properties": {
                    "adjustments": {
                        "type": "object",
                        "properties": properties,
                        "required": list(properties.keys()),
                        "additionalProperties": False,
                    }
                },
                "required": ["adjustments"],
                "additionalProperties": False,
            },
        },
    }


def _strict_hdelta_reflection_response_format(key_horizons: Sequence[int]) -> Dict:
    """OpenAI response_format schema for structured HDELTA reflection JSON."""
    horizon_properties = {
        f"h{int(h)}": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["freeze", "adjust"]},
                "preferred_sign": {
                    "type": "string",
                    "enum": ["positive", "negative", "zero"],
                },
                "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                "magnitude": {
                    "type": "string",
                    "enum": ["zero", "tiny", "small", "medium"],
                },
                "reason": {"type": "string"},
            },
            "required": ["mode", "preferred_sign", "confidence", "magnitude", "reason"],
            "additionalProperties": False,
        }
        for h in key_horizons
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "cot_sent_rf_hdelta_structured_reflection",
            "schema": {
                "type": "object",
                "properties": {
                    "horizons": {
                        "type": "object",
                        "properties": horizon_properties,
                        "required": list(horizon_properties.keys()),
                        "additionalProperties": False,
                    }
                },
                "required": ["horizons"],
                "additionalProperties": False,
            },
        },
    }


def _strict_hprice_response_format(key_horizons: Sequence[int]) -> Dict:
    """OpenAI response_format schema for horizon-anchor price JSON."""
    properties = {f"h{int(h)}": {"type": "number"} for h in key_horizons}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "cot_sent_rf_hprice",
            "schema": {
                "type": "object",
                "properties": {
                    "anchors": {
                        "type": "object",
                        "properties": properties,
                        "required": list(properties.keys()),
                        "additionalProperties": False,
                    }
                },
                "required": ["anchors"],
                "additionalProperties": False,
            },
        },
    }


def _should_retry_without_response_format(error_text: str) -> bool:
    """Whether an endpoint error likely means response_format is unsupported."""
    text = str(error_text or "").lower()
    return (
        "failed to parse input" in text
        or "response_format" in text
        or "json_schema" in text
    )


def interpolate_horizon_adjustments(
    pred_len: int,
    adjustments_pct: Dict[int, float],
) -> np.ndarray:
    """Linearly interpolate key-horizon percentage adjustments over the full path."""
    if pred_len <= 0:
        return np.zeros(0, dtype=float)
    key_horizons = np.array(sorted(int(h) for h in adjustments_pct.keys()), dtype=int)
    x = key_horizons - 1
    y = np.array([float(adjustments_pct[int(h)]) for h in key_horizons], dtype=float)
    return np.interp(np.arange(pred_len, dtype=float), x, y).astype(float)


def interpolate_horizon_anchor_prices(
    pred_len: int,
    anchor_prices: Dict[int, float],
) -> np.ndarray:
    """Linearly interpolate key-horizon anchor prices over the full path."""
    if pred_len <= 0:
        return np.zeros(0, dtype=float)
    key_horizons = np.array(sorted(int(h) for h in anchor_prices.keys()), dtype=int)
    x = key_horizons - 1
    y = np.array([float(anchor_prices[int(h)]) for h in key_horizons], dtype=float)
    return np.interp(np.arange(pred_len, dtype=float), x, y).astype(float)


def _safe_pct_change(current: float, previous: float) -> float:
    if abs(float(previous)) < 1e-8:
        return 0.0
    return float((float(current) / float(previous) - 1.0) * 100.0)


def build_numeric_analysis_payload(
    history: np.ndarray,
    tsm_forecast: np.ndarray,
    key_horizons: Sequence[int],
    *,
    per_horizon_max_adjustment_pct: Optional[Dict[int, float]] = None,
    structured_horizon_guidance: Optional[Dict[int, Dict[str, str]]] = None,
    current_case_summary: Optional[Dict[str, str]] = None,
    currency: str = "EUR",
) -> Dict[str, Any]:
    """Build deterministic numeric context for HDELTA sizing and verification."""
    hist = np.asarray(history, dtype=float)
    fc = np.asarray(tsm_forecast, dtype=float)
    horizons = [int(h) for h in key_horizons if 1 <= int(h) <= int(fc.size)]
    last_price = float(hist[-1]) if hist.size else 0.0
    change_5 = _safe_pct_change(float(hist[-1]), float(hist[-5])) if hist.size >= 5 else 0.0
    change_20 = _safe_pct_change(float(hist[-1]), float(hist[-20])) if hist.size >= 20 else 0.0
    vol_20_abs = float(np.std(hist[-20:])) if hist.size else 0.0
    vol_20_pct = float((vol_20_abs / last_price) * 100.0) if abs(last_price) > 1e-8 else 0.0
    forecast_change_30 = (
        _safe_pct_change(float(fc[min(29, fc.size - 1)]), last_price) if fc.size and abs(last_price) > 1e-8 else 0.0
    )

    horizon_payload: Dict[str, Dict[str, Any]] = {}
    for horizon in horizons:
        idx = int(horizon) - 1
        base_price = float(fc[idx])
        bound_pct = float((per_horizon_max_adjustment_pct or {}).get(int(horizon), 0.0))
        lower_price = base_price * (1.0 - bound_pct / 100.0)
        upper_price = base_price * (1.0 + bound_pct / 100.0)
        guidance = (structured_horizon_guidance or {}).get(int(horizon)) or {}
        horizon_payload[f"h{int(horizon)}"] = {
            "base_price": round(base_price, 6),
            "drift_vs_spot_pct": round(_safe_pct_change(base_price, last_price), 6)
            if abs(last_price) > 1e-8
            else 0.0,
            "max_adjustment_pct": round(bound_pct, 6),
            "lower_adjusted_price": round(lower_price, 6),
            "upper_adjusted_price": round(upper_price, 6),
            "structured_mode": str(guidance.get("mode", "")),
            "structured_sign": str(guidance.get("preferred_sign", "")),
            "structured_confidence": str(guidance.get("confidence", "")),
            "structured_magnitude": str(guidance.get("magnitude", "")),
        }

    payload: Dict[str, Any] = {
        "currency": currency,
        "current_price": round(last_price, 6),
        "change_5d_pct": round(change_5, 6),
        "change_20d_pct": round(change_20, 6),
        "vol_20d_abs": round(vol_20_abs, 6),
        "vol_20d_pct": round(vol_20_pct, 6),
        "base_forecast_change_h30_pct": round(forecast_change_30, 6),
        "horizons": horizon_payload,
    }
    if current_case_summary:
        payload["current_case_summary"] = {
            str(k): str(v) for k, v in current_case_summary.items()
        }
    return payload


def build_numeric_analysis_tool_spec(
    *,
    tool_name: str = NUMERIC_ANALYSIS_TOOL_NAME,
) -> Dict[str, Any]:
    """OpenAI tool schema for deterministic numeric analysis."""
    return {
        "type": "function",
        "function": {
            "name": str(tool_name),
            "description": (
                "Return exact numeric market state and horizon bounds for the current UK ETS forecast case. "
                "Use this to size bounded horizon adjustments instead of doing arithmetic in free text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "requested_horizons": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Optional subset of horizons to inspect, e.g. [1, 5, 20, 30].",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    }


def build_structured_case_retrieval_payload(
    case_controls: Optional[Dict[str, Any]],
    key_horizons: Sequence[int],
    *,
    requested_horizons: Optional[Sequence[int]] = None,
    max_examples: Optional[int] = None,
) -> Dict[str, Any]:
    """Build structured historical-case evidence for the current HDELTA sample."""
    controls = dict(case_controls or {})
    available_horizons = [int(h) for h in key_horizons]
    if requested_horizons:
        horizons = [int(h) for h in requested_horizons if int(h) in available_horizons]
    else:
        horizons = list(available_horizons)
    if not horizons:
        horizons = list(available_horizons)

    guidance_map: Dict[int, str] = {}
    for raw_line in controls.get("horizon_guidance_summary") or []:
        line = str(raw_line or "").strip()
        match = re.match(r"^h(\d+):\s*(.*)$", line)
        if not match:
            continue
        guidance_map[int(match.group(1))] = match.group(2).strip()

    bound_map = {
        int(k): float(v)
        for k, v in (controls.get("per_horizon_max_adjustment_pct") or {}).items()
        if _parse_horizon_key(k) is not None
    }
    dynamic_freeze = {
        int(h)
        for h in (controls.get("dynamic_freeze_horizons") or [])
        if int(h) in available_horizons
    }

    matched_examples = list(controls.get("matched_examples") or [])
    if max_examples is not None:
        matched_examples = matched_examples[: max(0, int(max_examples))]

    example_payloads: List[Dict[str, Any]] = []
    for example in matched_examples:
        case_summary = example.get("case_summary") or {}
        anchor_error_pct = example.get("anchor_error_pct") or {}
        filtered_anchor_error = {
            f"h{int(h)}": round(float(anchor_error_pct.get(int(h), 0.0)), 6)
            for h in horizons
            if int(h) in anchor_error_pct
        }
        example_payloads.append(
            {
                "date": str(example.get("date", "")),
                "match_rank": int(example.get("match_rank", 0) or 0),
                "match_distance": round(float(example.get("match_distance", 0.0) or 0.0), 6),
                "selection_role": str(case_summary.get("selection_role", "")),
                "retrieval_tag": str(
                    case_summary.get("retrieval_tag")
                    or example.get("retrieval_tag")
                    or ""
                ),
                "hindsight_feedback": str(case_summary.get("hindsight_feedback", "")),
                "anchor_error_pct": filtered_anchor_error,
            }
        )

    horizon_payload = []
    for horizon in horizons:
        horizon_payload.append(
            {
                "horizon": f"h{int(horizon)}",
                "bound_pct": round(float(bound_map.get(int(horizon), 0.0)), 6),
                "freeze": bool(int(horizon) in dynamic_freeze),
                "guidance": str(guidance_map.get(int(horizon), "")),
            }
        )

    current_case_summary = {
        str(k): str(v)
        for k, v in (controls.get("current_case_summary") or {}).items()
    }
    return {
        "current_case_summary": current_case_summary,
        "matched_examples": example_payloads,
        "horizon_guidance": horizon_payload,
        "dynamic_freeze_horizons": [f"h{int(h)}" for h in sorted(dynamic_freeze)],
        "per_horizon_max_adjustment_pct": {
            f"h{int(h)}": round(float(bound_map.get(int(h), 0.0)), 6)
            for h in horizons
        },
    }


def build_structured_case_retrieval_tool_spec(
    *,
    tool_name: str = CASE_RETRIEVAL_TOOL_NAME,
) -> Dict[str, Any]:
    """OpenAI tool schema for deterministic historical case retrieval."""
    return {
        "type": "function",
        "function": {
            "name": str(tool_name),
            "description": (
                "Return structured, non-leaking historical UK ETS analogue cases already selected for this sample, "
                "including past base-forecast errors, hindsight feedback, and horizon-specific guidance."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "requested_horizons": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Optional subset of horizons to inspect, e.g. [5, 20, 30].",
                    },
                    "max_examples": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 8,
                        "description": "Optional cap on the number of matched examples to return.",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    }


def _normalize_hdelta_adjustments(
    proposed_adjustments: Optional[Dict[Union[int, str], Any]],
    key_horizons: Sequence[int],
) -> Dict[int, float]:
    """Normalize a partial horizon-adjustment mapping to the configured anchors."""
    normalized = {int(h): 0.0 for h in key_horizons}
    if not isinstance(proposed_adjustments, dict):
        return normalized
    for raw_key, raw_value in proposed_adjustments.items():
        parsed = _parse_horizon_key(raw_key)
        if parsed is None or parsed not in normalized:
            continue
        try:
            normalized[int(parsed)] = float(raw_value)
        except (TypeError, ValueError):
            normalized[int(parsed)] = 0.0
    return normalized


def build_hdelta_delta_verification_payload(
    proposed_adjustments: Optional[Dict[Union[int, str], Any]],
    key_horizons: Sequence[int],
    *,
    per_horizon_max_adjustment_pct: Optional[Dict[int, float]] = None,
    structured_horizon_guidance: Optional[Dict[int, Dict[str, str]]] = None,
    frozen_horizons: Optional[Sequence[int]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Deterministically verify tentative HDELTA adjustments against the structured contract."""
    normalized = _normalize_hdelta_adjustments(proposed_adjustments, key_horizons)
    bounds = {
        int(h): float((per_horizon_max_adjustment_pct or {}).get(int(h), abs(float(normalized[int(h)]))))
        for h in key_horizons
    }
    clipped = {
        int(h): float(np.clip(normalized[int(h)], -bounds[int(h)], bounds[int(h)]))
        for h in key_horizons
    }
    enforced = enforce_structured_hdelta_adjustments(
        clipped,
        structured_guidance=structured_horizon_guidance,
        per_horizon_max=bounds,
        config=config,
    )
    verified = apply_structured_hdelta_coherence_guards(
        enforced,
        config=config,
    )
    frozen = {int(h) for h in (frozen_horizons or [])}
    for horizon in frozen:
        if int(horizon) in verified:
            verified[int(horizon)] = 0.0

    diagnostics: Dict[str, Dict[str, Any]] = {}
    summary: List[str] = []
    for horizon in key_horizons:
        horizon = int(horizon)
        proposal = float(normalized.get(horizon, 0.0))
        clipped_value = float(clipped.get(horizon, 0.0))
        verified_value = float(verified.get(horizon, 0.0))
        guidance = (structured_horizon_guidance or {}).get(int(horizon)) or {}
        preferred_sign = str(guidance.get("preferred_sign", "")).strip().lower()
        confidence = str(guidance.get("confidence", "")).strip().lower()
        magnitude = str(guidance.get("magnitude", "")).strip().lower()
        mode = str(guidance.get("mode", "")).strip().lower()

        status = "accepted"
        if horizon in frozen or mode == "freeze" or preferred_sign == "zero" or bounds.get(horizon, 0.0) <= 0.0:
            status = "frozen"
        elif abs(proposal - clipped_value) > 1e-9:
            status = "clipped_to_bound"
        if abs(verified_value - clipped_value) > 1e-9:
            if abs(verified_value) < 1e-9:
                status = "zeroed_by_guidance"
            else:
                status = "adjusted_by_guidance"

        diagnostics[f"h{int(horizon)}"] = {
            "proposed_pct": round(proposal, 6),
            "clipped_pct": round(clipped_value, 6),
            "verified_pct": round(verified_value, 6),
            "bound_pct": round(float(bounds.get(horizon, 0.0)), 6),
            "status": status,
            "mode": mode,
            "preferred_sign": preferred_sign,
            "confidence": confidence,
            "magnitude": magnitude,
        }
        summary.append(
            f"h{int(horizon)}: proposed={proposal:+.2f}%, verified={verified_value:+.2f}%, status={status}"
        )

    return {
        "proposed_adjustments": {
            f"h{int(h)}": round(float(v), 6) for h, v in normalized.items()
        },
        "verified_adjustments": {
            f"h{int(h)}": round(float(v), 6) for h, v in verified.items()
        },
        "per_horizon_diagnostics": diagnostics,
        "summary": summary,
    }


def build_hdelta_delta_verifier_tool_spec(
    *,
    tool_name: str = DELTA_VERIFIER_TOOL_NAME,
    key_horizons: Sequence[int] = HDELTA_KEY_HORIZONS,
) -> Dict[str, Any]:
    """OpenAI tool schema for deterministic horizon-delta verification."""
    properties = {
        f"h{int(h)}": {
            "type": "number",
            "description": f"Proposed percentage adjustment for horizon h{int(h)}.",
        }
        for h in key_horizons
    }
    return {
        "type": "function",
        "function": {
            "name": str(tool_name),
            "description": (
                "Verify tentative HDELTA percentage adjustments against deterministic bounds, frozen horizons, "
                "and structured horizon guidance. Use the returned verified_adjustments as the final answer unless "
                "you intentionally shrink them further toward 0 without changing sign."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "proposed_adjustments": {
                        "type": "object",
                        "properties": properties,
                        "additionalProperties": False,
                    },
                },
                "required": ["proposed_adjustments"],
                "additionalProperties": False,
            },
        },
    }


def _parse_summary_metric_value(summary_text: str) -> Dict[str, float]:
    """Parse compact exogenous summary strings like 'last=1, mean20=0.25'."""
    text = str(summary_text or "")
    parsed: Dict[str, float] = {}
    patterns = {
        "last": r"last=([+-]?\d+(?:\.\d+)?)",
        "mean20": r"mean20=([+-]?\d+(?:\.\d+)?)",
        "change_5d_pct": r"5d=([+-]?\d+(?:\.\d+)?)%",
        "change_20d_pct": r"20d=([+-]?\d+(?:\.\d+)?)%",
        "vol20": r"vol20=([+-]?\d+(?:\.\d+)?)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            parsed[key] = float(match.group(1))
    return parsed


def build_market_microstructure_payload(
    exogenous_summary: Optional[Dict[str, Any]],
    *,
    current_case_summary: Optional[Dict[str, Any]] = None,
    requested_features: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Return structured UK auction / ICAP microstructure state from the current sample."""
    summary = dict(exogenous_summary or {})
    if requested_features:
        feature_names = [
            str(name)
            for name in requested_features
            if str(name) in summary
        ]
    else:
        preferred_keywords = ("auction", "icap", "print_day", "spread", "volume", "range_pct")
        feature_names = [
            str(name)
            for name in summary.keys()
            if any(keyword in str(name).lower() for keyword in preferred_keywords)
        ]
        if not feature_names:
            feature_names = [str(name) for name in summary.keys()]

    features: Dict[str, Dict[str, Any]] = {}
    for name in feature_names:
        raw_value = summary.get(name)
        parsed = _parse_summary_metric_value(raw_value)
        payload: Dict[str, Any] = {"summary": str(raw_value)}
        payload.update({k: round(v, 6) for k, v in parsed.items()})
        features[str(name)] = payload

    return {
        "market_features": features,
        "current_case_summary": {
            str(k): str(v) for k, v in (current_case_summary or {}).items()
        },
    }


def build_market_microstructure_tool_spec(
    *,
    tool_name: str = MARKET_MICROSTRUCTURE_TOOL_NAME,
) -> Dict[str, Any]:
    """OpenAI tool schema for structured UK market-state retrieval."""
    return {
        "type": "function",
        "function": {
            "name": str(tool_name),
            "description": (
                "Return structured UK ETS auction and ICAP market-state features for the current case, "
                "including exact parsed last values, recent changes, and spread state."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "requested_features": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional subset of exogenous feature names to inspect.",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    }


def derive_hdelta_freeze_counterexample(
    history: np.ndarray,
    forecast: np.ndarray,
    teaching_examples: Sequence[Dict[str, Any]],
    key_horizons: Sequence[int],
    *,
    quantile: float = 0.35,
) -> Dict[str, Any]:
    """Find a similar low-error historical case that argues for shrinking aggressive adjustments."""
    examples = list(teaching_examples or [])
    if not examples:
        return {}

    current_profile = _hdelta_profile_vector(history, forecast, key_horizons)
    profiles: List[np.ndarray] = []
    prepared: List[Dict[str, Any]] = []
    for ex in examples:
        ex_hist = np.asarray(ex.get("history", []), dtype=float)
        ex_fc = np.asarray(ex.get("forecast", []), dtype=float)
        ex_truth = np.asarray(ex.get("truth", []), dtype=float)
        if ex_hist.size == 0 or ex_fc.size == 0 or ex_truth.size == 0:
            continue
        profiles.append(_hdelta_profile_vector(ex_hist, ex_fc, key_horizons))
        path_len = min(ex_fc.size, ex_truth.size)
        path_mse = float(np.mean((ex_fc[:path_len] - ex_truth[:path_len]) ** 2))
        anchor_error_pct: Dict[int, float] = {}
        for horizon in key_horizons:
            idx = int(horizon) - 1
            if idx < 0 or idx >= ex_fc.size or idx >= ex_truth.size:
                continue
            base_val = float(ex_fc[idx])
            truth_val = float(ex_truth[idx])
            denom = base_val if abs(base_val) > 1e-8 else truth_val
            anchor_error_pct[int(horizon)] = (
                0.0 if abs(denom) < 1e-8 else float((base_val - truth_val) / denom * 100.0)
            )
        prepared.append(
            {
                "date": str(ex.get("date", "")),
                "path_mse": path_mse,
                "retrieval_tag": str(ex.get("retrieval_tag") or (ex.get("case_summary") or {}).get("retrieval_tag") or ""),
                "hindsight_feedback": str((ex.get("case_summary") or {}).get("hindsight_feedback") or ""),
                "selection_role": str((ex.get("case_summary") or {}).get("selection_role") or ""),
                "anchor_error_pct": anchor_error_pct,
            }
        )

    if not prepared:
        return {}

    profiles_arr = np.asarray(profiles, dtype=float)
    center = np.mean(profiles_arr, axis=0, keepdims=True)
    scale = np.std(profiles_arr, axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    distances = np.linalg.norm((profiles_arr - current_profile.reshape(1, -1)) / scale, axis=1)
    path_mse_values = np.asarray([ex["path_mse"] for ex in prepared], dtype=float)
    cutoff = float(np.quantile(path_mse_values, float(quantile)))
    low_error_positions = np.where(path_mse_values <= cutoff)[0]
    if low_error_positions.size == 0:
        low_error_positions = np.argsort(path_mse_values)[: max(1, min(3, len(prepared)))]
    best_pos = int(low_error_positions[np.argmin(distances[low_error_positions])])
    selected = dict(prepared[best_pos])
    selected["match_distance"] = round(float(distances[best_pos]), 6)
    selected["summary"] = (
        "Similar low-error case where the base forecast already stayed close to truth; "
        "shrink unsupported adjustments toward 0.0."
    )
    return selected


def build_freeze_counterexample_tool_spec(
    *,
    tool_name: str = COUNTEREXAMPLE_TOOL_NAME,
) -> Dict[str, Any]:
    """OpenAI tool schema for a contrastive low-error counterexample."""
    return {
        "type": "function",
        "function": {
            "name": str(tool_name),
            "description": (
                "Return one similar low-error historical case that argues for shrinking or freezing an overconfident correction."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    }


def _parse_horizon_key(key: object) -> Optional[int]:
    """Parse config keys like 30 or 'h30' into an integer horizon."""
    text = str(key).strip().lower()
    if not text:
        return None
    if text.startswith("h"):
        text = text[1:]
    try:
        value = int(text)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _hdelta_horizon_cfg(cfg: Optional[Dict], horizon: int) -> Dict:
    """Merge global HDELTA config with optional per-horizon overrides."""
    base = dict(cfg or {})
    overrides = base.get("horizon_overrides", {}) or {}
    merged = dict(base)
    for raw_key, raw_value in overrides.items():
        parsed = _parse_horizon_key(raw_key)
        if parsed != int(horizon) or not isinstance(raw_value, dict):
            continue
        merged.update(raw_value)
    return merged


def _profile_feature_indices_for_horizon(
    key_horizons: Sequence[int],
    horizon: int,
) -> List[int]:
    """Return profile-vector feature indices to use for a horizon-specific match."""
    horizons = [int(h) for h in key_horizons]
    indices = [0, 1, 2, 3]
    try:
        anchor_idx = horizons.index(int(horizon))
    except ValueError:
        return indices
    return indices + [4 + anchor_idx]


def _confidence_rank(value: str) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(str(value or "").strip().lower(), -1)


def enforce_structured_hdelta_adjustments(
    adjustments_pct: Dict[int, float],
    structured_guidance: Optional[Dict[int, Dict[str, str]]],
    per_horizon_max: Dict[int, float],
    config: Optional[Dict] = None,
) -> Dict[int, float]:
    """Apply deterministic sign/confidence/magnitude guards to HDELTA adjustments."""
    if not structured_guidance:
        return {int(h): float(v) for h, v in adjustments_pct.items()}

    cfg = config or {}
    sign_enforcement = bool(cfg.get("structured_enforce_sign", False))
    cap_by_guidance = bool(cfg.get("structured_cap_by_guidance", False))
    freeze_zero_mode = bool(cfg.get("structured_freeze_zero_mode", True))

    raw_min_conf = cfg.get("structured_min_confidence_by_horizon", {}) or {}
    min_conf_by_h = {
        int(parsed): str(value).strip().lower()
        for key, value in raw_min_conf.items()
        if (parsed := _parse_horizon_key(key)) is not None
    }

    magnitude_scale_cfg = {
        "zero": 0.0,
        "tiny": 0.25,
        "small": 0.50,
        "medium": 0.80,
    }
    magnitude_scale_cfg.update(
        {
            str(key).strip().lower(): float(value)
            for key, value in (cfg.get("structured_magnitude_scale", {}) or {}).items()
        }
    )
    confidence_scale_cfg = {
        "low": 0.60,
        "medium": 0.85,
        "high": 1.00,
    }
    confidence_scale_cfg.update(
        {
            str(key).strip().lower(): float(value)
            for key, value in (cfg.get("structured_confidence_scale", {}) or {}).items()
        }
    )

    enforced: Dict[int, float] = {}
    for raw_h, raw_v in adjustments_pct.items():
        horizon = int(raw_h)
        value = float(raw_v)
        guidance = (structured_guidance or {}).get(horizon) or {}
        mode = str(guidance.get("mode", "")).strip().lower()
        preferred_sign = str(guidance.get("preferred_sign", "")).strip().lower()
        confidence = str(guidance.get("confidence", "")).strip().lower()
        magnitude = str(guidance.get("magnitude", "")).strip().lower()
        per_h_max = float(per_horizon_max.get(horizon, abs(value)))

        if freeze_zero_mode and (mode == "freeze" or preferred_sign == "zero" or magnitude == "zero"):
            enforced[horizon] = 0.0
            continue

        required_conf = min_conf_by_h.get(horizon)
        if required_conf and _confidence_rank(confidence) < _confidence_rank(required_conf):
            enforced[horizon] = 0.0
            continue

        if sign_enforcement:
            if preferred_sign == "positive" and value < 0.0:
                value = 0.0
            elif preferred_sign == "negative" and value > 0.0:
                value = 0.0

        if cap_by_guidance:
            mag_scale = float(magnitude_scale_cfg.get(magnitude, 1.0))
            conf_scale = float(confidence_scale_cfg.get(confidence, 1.0))
            guided_cap = float(max(0.0, per_h_max * mag_scale * conf_scale))
        else:
            guided_cap = per_h_max

        if preferred_sign == "positive":
            value = float(np.clip(value, 0.0, guided_cap))
        elif preferred_sign == "negative":
            value = float(np.clip(value, -guided_cap, 0.0))
        else:
            value = float(np.clip(value, -guided_cap, guided_cap))

        enforced[horizon] = value

    return enforced


def apply_structured_hdelta_coherence_guards(
    adjustments_pct: Dict[int, float],
    config: Optional[Dict] = None,
) -> Dict[int, float]:
    """Apply narrow sign-coherence guards learned from prior structured HDELTA runs."""
    cfg = config or {}
    guarded = {int(h): float(v) for h, v in adjustments_pct.items()}
    if not bool(cfg.get("structured_coherence_guards", False)):
        return guarded

    def _sign(value: float) -> int:
        if value > 1e-12:
            return 1
        if value < -1e-12:
            return -1
        return 0

    h5_sign = _sign(guarded.get(5, 0.0))
    h20_sign = _sign(guarded.get(20, 0.0))
    h30_sign = _sign(guarded.get(30, 0.0))

    if bool(cfg.get("structured_zero_negative_long_when_h5_positive", False)) and h5_sign > 0:
        if h20_sign < 0:
            guarded[20] = 0.0
            h20_sign = 0
        if h30_sign < 0:
            guarded[30] = 0.0
            h30_sign = 0

    if (
        bool(cfg.get("structured_zero_h20_negative_when_h30_zero", False))
        and h20_sign < 0
        and h30_sign == 0
    ):
        guarded[20] = 0.0
        h20_sign = 0

    if (
        bool(cfg.get("structured_zero_mixed_long_signs", False))
        and h20_sign != 0
        and h30_sign != 0
        and h20_sign != h30_sign
    ):
        if bool(cfg.get("structured_prefer_positive_long_conflicts", False)):
            if h20_sign < 0:
                guarded[20] = 0.0
            if h30_sign < 0:
                guarded[30] = 0.0
        else:
            guarded[20] = 0.0
            guarded[30] = 0.0

    return guarded


def build_hdelta_case_summary(
    history: np.ndarray,
    forecast: np.ndarray,
    key_horizons: Sequence[int],
    currency: str = "EUR",
) -> Dict[str, str]:
    """Summarize the current history/forecast regime for HDELTA prompts."""
    hist = np.asarray(history, dtype=float)
    fc = np.asarray(forecast, dtype=float)
    if hist.size == 0:
        return {
            "current_price": f"0.00 {currency}",
            "change_5d_pct": "+0.00",
            "change_20d_pct": "+0.00",
            "vol_20d": "0.00",
            "base_anchors": "",
        }

    last_price = float(hist[-1])
    window_20 = hist[-20:] if hist.size >= 20 else hist
    change_5 = _safe_pct_change(hist[-1], hist[-5]) if hist.size >= 5 else 0.0
    change_20 = _safe_pct_change(hist[-1], hist[-20]) if hist.size >= 20 else 0.0
    vol_20 = float(np.std(window_20)) if window_20.size else 0.0
    anchor_parts: List[str] = []
    for horizon in key_horizons:
        idx = int(horizon) - 1
        if idx < 0 or idx >= fc.size:
            continue
        base_val = float(fc[idx])
        drift = _safe_pct_change(base_val, last_price)
        anchor_parts.append(f"h{int(horizon)}={base_val:.2f} ({drift:+.2f}% vs spot)")
    return {
        "current_price": f"{last_price:.2f} {currency}",
        "change_5d_pct": f"{change_5:+.2f}",
        "change_20d_pct": f"{change_20:+.2f}",
        "vol_20d": f"{vol_20:.2f}",
        "base_anchors": "; ".join(anchor_parts),
    }


def build_hdelta_retrieval_tag(history: np.ndarray, forecast: np.ndarray) -> str:
    """Coarse regime tag for retrieval-aware prompting and example annotation."""
    hist = np.asarray(history, dtype=float)
    fc = np.asarray(forecast, dtype=float)
    if hist.size == 0:
        return "trend5=flat|trend20=flat|drift20=flat|drift30=flat|vol=normal"

    last_price = max(abs(float(hist[-1])), 1e-8)
    change_5 = _safe_pct_change(hist[-1], hist[-5]) if hist.size >= 5 else 0.0
    change_20 = _safe_pct_change(hist[-1], hist[-20]) if hist.size >= 20 else 0.0
    drift_20 = _safe_pct_change(float(fc[19]), float(hist[-1])) if fc.size >= 20 else 0.0
    drift_30 = _safe_pct_change(float(fc[29]), float(hist[-1])) if fc.size >= 30 else 0.0
    vol_20 = float(np.std(hist[-20:])) if hist.size else 0.0
    vol_pct = float(vol_20 / last_price * 100.0)

    def _bucket(value: float, tol: float) -> str:
        if value >= tol:
            return "pos"
        if value <= -tol:
            return "neg"
        return "flat"

    if vol_pct < 1.0:
        vol_bucket = "calm"
    elif vol_pct < 2.5:
        vol_bucket = "normal"
    else:
        vol_bucket = "elevated"

    return (
        f"trend5={_bucket(change_5, 0.75)}|"
        f"trend20={_bucket(change_20, 1.50)}|"
        f"drift20={_bucket(drift_20, 0.75)}|"
        f"drift30={_bucket(drift_30, 1.00)}|"
        f"vol={vol_bucket}"
    )


def _hdelta_profile_vector(
    history: np.ndarray,
    forecast: np.ndarray,
    key_horizons: Sequence[int],
) -> np.ndarray:
    """Numerical regime features used to match teaching examples to the current case."""
    hist = np.asarray(history, dtype=float)
    fc = np.asarray(forecast, dtype=float)
    if hist.size == 0:
        return np.zeros(4 + len(tuple(key_horizons)), dtype=float)
    last_price = float(hist[-1])
    vol_20 = float(np.std(hist[-20:])) if hist.size else 0.0
    change_5 = _safe_pct_change(hist[-1], hist[-5]) if hist.size >= 5 else 0.0
    change_20 = _safe_pct_change(hist[-1], hist[-20]) if hist.size >= 20 else 0.0
    anchor_drifts: List[float] = []
    for horizon in key_horizons:
        idx = int(horizon) - 1
        if idx < 0 or idx >= fc.size or abs(last_price) < 1e-8:
            anchor_drifts.append(0.0)
        else:
            anchor_drifts.append(_safe_pct_change(float(fc[idx]), last_price))
    return np.array([last_price, change_5, change_20, vol_20, *anchor_drifts], dtype=float)


def derive_hdelta_case_controls(
    history: np.ndarray,
    forecast: np.ndarray,
    teaching_examples: Sequence[Dict],
    key_horizons: Sequence[int],
    config: Optional[Dict] = None,
    currency: str = "EUR",
) -> Dict:
    """Derive matched-example evidence, dynamic freezes, and per-horizon bounds."""
    cfg = config or {}
    include_retrieval_tag = bool(cfg.get("include_retrieval_tag", False))
    top_k = int(cfg.get("case_match_top_k", min(3, len(teaching_examples) or 1)))
    horizon_specific_matching = bool(cfg.get("horizon_specific_matching", False))
    horizon_top_k = int(cfg.get("horizon_match_top_k", top_k))
    horizon_union_limit = int(
        cfg.get(
            "horizon_match_union_limit",
            max(top_k, min(len(teaching_examples) or 1, len(tuple(key_horizons)) * 2)),
        )
    )
    global_max_pct = float(cfg.get("max_adjustment_pct", 3.0))

    case_summary = build_hdelta_case_summary(history, forecast, key_horizons, currency=currency)
    if include_retrieval_tag:
        case_summary["retrieval_tag"] = build_hdelta_retrieval_tag(history, forecast)
    if not teaching_examples:
        return {
            "current_case_summary": case_summary,
            "matched_examples": [],
            "matched_examples_summary": [],
            "dynamic_freeze_horizons": [],
            "per_horizon_max_adjustment_pct": {
                int(h): global_max_pct for h in key_horizons
            },
            "matched_dates": [],
        }

    current_profile = _hdelta_profile_vector(history, forecast, key_horizons)
    example_profiles = []
    prepared_examples = []
    for ex in teaching_examples:
        ex_hist = np.asarray(ex.get("history", []), dtype=float)
        ex_fc = np.asarray(ex.get("forecast", []), dtype=float)
        ex_truth = np.asarray(ex.get("truth", []), dtype=float)
        ex_profile = _hdelta_profile_vector(ex_hist, ex_fc, key_horizons)
        example_profiles.append(ex_profile)

        error_pct: Dict[int, float] = {}
        for horizon in key_horizons:
            idx = int(horizon) - 1
            if idx < 0 or idx >= ex_fc.size or idx >= ex_truth.size:
                continue
            base_val = float(ex_fc[idx])
            truth_val = float(ex_truth[idx])
            denom = base_val if abs(base_val) > 1e-8 else truth_val
            if abs(float(denom)) < 1e-8:
                error_pct[int(horizon)] = 0.0
            else:
                error_pct[int(horizon)] = float((base_val - truth_val) / denom * 100.0)

        prepared_examples.append(
            {
                **ex,
                "case_summary": {
                    **build_hdelta_case_summary(
                        ex_hist,
                        ex_fc,
                        key_horizons,
                        currency=currency,
                    ),
                    **(ex.get("case_summary") or {}),
                    **(
                        {"retrieval_tag": str(ex.get("retrieval_tag") or build_hdelta_retrieval_tag(ex_hist, ex_fc))}
                        if include_retrieval_tag
                        else {}
                    ),
                },
                "anchor_error_pct": error_pct,
            }
        )

    profiles = np.asarray(example_profiles, dtype=float)
    center = np.mean(profiles, axis=0, keepdims=True)
    scale = np.std(profiles, axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    distances = np.linalg.norm((profiles - current_profile.reshape(1, -1)) / scale, axis=1)
    order = np.argsort(distances)
    top_k = max(1, min(top_k, len(prepared_examples)))
    matched = []
    matched_summaries = []
    matched_by_horizon: Dict[int, List[Dict]] = {}
    horizon_guidance_summary: List[str] = []
    if horizon_specific_matching:
        union_distance: Dict[int, float] = {}
        top_h = max(1, min(horizon_top_k, len(prepared_examples)))
        for horizon in key_horizons:
            feature_idx = _profile_feature_indices_for_horizon(key_horizons, int(horizon))
            current_slice = current_profile[feature_idx]
            horizon_profiles = profiles[:, feature_idx]
            horizon_center = np.mean(horizon_profiles, axis=0, keepdims=True)
            horizon_scale = np.std(horizon_profiles, axis=0, keepdims=True)
            horizon_scale[horizon_scale < 1e-6] = 1.0
            horizon_distances = np.linalg.norm(
                (horizon_profiles - current_slice.reshape(1, -1)) / horizon_scale,
                axis=1,
            )
            horizon_order = np.argsort(horizon_distances)
            horizon_matches = []
            summary_parts = []
            for rank, pos in enumerate(horizon_order[:top_h], start=1):
                ex = dict(prepared_examples[int(pos)])
                ex["match_rank"] = rank
                ex["match_distance"] = float(horizon_distances[int(pos)])
                horizon_matches.append(ex)
                prev_dist = union_distance.get(int(pos))
                if prev_dist is None or float(horizon_distances[int(pos)]) < prev_dist:
                    union_distance[int(pos)] = float(horizon_distances[int(pos)])
                err_pct = float(ex["anchor_error_pct"].get(int(horizon), 0.0))
                summary_parts.append(
                    f"rank={rank}; date={ex.get('date', '')}; distance={float(horizon_distances[int(pos)]):.2f}; "
                    f"h{int(horizon)}={err_pct:+.2f}%"
                )
            matched_by_horizon[int(horizon)] = horizon_matches
            matched_summaries.append(
                f"h{int(horizon)} matches: {' | '.join(summary_parts)}"
            )

        union_limit = max(1, min(horizon_union_limit, len(prepared_examples)))
        union_order = sorted(
            union_distance.items(),
            key=lambda item: (float(item[1]), str(prepared_examples[int(item[0])].get("date", ""))),
        )
        for rank, (pos, distance) in enumerate(union_order[:union_limit], start=1):
            ex = dict(prepared_examples[int(pos)])
            ex["match_rank"] = rank
            ex["match_distance"] = float(distance)
            matched.append(ex)
    else:
        for rank, pos in enumerate(order[:top_k], start=1):
            ex = dict(prepared_examples[int(pos)])
            ex["match_rank"] = rank
            ex["match_distance"] = float(distances[int(pos)])
            matched.append(ex)
            anchor_parts = []
            for horizon in key_horizons:
                if int(horizon) not in ex["anchor_error_pct"]:
                    continue
                err_pct = ex["anchor_error_pct"][int(horizon)]
                anchor_parts.append(f"h{int(horizon)}={err_pct:+.2f}%")
            matched_summaries.append(
                (
                    f"rank={rank}; date={ex.get('date', '')}; "
                    f"distance={float(distances[int(pos)]):.2f}; "
                    f"errors: {'; '.join(anchor_parts)}"
                )
            )

    dynamic_freeze: List[int] = []
    per_horizon_max = {int(h): global_max_pct for h in key_horizons}
    for horizon in key_horizons:
        horizon_cfg = _hdelta_horizon_cfg(cfg, int(horizon))
        min_examples = int(horizon_cfg.get("case_min_examples", 2))
        min_sign_agreement = float(horizon_cfg.get("case_min_sign_agreement", 0.67))
        min_mean_abs_error_pct = float(horizon_cfg.get("case_min_mean_abs_error_pct", 0.20))
        bound_scale = float(horizon_cfg.get("case_bound_scale", 0.75))
        min_bound_pct = float(horizon_cfg.get("case_min_bound_pct", 0.15))
        if bool(horizon_cfg.get("force_freeze", False)):
            dynamic_freeze.append(int(horizon))
            per_horizon_max[int(horizon)] = 0.0
            horizon_guidance_summary.append(
                f"h{int(horizon)}: freeze to 0.0, forced by horizon-specific configuration."
            )
            continue
        horizon_matches = (
            matched_by_horizon.get(int(horizon), matched)
            if horizon_specific_matching
            else matched
        )
        err_values = [
            float(ex["anchor_error_pct"][int(horizon)])
            for ex in horizon_matches
            if int(horizon) in ex["anchor_error_pct"]
        ]
        if len(err_values) < min_examples:
            dynamic_freeze.append(int(horizon))
            per_horizon_max[int(horizon)] = 0.0
            horizon_guidance_summary.append(
                f"h{int(horizon)}: freeze to 0.0, only {len(err_values)} matched examples carried usable evidence."
            )
            continue
        err_arr = np.asarray(err_values, dtype=float)
        mean_abs = float(np.mean(np.abs(err_arr)))
        mean_err = float(np.mean(err_arr))
        sign_agreement = float(
            max(np.mean(err_arr > 0.0), np.mean(err_arr < 0.0))
        )
        if mean_abs < min_mean_abs_error_pct or sign_agreement < min_sign_agreement:
            dynamic_freeze.append(int(horizon))
            per_horizon_max[int(horizon)] = 0.0
            horizon_guidance_summary.append(
                f"h{int(horizon)}: freeze to 0.0, weak/conflicted evidence (mean_err={mean_err:+.2f}%, sign_agreement={sign_agreement:.2f})."
            )
            continue
        per_horizon_max[int(horizon)] = float(
            min(global_max_pct, max(min_bound_pct, mean_abs * bound_scale))
        )
        preferred_sign = "negative" if mean_err > 0.0 else "positive"
        source_text = (
            f"{len(horizon_matches)} horizon-specific matches"
            if horizon_specific_matching
            else f"{len(matched)} matched examples"
        )
        horizon_guidance_summary.append(
            f"h{int(horizon)}: actionable, {source_text} show mean_err={mean_err:+.2f}% with sign_agreement={sign_agreement:.2f}; prefer a small {preferred_sign} adjustment within +/-{per_horizon_max[int(horizon)]:.2f}%."
        )

    return {
        "current_case_summary": case_summary,
        "matched_examples": matched,
        "matched_examples_summary": matched_summaries,
        "horizon_guidance_summary": horizon_guidance_summary,
        "dynamic_freeze_horizons": sorted(set(dynamic_freeze)),
        "per_horizon_max_adjustment_pct": per_horizon_max,
        "matched_dates": [str(ex.get("date", "")) for ex in matched],
    }


def returns_to_prices(returns: np.ndarray, last_price: float) -> np.ndarray:
    """Convert log returns to price path."""
    last_price = float(last_price)
    return last_price * np.exp(np.cumsum(returns, axis=0))


class LLMRefiner:
    """
    LLM-based forecast refiner.
    
    Uses large language models to refine time series forecasts
    based on historical context and domain knowledge.
    """
    
    def __init__(
        self,
        config: Dict,
        cache_dir: Optional[Path] = None,
        log_dir: Optional[Path] = None
    ):
        """
        Initialize the LLM refiner.
        
        Args:
            config: LLM configuration dict
            cache_dir: Directory for response caching
            log_dir: Directory for logging
        """
        self.config = config
        
        self.provider = config.get("provider", "openai")
        self.model = config.get("model", "gpt-5.2")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 1000)
        self.max_retries = config.get("max_retries", 3)
        self.timeout_seconds = float(config.get("timeout_seconds", 120.0))
        self.client_max_retries = int(config.get("client_max_retries", 2))
        self.blend_config = config.get("blend", {})
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url")
        self.disable_response_format = bool(config.get("disable_response_format", False))
        self.disable_response_format_base_urls = tuple(
            str(item).strip().lower()
            for item in (config.get("disable_response_format_base_urls", []) or [])
            if str(item).strip()
        )
        self.disable_response_format_models = tuple(
            str(item).strip().lower()
            for item in (config.get("disable_response_format_models", []) or [])
            if str(item).strip()
        )
        self.currency = str(config.get("currency", "EUR"))
        self.market_name = str(config.get("market_name", "carbon allowance market"))
        
        # Initialize cache
        if cache_dir and config.get("cache_enabled", True):
            self.cache = ResponseCache(cache_dir)
        else:
            self.cache = None
        
        # Initialize logger
        if log_dir:
            self.log_store = LogStore(log_dir)
        else:
            self.log_store = None
        
        # API client (lazy initialization)
        self._clients: Dict[Tuple[str, str, str], object] = {}
        self._response_format_support: Dict[Tuple[str, str], bool] = {}
        self._last_llm_response: Dict[str, object] = {}
    
    def _blend_forecast(self, base: np.ndarray, refined: np.ndarray) -> np.ndarray:
        """Blend refined forecast with base using horizon-weighted weights."""
        mode = self.blend_config.get("mode", "horizon_weighted")
        if mode == "fixed":
            weight = float(self.blend_config.get("weight", 0.5))
            return base * (1 - weight) + refined * weight
        if mode != "horizon_weighted":
            return refined
        
        min_weight = float(self.blend_config.get("min_weight", 0.1))
        max_weight = float(self.blend_config.get("max_weight", 0.6))
        power = float(self.blend_config.get("power", 1.0))
        
        ramp = np.linspace(min_weight, max_weight, base.shape[-1])
        if power != 1.0:
            ramp = np.power(ramp, power)
        return base * (1 - ramp) + refined * ramp

    def _response_format_key(
        self,
        model_override: Optional[str] = None,
        base_url_override: Optional[str] = None,
    ) -> Tuple[str, str]:
        import os

        model_name = str(model_override or self.model)
        base_url = base_url_override or self.base_url or os.getenv("OPENAI_BASE_URL") or ""
        return model_name, base_url

    def _response_format_allowed(
        self,
        response_format: Optional[Dict],
        model_override: Optional[str] = None,
        base_url_override: Optional[str] = None,
    ) -> Optional[Dict]:
        if response_format is None:
            return None
        if self._response_format_disabled_by_policy(
            model_override=model_override,
            base_url_override=base_url_override,
        ):
            return None
        if not self._response_format_support.get(
            self._response_format_key(model_override, base_url_override),
            True,
        ):
            return None
        return response_format

    def _response_format_disabled_by_policy(
        self,
        model_override: Optional[str] = None,
        base_url_override: Optional[str] = None,
    ) -> bool:
        """Whether response_format should be disabled for this model/endpoint."""
        import os

        if self.disable_response_format:
            return True

        model_name = str(model_override or self.model).strip().lower()
        base_url = (
            base_url_override or self.base_url or os.getenv("OPENAI_BASE_URL") or ""
        ).strip().lower()

        if any(token in model_name for token in self.disable_response_format_models):
            return True
        if any(token in base_url for token in self.disable_response_format_base_urls):
            return True
        return False

    def _mark_response_format_unsupported(
        self,
        model_override: Optional[str] = None,
        base_url_override: Optional[str] = None,
    ) -> None:
        self._response_format_support[
            self._response_format_key(model_override, base_url_override)
        ] = False
    
    def _get_client(
        self,
        api_key_override: Optional[str] = None,
        base_url_override: Optional[str] = None,
    ):
        """Get or create API client."""
        import os

        api_key = api_key_override or self.api_key or os.getenv("OPENAI_API_KEY") or ""
        base_url = base_url_override or self.base_url or os.getenv("OPENAI_BASE_URL") or ""
        cache_key = (self.provider, base_url, api_key)
        if cache_key in self._clients:
            return self._clients[cache_key]
        
        if self.provider == "openai":
            try:
                from openai import OpenAI
                kwargs = {}
                if api_key:
                    kwargs["api_key"] = api_key
                if base_url:
                    kwargs["base_url"] = base_url
                kwargs["timeout"] = self.timeout_seconds
                kwargs["max_retries"] = self.client_max_retries
                client = OpenAI(**kwargs)
            except ImportError:
                raise ImportError("openai package required for OpenAI provider")
        elif self.provider == "anthropic":
            try:
                import anthropic
                client = anthropic.Anthropic()
            except ImportError:
                raise ImportError("anthropic package required for Anthropic provider")
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
        self._clients[cache_key] = client
        return client
    
    def _call_llm(
        self,
        prompt: str,
        system_message: str,
        response_format: Optional[Dict] = None,
        model_override: Optional[str] = None,
        api_key_override: Optional[str] = None,
        base_url_override: Optional[str] = None,
        temperature_override: Optional[float] = None,
        cache_key_suffix: Optional[str] = None,
        max_tokens_override: Optional[int] = None,
    ) -> str:
        """
        Call the LLM API.
        
        Args:
            prompt: User prompt
            system_message: System message
            
        Returns:
            Response text
        """
        import os

        model_name = str(model_override or self.model)
        base_url = base_url_override or self.base_url or os.getenv("OPENAI_BASE_URL") or ""
        temperature = float(self.temperature if temperature_override is None else temperature_override)
        max_tokens = int(self.max_tokens if max_tokens_override is None else max_tokens_override)
        prompt_key = prompt if cache_key_suffix is None else f"{prompt}\n<!-- {cache_key_suffix} -->"
        response_format = self._response_format_allowed(
            response_format,
            model_override=model_override,
            base_url_override=base_url_override,
        )

        # Check cache first
        if self.cache:
            cache_context = {
                "provider": self.provider,
                "base_url": base_url,
                "system_message": system_message,
                "max_tokens": max_tokens,
                "response_format": response_format,
            }
            cached = self.cache.get(
                prompt_key, model_name, temperature, context=cache_context
            )
            if cached:
                self._last_llm_response = dict(cached)
                return cached.get("content", "")
        
        client = self._get_client(
            api_key_override=api_key_override,
            base_url_override=base_url_override,
        )
        
        if self.provider == "openai":
            request = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature
            }
            if response_format is not None:
                request["response_format"] = response_format
            if model_name.startswith("gpt-5"):
                request["max_completion_tokens"] = max_tokens
            else:
                request["max_tokens"] = max_tokens
            response = client.chat.completions.create(**request)
            choice = response.choices[0]
            message = choice.message
            content = _extract_openai_message_text(message)
            reasoning_content = _extract_openai_reasoning_text(message)
            used_reasoning_fallback = False
            if not content and reasoning_content:
                content = reasoning_content
                used_reasoning_fallback = True
            self._last_llm_response = {
                "content": content,
                "reasoning_content": reasoning_content,
                "finish_reason": getattr(choice, "finish_reason", None),
                "response_model": getattr(response, "model", model_name),
                "used_reasoning_fallback": used_reasoning_fallback,
            }
        
        elif self.provider == "anthropic":
            response = client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                system=system_message,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            content = response.content[0].text
            self._last_llm_response = {
                "content": content,
                "reasoning_content": "",
                "finish_reason": None,
                "response_model": model_name,
                "used_reasoning_fallback": False,
            }
        
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
        
        # Cache response
        if self.cache:
            self.cache.set(
                prompt_key,
                model_name,
                temperature,
                dict(self._last_llm_response),
                context=cache_context,
            )
        
        return content

    def _call_llm_messages(
        self,
        messages: List[Dict[str, str]],
        system_message: str = "",
        response_format: Optional[Dict] = None,
        model_override: Optional[str] = None,
        api_key_override: Optional[str] = None,
        base_url_override: Optional[str] = None,
        temperature_override: Optional[float] = None,
        cache_key_suffix: Optional[str] = None,
        max_tokens_override: Optional[int] = None,
    ) -> str:
        """Call the LLM API with an explicit multi-message chat history."""
        import os

        payload = {"messages": messages}
        if system_message:
            payload["system_message"] = system_message
        prompt_key = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        if cache_key_suffix is not None:
            prompt_key = f"{prompt_key}\n<!-- {cache_key_suffix} -->"
        model_name = str(model_override or self.model)
        base_url = base_url_override or self.base_url or os.getenv("OPENAI_BASE_URL") or ""
        temperature = float(self.temperature if temperature_override is None else temperature_override)
        max_tokens = int(self.max_tokens if max_tokens_override is None else max_tokens_override)
        response_format = self._response_format_allowed(
            response_format,
            model_override=model_override,
            base_url_override=base_url_override,
        )

        # Cache
        if self.cache:
            cache_context = {
                "provider": self.provider,
                "base_url": base_url,
                "system_message": system_message,
                "max_tokens": max_tokens,
                "response_format": response_format,
            }
            cached = self.cache.get(
                prompt_key, model_name, temperature, context=cache_context
            )
            if cached:
                self._last_llm_response = dict(cached)
                return cached.get("content", "")

        client = self._get_client(
            api_key_override=api_key_override,
            base_url_override=base_url_override,
        )

        if self.provider == "openai":
            request_messages: List[Dict[str, str]] = []
            if system_message:
                request_messages.append({"role": "system", "content": system_message})
            request_messages.extend(messages)

            request = {
                "model": model_name,
                "messages": request_messages,
                "temperature": temperature,
            }
            if response_format is not None:
                request["response_format"] = response_format
            if model_name.startswith("gpt-5"):
                request["max_completion_tokens"] = max_tokens
            else:
                request["max_tokens"] = max_tokens

            response = client.chat.completions.create(**request)
            choice = response.choices[0]
            message = choice.message
            content = _extract_openai_message_text(message)
            reasoning_content = _extract_openai_reasoning_text(message)
            used_reasoning_fallback = False
            if not content and reasoning_content:
                content = reasoning_content
                used_reasoning_fallback = True
            self._last_llm_response = {
                "content": content,
                "reasoning_content": reasoning_content,
                "finish_reason": getattr(choice, "finish_reason", None),
                "response_model": getattr(response, "model", model_name),
                "used_reasoning_fallback": used_reasoning_fallback,
            }

        elif self.provider == "anthropic":
            response = client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                system=system_message,
                messages=messages,
            )
            content = response.content[0].text
            self._last_llm_response = {
                "content": content,
                "reasoning_content": "",
                "finish_reason": None,
                "response_model": model_name,
                "used_reasoning_fallback": False,
            }
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        if self.cache:
            self.cache.set(
                prompt_key,
                model_name,
                temperature,
                dict(self._last_llm_response),
                context=cache_context,
            )

        return content

    def _call_llm_messages_with_tools(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: List[Dict[str, Any]],
        tool_executor: Callable[[str, Dict[str, Any]], Dict[str, Any]],
        system_message: str = "",
        model_override: Optional[str] = None,
        api_key_override: Optional[str] = None,
        base_url_override: Optional[str] = None,
        temperature_override: Optional[float] = None,
        max_tokens_override: Optional[int] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = "auto",
        max_tool_rounds: int = 2,
        required_tool_names: Optional[Sequence[str]] = None,
    ) -> str:
        """Call the LLM with OpenAI-style function tools and execute tool calls locally."""
        import os

        if self.provider != "openai":
            raise ValueError("Tool calling is only implemented for OpenAI-compatible providers.")

        model_name = str(model_override or self.model)
        base_url = base_url_override or self.base_url or os.getenv("OPENAI_BASE_URL") or ""
        temperature = float(self.temperature if temperature_override is None else temperature_override)
        max_tokens = int(self.max_tokens if max_tokens_override is None else max_tokens_override)
        client = self._get_client(
            api_key_override=api_key_override,
            base_url_override=base_url_override,
        )

        request_messages: List[Dict[str, Any]] = []
        if system_message:
            request_messages.append({"role": "system", "content": system_message})
        request_messages.extend(list(messages))

        executed_tool_names: List[str] = []
        tool_rounds = 0
        required_names = [str(name) for name in (required_tool_names or []) if str(name)]

        for round_idx in range(max(1, int(max_tool_rounds) + 1)):
            request: Dict[str, Any] = {
                "model": model_name,
                "messages": request_messages,
                "temperature": temperature,
                "tools": tools,
            }
            if round_idx == 0 and tool_choice is not None:
                request["tool_choice"] = tool_choice
            if model_name.startswith("gpt-5"):
                request["max_completion_tokens"] = max_tokens
            else:
                request["max_tokens"] = max_tokens

            response = client.chat.completions.create(**request)
            choice = response.choices[0]
            message = choice.message
            content = _extract_openai_message_text(message)
            reasoning_content = _extract_openai_reasoning_text(message)
            used_reasoning_fallback = False
            if not content and reasoning_content:
                content = reasoning_content
                used_reasoning_fallback = True

            raw_tool_calls = list(getattr(message, "tool_calls", None) or [])
            if raw_tool_calls:
                tool_rounds += 1
                assistant_message: Dict[str, Any] = {
                    "role": "assistant",
                    "content": content or "",
                    "tool_calls": [],
                }
                for raw_call in raw_tool_calls:
                    fn = getattr(raw_call, "function", None)
                    fn_name = str(getattr(fn, "name", "") or "")
                    fn_args_raw = str(getattr(fn, "arguments", "") or "")
                    fn_args: Dict[str, Any]
                    try:
                        parsed_args = json.loads(fn_args_raw) if fn_args_raw.strip() else {}
                        fn_args = parsed_args if isinstance(parsed_args, dict) else {}
                    except json.JSONDecodeError:
                        fn_args = {}

                    tool_call_id = str(getattr(raw_call, "id", "") or f"tool_call_{len(executed_tool_names)}")
                    assistant_message["tool_calls"].append(
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": fn_name,
                                "arguments": fn_args_raw,
                            },
                        }
                    )
                    executed_tool_names.append(fn_name)
                request_messages.append(assistant_message)

                for raw_call in raw_tool_calls:
                    fn = getattr(raw_call, "function", None)
                    fn_name = str(getattr(fn, "name", "") or "")
                    fn_args_raw = str(getattr(fn, "arguments", "") or "")
                    try:
                        parsed_args = json.loads(fn_args_raw) if fn_args_raw.strip() else {}
                        fn_args = parsed_args if isinstance(parsed_args, dict) else {}
                    except json.JSONDecodeError:
                        fn_args = {}
                    tool_call_id = str(getattr(raw_call, "id", "") or "")
                    try:
                        tool_result = tool_executor(fn_name, fn_args)
                    except Exception as e:
                        tool_result = {"error": str(e), "tool_name": fn_name}
                    request_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": fn_name,
                            "content": json.dumps(tool_result, ensure_ascii=True),
                        }
                    )
                continue

            missing_required = [
                name for name in required_names if name not in executed_tool_names
            ]
            if missing_required and round_idx < int(max_tool_rounds):
                if content or reasoning_content:
                    request_messages.append(
                        {
                            "role": "assistant",
                            "content": content or reasoning_content or "",
                        }
                    )
                request_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Before finalizing, call each remaining required tool exactly once: "
                            + ", ".join(missing_required)
                            + ". Then return the final JSON only."
                        ),
                    }
                )
                continue

            self._last_llm_response = {
                "content": content,
                "reasoning_content": reasoning_content,
                "finish_reason": getattr(choice, "finish_reason", None),
                "response_model": getattr(response, "model", model_name),
                "used_reasoning_fallback": used_reasoning_fallback,
                "tool_calls": list(executed_tool_names),
                "tool_rounds": int(tool_rounds),
                "tool_invocations": int(len(executed_tool_names)),
                "missing_required_tool_calls": missing_required,
                "base_url": base_url,
            }
            return content

        raise RuntimeError("LLM tool-calling loop exceeded max_tool_rounds without final content.")

    def refine(
        self,
        method: str,
        history: np.ndarray,
        dates: List[str],
        tsm_forecast: Optional[np.ndarray] = None,
        pred_len: int = 30,
        exogenous_summary: Optional[Dict] = None,
        price_base: Optional[float] = None,
        teaching_examples: Optional[List[Dict]] = None,
        sentiment_history: Optional[np.ndarray] = None
    ) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Generate or refine forecast using LLM.
        
        Args:
            method: Prompt method (DP, CoT, CoT-RF, TSM+LLM)
            history: Historical price values
            dates: Historical dates
            tsm_forecast: TSM model forecast (required for TSM+LLM)
            pred_len: Prediction length
            exogenous_summary: Summary of exogenous variables
            
        Returns:
            Tuple of (forecast array, metadata dict)
        """
        if method == "TSM+NEWS-DRIFT":
            if tsm_forecast is None:
                raise ValueError("TSM forecast required for TSM+NEWS-DRIFT method")
            if sentiment_history is None:
                raise ValueError("Sentiment history required for TSM+NEWS-DRIFT method")

            drift_cfg = self.config.get("news_drift", {})
            horizon = int(drift_cfg.get("horizons", 5))
            sentiment_window = int(drift_cfg.get("sentiment_window", 1))
            min_abs = float(drift_cfg.get("min_abs", 0.0))
            scale = float(drift_cfg.get("scale", 1.0))
            pos_mult = float(drift_cfg.get("pos_mult", 1.0))
            neg_mult = float(drift_cfg.get("neg_mult", 1.5))
            vol_window = int(drift_cfg.get("vol_window", 20))
            max_pct = float(drift_cfg.get("max_adjust_pct", 0.03))
            decay_mode = str(drift_cfg.get("decay", "linear")).lower()
            half_life = float(drift_cfg.get("half_life", 3.0))
            calib_cfg = drift_cfg.get("calibration", {})

            if sentiment_window > 0:
                recent_sent = sentiment_history[-sentiment_window:]
            else:
                recent_sent = sentiment_history
            score = float(np.mean(recent_sent)) if len(recent_sent) else 0.0

            if abs(score) < min_abs or horizon <= 0:
                return tsm_forecast, {
                    "method": method,
                    "model": self.model,
                    "temperature": self.temperature,
                    "success": True,
                    "applied": False,
                    "sentiment_score": score,
                }

            if len(history) >= vol_window + 1:
                history_slice = history[-(vol_window + 1):]
            else:
                history_slice = history
            safe_hist = np.clip(history_slice, 1e-8, None)
            returns = np.diff(np.log(safe_hist))
            return_std = float(np.std(returns)) if len(returns) else 0.0
            if return_std < 1e-6:
                return_std = 0.01

            weight = pos_mult if score >= 0 else neg_mult

            if calib_cfg.get("enabled") and "beta" in calib_cfg:
                alpha = float(calib_cfg.get("alpha", 0.0))
                beta = float(calib_cfg.get("beta", 0.0))
                calib_scale = float(calib_cfg.get("scale", 1.0))
                base_adj = (alpha + beta * score) * calib_scale * weight
            else:
                base_adj = score * scale * weight * return_std
            horizon = min(horizon, pred_len, len(tsm_forecast))

            if decay_mode == "exp":
                decay = np.exp(-np.arange(horizon) / max(half_life, 1e-6))
            else:
                decay = np.linspace(1.0, 0.0, horizon, endpoint=True)

            adjusted = np.array(tsm_forecast, copy=True)
            for idx in range(horizon):
                factor = float(np.exp(base_adj * decay[idx]))
                if max_pct > 0.0:
                    factor = max(1.0 - max_pct, min(1.0 + max_pct, factor))
                adjusted[idx] = tsm_forecast[idx] * factor
            adjusted = np.clip(adjusted, 0.01, None)

            return adjusted, {
                "method": method,
                "model": self.model,
                "temperature": self.temperature,
                "success": True,
                "applied": True,
                "sentiment_score": score,
                "return_std": return_std,
                "base_adjustment": base_adj,
                "horizons": horizon,
                "calibrated": bool(calib_cfg.get("enabled") and "beta" in calib_cfg),
            }

        if method == "NEWS-SENTIMENT-ONLY":
            if sentiment_history is None:
                raise ValueError("Sentiment history required for NEWS-SENTIMENT-ONLY method")
            if price_base is None:
                if len(history):
                    price_base = float(history[-1])
                else:
                    raise ValueError("price_base required for NEWS-SENTIMENT-ONLY method")

            signal_cfg = self.config.get("news_signal")
            if signal_cfg is None:
                signal_cfg = self.config.get("news_drift", {})

            horizon = int(signal_cfg.get("horizons", pred_len))
            sentiment_window = int(signal_cfg.get("sentiment_window", 1))
            min_abs = float(signal_cfg.get("min_abs", 0.0))
            scale = float(signal_cfg.get("scale", 0.5))
            pos_mult = float(signal_cfg.get("pos_mult", 1.0))
            neg_mult = float(signal_cfg.get("neg_mult", 1.0))
            vol_window = int(signal_cfg.get("vol_window", 20))
            max_pct = float(signal_cfg.get("max_adjust_pct", 0.03))
            decay_mode = str(signal_cfg.get("decay", "linear")).lower()
            half_life = float(signal_cfg.get("half_life", 3.0))

            if sentiment_window > 0:
                recent_sent = sentiment_history[-sentiment_window:]
            else:
                recent_sent = sentiment_history
            score = float(np.mean(recent_sent)) if len(recent_sent) else 0.0

            horizon = min(horizon, pred_len)
            if abs(score) < min_abs or horizon <= 0:
                forecast = np.full(pred_len, float(price_base))
                return forecast, {
                    "method": method,
                    "model": self.model,
                    "temperature": self.temperature,
                    "success": True,
                    "applied": False,
                    "sentiment_score": score,
                }

            if len(history) >= vol_window + 1:
                history_slice = history[-(vol_window + 1):]
            else:
                history_slice = history
            safe_hist = np.clip(history_slice, 1e-8, None)
            returns = np.diff(np.log(safe_hist))
            return_std = float(np.std(returns)) if len(returns) else 0.0
            if return_std < 1e-6:
                return_std = 0.01

            weight = pos_mult if score >= 0 else neg_mult
            base_adj = score * scale * weight * return_std

            if decay_mode == "exp":
                decay = np.exp(-np.arange(horizon) / max(half_life, 1e-6))
            else:
                decay = np.linspace(1.0, 0.0, horizon, endpoint=True)

            forecast = np.zeros(pred_len)
            current = float(price_base)
            for idx in range(pred_len):
                step = base_adj * decay[idx] if idx < horizon else 0.0
                factor = float(np.exp(step))
                if max_pct > 0.0:
                    factor = max(1.0 - max_pct, min(1.0 + max_pct, factor))
                current *= factor
                forecast[idx] = max(current, 0.01)

            return forecast, {
                "method": method,
                "model": self.model,
                "temperature": self.temperature,
                "success": True,
                "applied": True,
                "sentiment_score": score,
                "return_std": return_std,
                "base_adjustment": base_adj,
                "horizons": horizon,
            }

        template = None
        if method not in (
            "TSM+LLM-COT-RF",
            "TSM+LLM-COT-RF-HDELTA",
            "TSM+LLM-COT-SENT-RF",
            "TSM+LLM-COT-SENT-RF-DELTA",
            "TSM+LLM-COT-SENT-RF-HDELTA",
            "TSM+LLM-COT-SENT-RF-HPRICE",
        ):
            template = get_template(method)

        # Prepare prompt
        if method == "TSM+LLM":
            if tsm_forecast is None:
                raise ValueError("TSM forecast required for TSM+LLM method")
            prompt = template.format(
                history=history,
                dates=dates,
                tsm_forecast=tsm_forecast,
                pred_len=pred_len,
                exogenous_summary=exogenous_summary,
                currency=self.currency,
            )
        elif method == "TSM+LLM-NORM-DELTA":
            if tsm_forecast is None:
                raise ValueError("TSM forecast required for TSM+LLM-NORM-DELTA method")
            norm_cfg = self.config.get("norm_delta", {})
            window = int(norm_cfg.get("history_window", len(history)))
            max_delta = float(norm_cfg.get("max_delta", 0.5))
            hist_window = history[-window:] if len(history) >= window else history
            mean = float(np.mean(hist_window)) if len(hist_window) else 0.0
            std = float(np.std(hist_window)) if len(hist_window) else 1.0
            if std < 1e-6:
                std = 1.0
            norm_history = (history - mean) / std
            norm_forecast = (tsm_forecast - mean) / std
            prompt = template.format(
                history=norm_history,
                dates=dates,
                tsm_forecast=norm_forecast,
                pred_len=pred_len,
                currency=self.currency,
                history_mean=mean,
                history_std=std,
                max_delta=max_delta
            )
        elif method == "TSM+LLM-COT-SENT":
            if tsm_forecast is None:
                raise ValueError("TSM forecast required for TSM+LLM-COT-SENT method")
            if sentiment_history is None:
                raise ValueError("Sentiment history required for TSM+LLM-COT-SENT method")
            prompt = template.format(
                history=history,
                dates=dates,
                tsm_forecast=tsm_forecast,
                sentiment_history=sentiment_history,
                pred_len=pred_len,
                currency=self.currency,
            )
        elif method == "TSM+LLM-COT-RF":
            if tsm_forecast is None:
                raise ValueError("TSM forecast required for CoT-RF method")
            if not teaching_examples:
                raise ValueError("Teaching examples required for CoT-RF method")
            cot_rf_cfg = self.config.get("cot_rf", {}) or {}
            strict_json_prompt = bool(cot_rf_cfg.get("strict_json_prompt", False))
            if strict_json_prompt:
                reflection_template = get_template("CoT-RF-REFLECT-STRICTJSON")
            else:
                reflection_template = get_template("CoT-RF-REFLECT")
            apply_template = get_template("CoT-RF-APPLY")
            prompt_hist = int(self.config.get("prompt_history_points", 30))
            prompt = reflection_template.format(
                examples=teaching_examples,
                pred_len=pred_len,
                history_points=prompt_hist,
                exogenous_summary=exogenous_summary,
                currency=self.currency,
            )
        elif method == "TSM+LLM-COT-RF-HDELTA":
            if tsm_forecast is None:
                raise ValueError("TSM forecast required for CoT-RF-HDELTA method")
            if not teaching_examples:
                raise ValueError("Teaching examples required for CoT-RF-HDELTA method")
            cot_rf_cfg = self.config.get("cot_rf", {}) or {}
            strict_json_prompt = bool(cot_rf_cfg.get("strict_json_prompt", False))
            if strict_json_prompt:
                reflection_template = get_template("CoT-RF-HDELTA-REFLECT-STRICTJSON")
            else:
                reflection_template = get_template("CoT-RF-REFLECT")
            apply_template = get_template("CoT-RF-HDELTA-APPLY")
            prompt_hist = int(self.config.get("prompt_history_points", 30))
            prompt = reflection_template.format(
                examples=teaching_examples,
                pred_len=pred_len,
                history_points=prompt_hist,
                exogenous_summary=exogenous_summary,
                currency=self.currency,
            )
        elif method == "TSM+LLM-COT-SENT-RF":
            if tsm_forecast is None:
                raise ValueError("TSM forecast required for CoT-SENT-RF method")
            if not teaching_examples:
                raise ValueError("Teaching examples required for CoT-SENT-RF method")
            if sentiment_history is None:
                raise ValueError("Sentiment history required for CoT-SENT-RF method")
            cot_rf_cfg = self.config.get("cot_rf", {}) or {}
            strict_json_prompt = bool(cot_rf_cfg.get("strict_json_prompt", False))
            if strict_json_prompt:
                reflection_template = get_template("CoT-SENT-RF-REFLECT-STRICTJSON")
                apply_template = get_template("CoT-SENT-RF-APPLY-STRICTJSON")
            else:
                reflection_template = get_template("CoT-SENT-RF-REFLECT")
                apply_template = get_template("CoT-SENT-RF-APPLY")
            prompt_hist = int(self.config.get("prompt_history_points", 30))
            prompt_sent = int(self.config.get("prompt_sentiment_points", 30))
            prompt = reflection_template.format(
                examples=teaching_examples,
                pred_len=pred_len,
                history_points=prompt_hist,
                sentiment_points=prompt_sent,
                exogenous_summary=exogenous_summary,
                currency=self.currency,
            )
        elif method == "TSM+LLM-COT-SENT-RF-DELTA":
            if tsm_forecast is None:
                raise ValueError("TSM forecast required for CoT-SENT-RF-DELTA method")
            if not teaching_examples:
                raise ValueError("Teaching examples required for CoT-SENT-RF-DELTA method")
            if sentiment_history is None:
                raise ValueError("Sentiment history required for CoT-SENT-RF-DELTA method")
            reflection_template = get_template("CoT-SENT-RF-REFLECT")
            apply_template = get_template("CoT-SENT-RF-DELTA-APPLY")
            prompt_hist = int(self.config.get("prompt_history_points", 30))
            prompt_sent = int(self.config.get("prompt_sentiment_points", 30))
            prompt = reflection_template.format(
                examples=teaching_examples,
                pred_len=pred_len,
                history_points=prompt_hist,
                sentiment_points=prompt_sent,
                exogenous_summary=exogenous_summary,
                currency=self.currency,
            )
        elif method == "TSM+LLM-COT-SENT-RF-HDELTA":
            if tsm_forecast is None:
                raise ValueError("TSM forecast required for CoT-SENT-RF-HDELTA method")
            if not teaching_examples:
                raise ValueError("Teaching examples required for CoT-SENT-RF-HDELTA method")
            if sentiment_history is None:
                raise ValueError("Sentiment history required for CoT-SENT-RF-HDELTA method")
            reflection_template = get_template("CoT-SENT-RF-REFLECT")
            apply_template = get_template("CoT-SENT-RF-HDELTA-APPLY")
            prompt_hist = int(self.config.get("prompt_history_points", 30))
            prompt_sent = int(self.config.get("prompt_sentiment_points", 30))
            prompt = reflection_template.format(
                examples=teaching_examples,
                pred_len=pred_len,
                history_points=prompt_hist,
                sentiment_points=prompt_sent,
                exogenous_summary=exogenous_summary,
                currency=self.currency,
            )
        elif method == "TSM+LLM-COT-SENT-RF-HPRICE":
            if tsm_forecast is None:
                raise ValueError("TSM forecast required for CoT-SENT-RF-HPRICE method")
            if not teaching_examples:
                raise ValueError("Teaching examples required for CoT-SENT-RF-HPRICE method")
            if sentiment_history is None:
                raise ValueError("Sentiment history required for CoT-SENT-RF-HPRICE method")
            cot_rf_cfg = self.config.get("cot_rf", {}) or {}
            strict_json_prompt = bool(cot_rf_cfg.get("strict_json_prompt", False))
            if strict_json_prompt:
                reflection_template = get_template("CoT-SENT-RF-REFLECT-STRICTJSON")
            else:
                reflection_template = get_template("CoT-SENT-RF-REFLECT")
            apply_template = get_template("CoT-SENT-RF-HPRICE-APPLY")
            prompt_hist = int(self.config.get("prompt_history_points", 30))
            prompt_sent = int(self.config.get("prompt_sentiment_points", 30))
            prompt = reflection_template.format(
                examples=teaching_examples,
                pred_len=pred_len,
                history_points=prompt_hist,
                sentiment_points=prompt_sent,
                exogenous_summary=exogenous_summary,
                currency=self.currency,
            )
        elif method == "CoT-RF":
            # First get initial forecast
            initial_result, _ = self.refine(
                "CoT", history, dates, pred_len=pred_len,
                exogenous_summary=exogenous_summary
            )
            if initial_result is None:
                return None, {"error": "Initial forecast failed"}
            
            prompt = template.format(
                history=history,
                dates=dates,
                initial_forecast=initial_result,
                pred_len=pred_len,
                currency=self.currency,
            )
        else:
            prompt = template.format(
                history=history,
                dates=dates,
                pred_len=pred_len,
                exogenous_summary=exogenous_summary,
                currency=self.currency,
            )

        # Special handling: two-stage reflect→apply methods
        if method in (
            "TSM+LLM-COT-RF",
            "TSM+LLM-COT-RF-HDELTA",
            "TSM+LLM-COT-SENT-RF",
            "TSM+LLM-COT-SENT-RF-DELTA",
            "TSM+LLM-COT-SENT-RF-HDELTA",
            "TSM+LLM-COT-SENT-RF-HPRICE",
        ):
            cot_rf_cfg = self.config.get("cot_rf", {}) or {}
            hdelta_case_controls: Dict = {}
            hdelta_structured_guidance: Optional[Dict[int, Dict[str, str]]] = None
            strict_json_prompt = bool(cot_rf_cfg.get("strict_json_prompt", False)) and (
                method in (
                    "TSM+LLM-COT-RF",
                    "TSM+LLM-COT-RF-HDELTA",
                    "TSM+LLM-COT-SENT-RF",
                    "TSM+LLM-COT-SENT-RF-HDELTA",
                    "TSM+LLM-COT-SENT-RF-HPRICE",
                )
            )
            retain_context = bool(
                cot_rf_cfg.get("retain_context", False)
            )
            strict_json_response_format = bool(cot_rf_cfg.get("strict_json_response_format", True))
            structured_hdelta_reflection = bool(
                cot_rf_cfg.get("structured_horizon_reflection", False)
            ) and method in ("TSM+LLM-COT-RF-HDELTA", "TSM+LLM-COT-SENT-RF-HDELTA")
            reflect_response_format = (
                _strict_rules_response_format()
                if strict_json_prompt and strict_json_response_format
                else None
            )
            apply_response_format = (
                _strict_yhat_response_format(pred_len)
                if strict_json_prompt and strict_json_response_format and method in ("TSM+LLM-COT-RF", "TSM+LLM-COT-SENT-RF")
                else None
            )
            if method in ("TSM+LLM-COT-RF-HDELTA", "TSM+LLM-COT-SENT-RF-HDELTA"):
                hdelta_cfg = self.config.get("hdelta", {}) or {}
                key_horizons = tuple(
                    int(h) for h in hdelta_cfg.get("key_horizons", HDELTA_KEY_HORIZONS)
                    if 1 <= int(h) <= int(pred_len)
                )
                frozen_horizons = tuple(
                    int(h) for h in hdelta_cfg.get("freeze_horizons", [])
                    if int(h) in key_horizons
                )
                if structured_hdelta_reflection and strict_json_prompt:
                    reflection_template = get_template(
                        "CoT-RF-HDELTA-REFLECT-STRUCTURED-STRICTJSON"
                    )
                if structured_hdelta_reflection and strict_json_prompt and strict_json_response_format:
                    reflect_response_format = _strict_hdelta_reflection_response_format(
                        key_horizons
                    )
                apply_response_format = _strict_hdelta_response_format(key_horizons)
                if bool(hdelta_cfg.get("case_conditioned", False)):
                    hdelta_case_controls = derive_hdelta_case_controls(
                        history=history,
                        forecast=tsm_forecast,
                        teaching_examples=teaching_examples or [],
                        key_horizons=key_horizons,
                        config=hdelta_cfg,
                        currency=self.currency,
                    )
                    dynamic_frozen = tuple(
                        int(h)
                        for h in hdelta_case_controls.get("dynamic_freeze_horizons", [])
                        if int(h) in key_horizons
                    )
                    frozen_horizons = tuple(sorted(set(frozen_horizons).union(dynamic_frozen)))
                    prompt_hist = int(self.config.get("prompt_history_points", 30))
                    reflect_examples = hdelta_case_controls.get("matched_examples") or list(teaching_examples or [])
                    prompt = reflection_template.format(
                        examples=reflect_examples,
                        pred_len=pred_len,
                        history_points=prompt_hist,
                        exogenous_summary=exogenous_summary,
                        current_case_summary=hdelta_case_controls.get("current_case_summary"),
                        key_horizons=list(key_horizons),
                        reasoning_style=cot_rf_cfg.get("reasoning_style", "default"),
                        compact_reasoning=bool(cot_rf_cfg.get("compact_reasoning", False)),
                        reflection_memory=cot_rf_cfg.get("reflection_memory"),
                        currency=self.currency,
                    )
            if method == "TSM+LLM-COT-SENT-RF-HPRICE":
                hprice_cfg = self.config.get("hprice", {}) or {}
                key_horizons = tuple(
                    int(h) for h in hprice_cfg.get("key_horizons", HPRICE_KEY_HORIZONS)
                    if 1 <= int(h) <= int(pred_len)
                )
                frozen_horizons = tuple(
                    int(h) for h in hprice_cfg.get("freeze_horizons", [])
                    if int(h) in key_horizons
                )
                apply_response_format = _strict_hprice_response_format(key_horizons)

            # Stage A: reflection → rules_text (plain text)
            reflect_system = reflection_template.get_system_message()
            reflect_samples = 1
            reflect_aggregation = "single"
            reflect_temperature = float(cot_rf_cfg.get("reflect_temperature", self.temperature))
            reflect_max_tokens = int(cot_rf_cfg.get("reflect_max_tokens", self.max_tokens))
            reflect_uncertainty_routed = False
            reflect_uncertainty_diagnostics: Dict[str, float] = {}
            if structured_hdelta_reflection:
                reflect_aggregation = str(
                    cot_rf_cfg.get("reflect_aggregation", "conservative_majority")
                ).strip().lower()
                reflect_samples, reflect_uncertainty_routed, reflect_uncertainty_diagnostics = (
                    select_structured_reflect_sample_budget(
                        cot_rf_cfg=cot_rf_cfg,
                        hdelta_case_controls=hdelta_case_controls,
                        frozen_horizons=frozen_horizons,
                    )
                )
            rules_text = None
            reflect_response = ""
            reflect_error = None
            reflect_attempts = 0
            reflect_valid_samples = 0
            reflect_sample_responses: List[str] = []
            reflect_prompt_current = prompt
            reflect_llm_debug: Dict[str, object] = {}
            reflect_response_format_current = reflect_response_format
            reflect_response_format_fallback = False
            for reflect_attempt in range(self.max_retries):
                reflect_attempts = reflect_attempt + 1
                try:
                    reflect_model = cot_rf_cfg.get("reflect_model")
                    reflect_api_key = cot_rf_cfg.get("reflect_api_key")
                    reflect_base_url = cot_rf_cfg.get("reflect_base_url")
                    reflect_response = self._call_llm(
                        reflect_prompt_current,
                        reflect_system,
                        response_format=reflect_response_format_current,
                        model_override=str(reflect_model) if reflect_model else None,
                        api_key_override=str(reflect_api_key) if reflect_api_key else None,
                        base_url_override=str(reflect_base_url) if reflect_base_url else None,
                        temperature_override=reflect_temperature,
                        max_tokens_override=reflect_max_tokens,
                    )
                    reflect_llm_debug = dict(self._last_llm_response)
                    if structured_hdelta_reflection and reflect_samples > 1:
                        sampled_guidance: List[Dict[int, Dict[str, str]]] = []
                        sampled_responses = []
                        first_response = reflect_response
                        first_guidance = parse_hdelta_reflection_guidance(
                            (first_response or "").strip(),
                            key_horizons=key_horizons,
                        )
                        if first_guidance is not None:
                            sampled_guidance.append(first_guidance)
                            sampled_responses.append(first_response)
                        for sample_idx in range(1, reflect_samples):
                            candidate_response = self._call_llm(
                                reflect_prompt_current,
                                reflect_system,
                                response_format=reflect_response_format_current,
                                model_override=str(reflect_model) if reflect_model else None,
                                api_key_override=str(reflect_api_key) if reflect_api_key else None,
                                base_url_override=str(reflect_base_url) if reflect_base_url else None,
                                temperature_override=reflect_temperature,
                                cache_key_suffix=f"reflect_sample_{sample_idx}",
                                max_tokens_override=reflect_max_tokens,
                            )
                            reflect_llm_debug = dict(self._last_llm_response)
                            candidate_guidance = parse_hdelta_reflection_guidance(
                                (candidate_response or "").strip(),
                                key_horizons=key_horizons,
                            )
                            if candidate_guidance is not None:
                                sampled_guidance.append(candidate_guidance)
                                sampled_responses.append(candidate_response)
                        reflect_valid_samples = len(sampled_guidance)
                        reflect_sample_responses = list(sampled_responses)
                        if sampled_guidance:
                            hdelta_structured_guidance = aggregate_structured_hdelta_guidance(
                                sampled_guidance,
                                key_horizons=key_horizons,
                                mode=reflect_aggregation,
                            )
                            rules_text = hdelta_guidance_to_rules_text(
                                hdelta_structured_guidance
                            )
                            reflect_response = json.dumps(
                                {
                                    "aggregation": reflect_aggregation,
                                    "n_valid": reflect_valid_samples,
                                    "samples": sampled_responses,
                                    "aggregated_guidance": hdelta_structured_guidance,
                                },
                                ensure_ascii=False,
                            )
                    else:
                        raw_reflect = (reflect_response or "").strip()
                        if strict_json_prompt:
                            if structured_hdelta_reflection:
                                hdelta_structured_guidance = parse_hdelta_reflection_guidance(
                                    raw_reflect,
                                    key_horizons=key_horizons,
                                )
                                reflect_valid_samples = 1 if hdelta_structured_guidance else 0
                                reflect_sample_responses = [raw_reflect] if raw_reflect else []
                                rules_text = hdelta_guidance_to_rules_text(
                                    hdelta_structured_guidance
                                )
                            else:
                                rules_text = parse_reflection_rules(raw_reflect)
                        else:
                            rules_text = raw_reflect
                    if rules_text:
                        break
                    if reflect_attempt < self.max_retries - 1 and strict_json_prompt:
                        reflect_prompt_current += (
                            "\n\nIMPORTANT: Put the final JSON object in assistant content. "
                            "Keep internal reasoning internal and do not leave content empty."
                        )
                except Exception as e:
                    reflect_error = str(e)
                    if (
                        reflect_response_format_current is not None
                        and _should_retry_without_response_format(reflect_error)
                    ):
                        self._mark_response_format_unsupported(
                            model_override=str(reflect_model) if reflect_model else None,
                            base_url_override=str(reflect_base_url) if reflect_base_url else None,
                        )
                        reflect_response_format_current = None
                        reflect_response_format_fallback = True
                        logger.warning(
                            "LLM reflection response_format rejected; retrying without response_format."
                        )
                        continue
                    logger.warning(
                        "LLM reflection call failed (attempt %d): %s",
                        reflect_attempt + 1,
                        e,
                    )

            if self.log_store:
                self.log_store.log_call(
                    prompt=prompt,
                    response={"content": reflect_response},
                    model=self.model,
                    temperature=self.temperature,
                    method=f"{method}:reflect",
                    metadata={
                        "success": bool(rules_text),
                        "attempts": reflect_attempts,
                        "error": reflect_error,
                        "retain_context": retain_context,
                        "strict_json_prompt": strict_json_prompt,
                        "strict_json_response_format": strict_json_response_format,
                        "structured_horizon_reflection": structured_hdelta_reflection,
                        "response_format_fallback": reflect_response_format_fallback,
                        "teaching_examples": len(teaching_examples or []),
                        "matched_teaching_dates": hdelta_case_controls.get("matched_dates"),
                        "dynamic_frozen_horizons": hdelta_case_controls.get("dynamic_freeze_horizons"),
                        "structured_horizon_guidance": hdelta_structured_guidance,
                        "reflect_samples": reflect_samples,
                        "reflect_valid_samples": reflect_valid_samples,
                        "reflect_aggregation": reflect_aggregation,
                        "reflect_temperature": reflect_temperature,
                        "reflect_max_tokens": reflect_max_tokens,
                        "reflect_uncertainty_routed": reflect_uncertainty_routed,
                        "reflect_uncertainty_diagnostics": reflect_uncertainty_diagnostics,
                        "reflect_model": cot_rf_cfg.get("reflect_model", self.model),
                        "reflect_base_url": cot_rf_cfg.get("reflect_base_url", self.base_url),
                        "llm_finish_reason": reflect_llm_debug.get("finish_reason"),
                        "llm_response_model": reflect_llm_debug.get("response_model"),
                        "llm_reasoning_present": bool(reflect_llm_debug.get("reasoning_content")),
                        "llm_used_reasoning_fallback": bool(reflect_llm_debug.get("used_reasoning_fallback")),
                    },
                )

            if not rules_text:
                metadata = {
                    "method": method,
                    "model": self.model,
                    "temperature": self.temperature,
                    "success": False,
                    "reflect_attempts": reflect_attempts,
                    "apply_attempts": 0,
                    "retain_context": retain_context,
                    "strict_json_prompt": strict_json_prompt,
                    "strict_json_response_format": strict_json_response_format,
                    "structured_horizon_reflection": structured_hdelta_reflection,
                    "response_format_fallback": reflect_response_format_fallback,
                    "error": reflect_error or "Empty rules_text from reflection stage",
                    "reflect_samples": reflect_samples,
                    "reflect_valid_samples": reflect_valid_samples,
                    "reflect_aggregation": reflect_aggregation,
                    "reflect_temperature": reflect_temperature,
                    "reflect_max_tokens": reflect_max_tokens,
                    "reflect_uncertainty_routed": reflect_uncertainty_routed,
                    "reflect_uncertainty_diagnostics": reflect_uncertainty_diagnostics,
                    "llm_finish_reason": reflect_llm_debug.get("finish_reason"),
                    "llm_response_model": reflect_llm_debug.get("response_model"),
                    "llm_reasoning_present": bool(reflect_llm_debug.get("reasoning_content")),
                    "llm_used_reasoning_fallback": bool(reflect_llm_debug.get("used_reasoning_fallback")),
                }
                return None, metadata

            # Stage B: apply rules → refined forecast (JSON)
            prompt_hist = int(self.config.get("prompt_history_points", 30))
            prompt_sent = int(self.config.get("prompt_sentiment_points", 30))

            if method == "TSM+LLM-COT-RF":
                apply_prompt = apply_template.format(
                    history=history,
                    dates=dates,
                    tsm_forecast=tsm_forecast,
                    rules_text=rules_text,
                    pred_len=pred_len,
                    currency=self.currency,
                )
            else:
                apply_kwargs = {
                    "history": history,
                    "dates": dates,
                    "tsm_forecast": tsm_forecast,
                    "sentiment_history": sentiment_history,
                    "rules_text": rules_text,
                    "pred_len": pred_len,
                    "history_points": prompt_hist,
                    "sentiment_points": prompt_sent,
                    "exogenous_summary": exogenous_summary,
                    "currency": self.currency,
                }
                if method == "TSM+LLM-COT-SENT-RF-DELTA":
                    delta_cfg = self.config.get("delta", {})
                    max_std = float(delta_cfg.get("max_delta_std", 0.5))
                    max_abs = delta_cfg.get("max_delta_abs")
                    hist_window = history[-20:] if len(history) >= 20 else history
                    hist_std = float(np.std(hist_window)) if len(hist_window) else 0.0
                    max_delta = float(max_abs) if max_abs is not None else max_std * hist_std
                    apply_kwargs["max_delta"] = max_delta
                elif method in ("TSM+LLM-COT-RF-HDELTA", "TSM+LLM-COT-SENT-RF-HDELTA"):
                    hdelta_cfg = self.config.get("hdelta", {}) or {}
                    numeric_tool_enabled = bool(cot_rf_cfg.get("enable_numeric_tool", False))
                    case_retrieval_tool_enabled = bool(
                        cot_rf_cfg.get("enable_case_retrieval_tool", False)
                    )
                    delta_verifier_tool_enabled = bool(
                        cot_rf_cfg.get("enable_delta_verifier_tool", False)
                    )
                    market_microstructure_tool_enabled = bool(
                        cot_rf_cfg.get("enable_market_microstructure_tool", False)
                    )
                    freeze_counterexample_tool_enabled = bool(
                        cot_rf_cfg.get("enable_freeze_counterexample_tool", False)
                    )
                    apply_kwargs["key_horizons"] = list(key_horizons)
                    apply_kwargs["max_adjustment_pct"] = float(
                        hdelta_cfg.get("max_adjustment_pct", 3.0)
                    )
                    apply_kwargs["frozen_horizons"] = list(frozen_horizons)
                    apply_kwargs["per_horizon_max_adjustment_pct"] = hdelta_case_controls.get(
                        "per_horizon_max_adjustment_pct"
                    )
                    apply_kwargs["current_case_summary"] = hdelta_case_controls.get(
                        "current_case_summary"
                    )
                    apply_kwargs["exogenous_summary"] = (
                        None if market_microstructure_tool_enabled else exogenous_summary
                    )
                    apply_kwargs["matched_examples_summary"] = (
                        None
                        if case_retrieval_tool_enabled
                        else hdelta_case_controls.get("matched_examples_summary")
                    )
                    apply_kwargs["horizon_guidance_summary"] = (
                        None
                        if case_retrieval_tool_enabled
                        else hdelta_case_controls.get("horizon_guidance_summary")
                    )
                    apply_kwargs["structured_horizon_guidance"] = hdelta_structured_guidance
                    apply_kwargs["apply_style"] = cot_rf_cfg.get("apply_style", "default")
                    apply_kwargs["reflection_memory"] = cot_rf_cfg.get("reflection_memory")
                    apply_kwargs["numeric_tool_enabled"] = numeric_tool_enabled
                    apply_kwargs["numeric_tool_name"] = NUMERIC_ANALYSIS_TOOL_NAME
                    apply_kwargs["case_retrieval_tool_enabled"] = case_retrieval_tool_enabled
                    apply_kwargs["case_retrieval_tool_name"] = CASE_RETRIEVAL_TOOL_NAME
                    apply_kwargs["delta_verifier_tool_enabled"] = delta_verifier_tool_enabled
                    apply_kwargs["delta_verifier_tool_name"] = DELTA_VERIFIER_TOOL_NAME
                    apply_kwargs["market_microstructure_tool_enabled"] = market_microstructure_tool_enabled
                    apply_kwargs["market_microstructure_tool_name"] = MARKET_MICROSTRUCTURE_TOOL_NAME
                    apply_kwargs["freeze_counterexample_tool_enabled"] = freeze_counterexample_tool_enabled
                    apply_kwargs["freeze_counterexample_tool_name"] = COUNTEREXAMPLE_TOOL_NAME
                    if method == "TSM+LLM-COT-SENT-RF-HDELTA":
                        apply_kwargs["sentiment_secondary"] = bool(
                            hdelta_cfg.get("sentiment_secondary", False)
                        )
                elif method == "TSM+LLM-COT-SENT-RF-HPRICE":
                    hprice_cfg = self.config.get("hprice", {}) or {}
                    apply_kwargs["key_horizons"] = list(key_horizons)
                    apply_kwargs["max_adjustment_pct"] = float(
                        hprice_cfg.get("max_adjustment_pct", 1.0)
                    )
                    apply_kwargs["frozen_horizons"] = list(frozen_horizons)
                    apply_kwargs["sentiment_secondary"] = bool(
                        hprice_cfg.get("sentiment_secondary", False)
                    )

                apply_prompt = apply_template.format(**apply_kwargs)

            apply_system = apply_template.get_system_message()
            forecast = None
            last_error = None
            apply_response = ""
            apply_prompt_for_log = apply_prompt
            apply_attempts = 0
            apply_response_format_current = apply_response_format
            apply_response_format_fallback = False
            apply_samples = 1
            apply_aggregation = "single"
            apply_temperature = float(self.temperature)
            apply_max_tokens = int(cot_rf_cfg.get("apply_max_tokens", self.max_tokens))
            apply_llm_debug: Dict[str, object] = {}
            if method in ("TSM+LLM-COT-RF-HDELTA", "TSM+LLM-COT-SENT-RF-HDELTA"):
                apply_samples = max(1, int(cot_rf_cfg.get("apply_samples", 1)))
                apply_aggregation = str(cot_rf_cfg.get("apply_aggregation", "median")).strip().lower()
                apply_temperature = float(cot_rf_cfg.get("apply_temperature", self.temperature))
            numeric_tool_enabled = bool(
                method in ("TSM+LLM-COT-RF-HDELTA", "TSM+LLM-COT-SENT-RF-HDELTA")
                and cot_rf_cfg.get("enable_numeric_tool", False)
            )
            case_retrieval_tool_enabled = bool(
                method in ("TSM+LLM-COT-RF-HDELTA", "TSM+LLM-COT-SENT-RF-HDELTA")
                and cot_rf_cfg.get("enable_case_retrieval_tool", False)
            )
            delta_verifier_tool_enabled = bool(
                method in ("TSM+LLM-COT-RF-HDELTA", "TSM+LLM-COT-SENT-RF-HDELTA")
                and cot_rf_cfg.get("enable_delta_verifier_tool", False)
            )
            market_microstructure_tool_enabled = bool(
                method in ("TSM+LLM-COT-RF-HDELTA", "TSM+LLM-COT-SENT-RF-HDELTA")
                and cot_rf_cfg.get("enable_market_microstructure_tool", False)
            )
            freeze_counterexample_tool_enabled = bool(
                method in ("TSM+LLM-COT-RF-HDELTA", "TSM+LLM-COT-SENT-RF-HDELTA")
                and cot_rf_cfg.get("enable_freeze_counterexample_tool", False)
            )
            numeric_tool_force = bool(cot_rf_cfg.get("force_numeric_tool", numeric_tool_enabled))
            case_retrieval_tool_force = bool(
                cot_rf_cfg.get("force_case_retrieval_tool", case_retrieval_tool_enabled)
            )
            delta_verifier_tool_force = bool(
                cot_rf_cfg.get("force_delta_verifier_tool", delta_verifier_tool_enabled)
            )
            market_microstructure_tool_force = bool(
                cot_rf_cfg.get("force_market_microstructure_tool", market_microstructure_tool_enabled)
            )
            freeze_counterexample_tool_force = bool(
                cot_rf_cfg.get("force_freeze_counterexample_tool", freeze_counterexample_tool_enabled)
            )
            hdelta_tool_specs: List[Dict[str, Any]] = []
            if numeric_tool_enabled:
                hdelta_tool_specs.append(
                    build_numeric_analysis_tool_spec(tool_name=NUMERIC_ANALYSIS_TOOL_NAME)
                )
            if market_microstructure_tool_enabled:
                hdelta_tool_specs.append(
                    build_market_microstructure_tool_spec(
                        tool_name=MARKET_MICROSTRUCTURE_TOOL_NAME
                    )
                )
            if freeze_counterexample_tool_enabled:
                hdelta_tool_specs.append(
                    build_freeze_counterexample_tool_spec(
                        tool_name=COUNTEREXAMPLE_TOOL_NAME
                    )
                )
            if case_retrieval_tool_enabled:
                hdelta_tool_specs.append(
                    build_structured_case_retrieval_tool_spec(
                        tool_name=CASE_RETRIEVAL_TOOL_NAME
                    )
                )
            if delta_verifier_tool_enabled:
                hdelta_tool_specs.append(
                    build_hdelta_delta_verifier_tool_spec(
                        tool_name=DELTA_VERIFIER_TOOL_NAME,
                        key_horizons=key_horizons,
                    )
                )
            forced_tool_names = [
                name
                for name, enabled in (
                    (NUMERIC_ANALYSIS_TOOL_NAME, numeric_tool_force and numeric_tool_enabled),
                    (MARKET_MICROSTRUCTURE_TOOL_NAME, market_microstructure_tool_force and market_microstructure_tool_enabled),
                    (COUNTEREXAMPLE_TOOL_NAME, freeze_counterexample_tool_force and freeze_counterexample_tool_enabled),
                    (CASE_RETRIEVAL_TOOL_NAME, case_retrieval_tool_force and case_retrieval_tool_enabled),
                    (DELTA_VERIFIER_TOOL_NAME, delta_verifier_tool_force and delta_verifier_tool_enabled),
                )
                if enabled
            ]
            hdelta_tool_choice: Optional[Union[str, Dict[str, Any]]] = (
                "required" if forced_tool_names else "auto"
            )
            hdelta_max_tool_rounds = max(
                2,
                int(cot_rf_cfg.get("max_tool_rounds", len(hdelta_tool_specs) + 1 if hdelta_tool_specs else 2)),
            )

            for apply_attempt in range(self.max_retries):
                apply_attempts = apply_attempt + 1
                try:
                    def _apply_once(sample_idx: int = 0) -> str:
                        cache_suffix = None if apply_samples <= 1 else f"apply_sample_{sample_idx}"
                        any_hdelta_tools_enabled = bool(hdelta_tool_specs)
                        if any_hdelta_tools_enabled:
                            if retain_context:
                                messages: List[Dict[str, Any]] = [
                                    {"role": "user", "content": prompt},
                                    {"role": "assistant", "content": rules_text},
                                    {"role": "user", "content": apply_prompt},
                                ]
                            else:
                                messages = [{"role": "user", "content": apply_prompt}]

                            def _tool_executor(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
                                if not isinstance(arguments, dict):
                                    arguments = {}
                                if tool_name == NUMERIC_ANALYSIS_TOOL_NAME:
                                    requested = arguments.get("requested_horizons") or []
                                    if requested:
                                        requested_horizons = [
                                            int(h)
                                            for h in requested
                                            if str(h).strip().lstrip("-").isdigit() and int(h) in key_horizons
                                        ]
                                    else:
                                        requested_horizons = list(key_horizons)
                                    return build_numeric_analysis_payload(
                                        history=history,
                                        tsm_forecast=tsm_forecast,
                                        key_horizons=requested_horizons,
                                        per_horizon_max_adjustment_pct=hdelta_case_controls.get(
                                            "per_horizon_max_adjustment_pct"
                                        ),
                                        structured_horizon_guidance=hdelta_structured_guidance,
                                        current_case_summary=hdelta_case_controls.get("current_case_summary"),
                                        currency=self.currency,
                                    )
                                if tool_name == MARKET_MICROSTRUCTURE_TOOL_NAME:
                                    requested_features = arguments.get("requested_features") or []
                                    return build_market_microstructure_payload(
                                        exogenous_summary,
                                        current_case_summary=hdelta_case_controls.get("current_case_summary"),
                                        requested_features=requested_features,
                                    )
                                if tool_name == COUNTEREXAMPLE_TOOL_NAME:
                                    return derive_hdelta_freeze_counterexample(
                                        history=history,
                                        forecast=tsm_forecast,
                                        teaching_examples=teaching_examples or [],
                                        key_horizons=key_horizons,
                                        quantile=float(cot_rf_cfg.get("freeze_counterexample_quantile", 0.35)),
                                    )
                                if tool_name == CASE_RETRIEVAL_TOOL_NAME:
                                    requested = arguments.get("requested_horizons") or []
                                    if requested:
                                        requested_horizons = [
                                            int(h)
                                            for h in requested
                                            if str(h).strip().lstrip("-").isdigit() and int(h) in key_horizons
                                        ]
                                    else:
                                        requested_horizons = list(key_horizons)
                                    max_examples = arguments.get("max_examples")
                                    return build_structured_case_retrieval_payload(
                                        hdelta_case_controls,
                                        key_horizons,
                                        requested_horizons=requested_horizons,
                                        max_examples=max_examples,
                                    )
                                if tool_name == DELTA_VERIFIER_TOOL_NAME:
                                    return build_hdelta_delta_verification_payload(
                                        arguments.get("proposed_adjustments"),
                                        key_horizons,
                                        per_horizon_max_adjustment_pct=hdelta_case_controls.get(
                                            "per_horizon_max_adjustment_pct"
                                        ),
                                        structured_horizon_guidance=hdelta_structured_guidance,
                                        frozen_horizons=frozen_horizons,
                                        config=hdelta_cfg,
                                    )
                                raise ValueError(f"Unsupported tool requested: {tool_name}")

                            return self._call_llm_messages_with_tools(
                                messages,
                                tools=hdelta_tool_specs,
                                tool_executor=_tool_executor,
                                system_message=apply_system,
                                temperature_override=apply_temperature,
                                max_tokens_override=apply_max_tokens,
                                tool_choice=hdelta_tool_choice,
                                max_tool_rounds=hdelta_max_tool_rounds,
                                required_tool_names=forced_tool_names,
                            )
                        if retain_context:
                            messages = [
                                {"role": "user", "content": prompt},
                                {"role": "assistant", "content": rules_text},
                                {"role": "user", "content": apply_prompt},
                            ]
                            return self._call_llm_messages(
                                messages,
                                apply_system,
                                response_format=apply_response_format_current,
                                temperature_override=apply_temperature,
                                cache_key_suffix=cache_suffix,
                                max_tokens_override=apply_max_tokens,
                            )
                        return self._call_llm(
                            apply_prompt,
                            apply_system,
                            response_format=apply_response_format_current,
                            temperature_override=apply_temperature,
                            cache_key_suffix=cache_suffix,
                            max_tokens_override=apply_max_tokens,
                        )

                    if retain_context:
                        apply_prompt_for_log = json.dumps(
                            {
                                "system": apply_system,
                                "messages": [
                                    {"role": "user", "content": prompt},
                                    {"role": "assistant", "content": rules_text},
                                    {"role": "user", "content": apply_prompt},
                                ],
                            },
                            sort_keys=True,
                            ensure_ascii=False,
                        )
                    else:
                        apply_prompt_for_log = apply_prompt

                    if method == "TSM+LLM-COT-SENT-RF-DELTA":
                        apply_response = _apply_once()
                        apply_llm_debug = dict(self._last_llm_response)
                        delta = parse_json_array(
                            apply_response,
                            expected_len=pred_len,
                            keys=("delta", "delta_price", "adjustment"),
                            allow_bare_array=False,
                        )
                        if delta is not None:
                            max_delta = apply_kwargs.get("max_delta", 0.0)
                            if max_delta and max_delta > 0:
                                delta = np.clip(delta, -max_delta, max_delta)
                            refined = tsm_forecast + delta
                            forecast = self._blend_forecast(tsm_forecast, refined)
                    elif method in ("TSM+LLM-COT-RF-HDELTA", "TSM+LLM-COT-SENT-RF-HDELTA"):
                        sampled_adjustments: List[Dict[int, float]] = []
                        sampled_responses: List[str] = []
                        for sample_idx in range(apply_samples):
                            candidate_response = _apply_once(sample_idx)
                            apply_llm_debug = dict(self._last_llm_response)
                            candidate_adjustments = parse_horizon_deltas(
                                candidate_response,
                                key_horizons=key_horizons,
                            )
                            if candidate_adjustments is not None:
                                sampled_adjustments.append(candidate_adjustments)
                                sampled_responses.append(candidate_response)
                        adjustments_pct = aggregate_horizon_adjustments(
                            sampled_adjustments,
                            key_horizons=key_horizons,
                            mode=apply_aggregation,
                        )
                        if sampled_responses:
                            apply_response = json.dumps(
                                {
                                    "aggregation": apply_aggregation,
                                    "n_valid": len(sampled_responses),
                                    "samples": sampled_responses,
                                    "aggregated_adjustments": adjustments_pct,
                                },
                                ensure_ascii=False,
                            )
                        if adjustments_pct is not None:
                            hdelta_cfg = self.config.get("hdelta", {}) or {}
                            max_adjustment_pct = float(hdelta_cfg.get("max_adjustment_pct", 3.0))
                            per_horizon_max = {
                                int(h): float(
                                    hdelta_case_controls.get(
                                        "per_horizon_max_adjustment_pct",
                                        {},
                                    ).get(int(h), max_adjustment_pct)
                                )
                                for h in key_horizons
                            }
                            clipped = {
                                int(h): float(
                                    np.clip(
                                        v,
                                        -per_horizon_max.get(int(h), max_adjustment_pct),
                                        per_horizon_max.get(int(h), max_adjustment_pct),
                                    )
                                )
                                for h, v in adjustments_pct.items()
                            }
                            clipped = enforce_structured_hdelta_adjustments(
                                clipped,
                                structured_guidance=hdelta_structured_guidance,
                                per_horizon_max=per_horizon_max,
                                config=hdelta_cfg,
                            )
                            clipped = apply_structured_hdelta_coherence_guards(
                                clipped,
                                config=hdelta_cfg,
                            )
                            for h in frozen_horizons:
                                clipped[int(h)] = 0.0
                            path_pct = interpolate_horizon_adjustments(pred_len, clipped)
                            forecast = np.asarray(tsm_forecast, dtype=float) * (1.0 + path_pct / 100.0)
                    elif method == "TSM+LLM-COT-SENT-RF-HPRICE":
                        apply_response = _apply_once()
                        apply_llm_debug = dict(self._last_llm_response)
                        anchor_prices = parse_horizon_anchor_prices(
                            apply_response,
                            key_horizons=key_horizons,
                        )
                        if anchor_prices is not None:
                            hprice_cfg = self.config.get("hprice", {}) or {}
                            max_adjustment_pct = float(hprice_cfg.get("max_adjustment_pct", 1.0))
                            base_forecast = np.asarray(tsm_forecast, dtype=float)
                            clipped_anchors: Dict[int, float] = {}
                            for h, anchor in anchor_prices.items():
                                base_price = float(base_forecast[int(h) - 1])
                                if int(h) in frozen_horizons:
                                    clipped_anchors[int(h)] = base_price
                                    continue
                                lower = base_price * (1.0 - max_adjustment_pct / 100.0)
                                upper = base_price * (1.0 + max_adjustment_pct / 100.0)
                                clipped_anchors[int(h)] = float(np.clip(anchor, lower, upper))
                            forecast = interpolate_horizon_anchor_prices(pred_len, clipped_anchors)
                    else:
                        apply_response = _apply_once()
                        apply_llm_debug = dict(self._last_llm_response)
                        forecast = parse_json_forecast(apply_response, expected_len=pred_len)

                    if forecast is not None:
                        break

                    if apply_attempt < self.max_retries - 1:
                        apply_prompt += "\n\nIMPORTANT: Respond ONLY with valid JSON. No other text."

                except Exception as e:
                    last_error = str(e)
                    if (
                        apply_response_format_current is not None
                        and _should_retry_without_response_format(last_error)
                    ):
                        self._mark_response_format_unsupported()
                        apply_response_format_current = None
                        apply_response_format_fallback = True
                        logger.warning(
                            "LLM apply response_format rejected; retrying without response_format."
                        )
                        continue
                    logger.warning(
                        "LLM apply call failed (attempt %d): %s",
                        apply_attempt + 1,
                        e,
                    )

            metadata = {
                "method": method,
                "model": self.model,
                "temperature": self.temperature,
                "success": forecast is not None,
                "reflect_attempts": reflect_attempts,
                "apply_attempts": apply_attempts,
                "apply_samples": apply_samples,
                "apply_aggregation": apply_aggregation,
                "apply_temperature": apply_temperature,
                "apply_max_tokens": apply_max_tokens,
                "retain_context": retain_context,
                "strict_json_prompt": strict_json_prompt,
                "strict_json_response_format": strict_json_response_format,
                "structured_horizon_reflection": structured_hdelta_reflection,
                "response_format_fallback": (
                    reflect_response_format_fallback or apply_response_format_fallback
                ),
                "rules_text": rules_text,
                "teaching_examples": len(teaching_examples or []),
                "teaching_dates": [ex.get("date") for ex in (teaching_examples or [])],
                "matched_teaching_dates": hdelta_case_controls.get("matched_dates"),
                "dynamic_frozen_horizons": hdelta_case_controls.get("dynamic_freeze_horizons"),
                "per_horizon_max_adjustment_pct": hdelta_case_controls.get("per_horizon_max_adjustment_pct"),
                "structured_horizon_guidance": hdelta_structured_guidance,
                "reflect_samples": reflect_samples,
                "reflect_valid_samples": reflect_valid_samples,
                "reflect_aggregation": reflect_aggregation,
                "reflect_temperature": reflect_temperature,
                "reflect_max_tokens": reflect_max_tokens,
                "reflect_uncertainty_routed": reflect_uncertainty_routed,
                "reflect_uncertainty_diagnostics": reflect_uncertainty_diagnostics,
                "reflect_model": cot_rf_cfg.get("reflect_model", self.model),
                "reflect_base_url": cot_rf_cfg.get("reflect_base_url", self.base_url),
                "numeric_tool_enabled": numeric_tool_enabled,
                "market_microstructure_tool_enabled": market_microstructure_tool_enabled,
                "freeze_counterexample_tool_enabled": freeze_counterexample_tool_enabled,
                "case_retrieval_tool_enabled": case_retrieval_tool_enabled,
                "delta_verifier_tool_enabled": delta_verifier_tool_enabled,
                "tool_calls": apply_llm_debug.get("tool_calls", []),
                "tool_call_counts": dict(Counter(apply_llm_debug.get("tool_calls", []))),
                "numeric_tool_calls": apply_llm_debug.get("tool_calls", []),
                "numeric_tool_invocations": Counter(apply_llm_debug.get("tool_calls", [])).get(NUMERIC_ANALYSIS_TOOL_NAME, 0),
                "market_microstructure_tool_invocations": Counter(apply_llm_debug.get("tool_calls", [])).get(MARKET_MICROSTRUCTURE_TOOL_NAME, 0),
                "freeze_counterexample_tool_invocations": Counter(apply_llm_debug.get("tool_calls", [])).get(COUNTEREXAMPLE_TOOL_NAME, 0),
                "case_retrieval_tool_invocations": Counter(apply_llm_debug.get("tool_calls", [])).get(CASE_RETRIEVAL_TOOL_NAME, 0),
                "delta_verifier_tool_invocations": Counter(apply_llm_debug.get("tool_calls", [])).get(DELTA_VERIFIER_TOOL_NAME, 0),
                "llm_finish_reason": apply_llm_debug.get("finish_reason") or reflect_llm_debug.get("finish_reason"),
                "llm_response_model": apply_llm_debug.get("response_model") or reflect_llm_debug.get("response_model"),
                "llm_reasoning_present": bool(apply_llm_debug.get("reasoning_content") or reflect_llm_debug.get("reasoning_content")),
                "llm_used_reasoning_fallback": bool(apply_llm_debug.get("used_reasoning_fallback") or reflect_llm_debug.get("used_reasoning_fallback")),
            }
            if last_error:
                metadata["error"] = last_error

            if self.log_store:
                self.log_store.log_call(
                    prompt=apply_prompt_for_log,
                    response={"content": apply_response},
                    model=self.model,
                    temperature=self.temperature,
                    method=f"{method}:apply",
                    metadata=metadata,
                )

            return forecast, metadata

        system_message = template.get_system_message() if template else ""

        # Try with retries (single-call methods)
        forecast = None
        last_error = None
        response = ""

        for attempt in range(self.max_retries):
            try:
                response = self._call_llm(prompt, system_message)
                if method == "TSM+LLM-NORM-DELTA":
                    norm_cfg = self.config.get("norm_delta", {})
                    max_delta = float(norm_cfg.get("max_delta", 0.5))
                    window = int(norm_cfg.get("history_window", len(history)))
                    hist_window = history[-window:] if len(history) >= window else history
                    mean = float(np.mean(hist_window)) if len(hist_window) else 0.0
                    std = float(np.std(hist_window)) if len(hist_window) else 1.0
                    if std < 1e-6:
                        std = 1.0
                    delta = parse_json_array(
                        response,
                        expected_len=pred_len,
                        keys=("delta_z", "delta"),
                        allow_bare_array=False,
                    )
                    if delta is not None:
                        delta = np.clip(delta, -max_delta, max_delta)
                        norm_forecast = (tsm_forecast - mean) / std
                        refined_z = norm_forecast + delta
                        refined = refined_z * std + mean
                        forecast = self._blend_forecast(tsm_forecast, refined)
                else:
                    forecast = parse_json_forecast(response, expected_len=pred_len)

                if forecast is not None:
                    break

                if attempt < self.max_retries - 1:
                    prompt += "\n\nIMPORTANT: Respond ONLY with valid JSON. No other text."

            except Exception as e:
                last_error = str(e)
                logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}")

        metadata = {
            "method": method,
            "model": self.model,
            "temperature": self.temperature,
            "success": forecast is not None,
            "attempts": attempt + 1,
        }
        if last_error:
            metadata["error"] = last_error

        if self.log_store:
            self.log_store.log_call(
                prompt=prompt,
                response={"content": response},
                model=self.model,
                temperature=self.temperature,
                method=method,
                metadata=metadata,
            )

        return forecast, metadata
    
    def refine_batch(
        self,
        method: str,
        histories: np.ndarray,
        date_arrays: List[List[str]],
        tsm_forecasts: Optional[np.ndarray] = None,
        pred_len: int = 30,
        exogenous_summaries: Optional[List[Optional[Dict]]] = None,
        price_bases: Optional[np.ndarray] = None,
        teaching_examples: Optional[List[List[Dict]]] = None,
        sentiment_histories: Optional[List[np.ndarray]] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        resume_checkpoint_dir: Optional[Union[str, Path]] = None,
        sample_keys: Optional[Sequence[Union[str, int]]] = None,
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Refine forecasts for a batch of samples.
        
        Args:
            method: Prompt method
            histories: Historical values (batch, seq_len)
            date_arrays: Dates for each sample
            tsm_forecasts: TSM forecasts (batch, pred_len)
            pred_len: Prediction length
            
        Returns:
            Tuple of (forecasts array, metadata list)
        """
        batch_size = len(histories)
        forecasts = np.zeros((batch_size, pred_len))
        metadata_list: List[Optional[Dict]] = [None] * batch_size
        checkpoint_dir_path = Path(checkpoint_dir) if checkpoint_dir else None
        resume_dir_path = Path(resume_checkpoint_dir) if resume_checkpoint_dir else None
        if sample_keys is None:
            sample_keys = [f"{i:05d}" for i in range(batch_size)]
        else:
            sample_keys = [str(key) for key in sample_keys]
        resumed_count = 0

        def _checkpoint_path(root: Path, sample_key: str) -> Path:
            safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample_key))
            return root / f"{safe_key}.json"

        def _json_safe(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, (np.floating,)):
                return float(value)
            if isinstance(value, (np.integer,)):
                return int(value)
            if isinstance(value, (np.bool_,)):
                return bool(value)
            if isinstance(value, dict):
                return {str(k): _json_safe(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_json_safe(v) for v in value]
            return value

        def _load_checkpoint(root: Optional[Path], sample_key: str):
            if root is None:
                return None
            path = _checkpoint_path(root, sample_key)
            if not path.exists():
                return None
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                forecast = np.asarray(payload.get("forecast", []), dtype=float)
                if forecast.shape != (pred_len,):
                    raise ValueError(
                        f"checkpoint forecast shape {forecast.shape} != {(pred_len,)}"
                    )
                metadata = payload.get("metadata", {}) or {}
                metadata = dict(metadata)
                metadata["resumed_from_checkpoint"] = True
                metadata["checkpoint_path"] = str(path)
                return forecast, metadata
            except Exception as e:
                logger.warning("Failed to load LLM batch checkpoint %s: %s", path, e)
                return None

        def _save_checkpoint(root: Optional[Path], sample_key: str, forecast: np.ndarray, metadata: Dict):
            if root is None:
                return
            path = _checkpoint_path(root, sample_key)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "sample_key": str(sample_key),
                "forecast": np.asarray(forecast, dtype=float).tolist(),
                "metadata": _json_safe(metadata),
            }
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            tmp_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
            tmp_path.replace(path)
        
        for i in range(batch_size):
            sample_key = str(sample_keys[i])
            loaded = _load_checkpoint(checkpoint_dir_path, sample_key)
            if loaded is None:
                loaded = _load_checkpoint(resume_dir_path, sample_key)
                if loaded is not None and checkpoint_dir_path is not None:
                    _save_checkpoint(checkpoint_dir_path, sample_key, loaded[0], loaded[1])
            if loaded is not None:
                forecasts[i] = loaded[0]
                metadata_list[i] = loaded[1]
                resumed_count += 1
                continue

            tsm_forecast = tsm_forecasts[i] if tsm_forecasts is not None else None
            exo_summary = exogenous_summaries[i] if exogenous_summaries is not None else None
            price_base = price_bases[i] if price_bases is not None else None
            examples = teaching_examples[i] if teaching_examples is not None else None
            sentiment_history = (
                sentiment_histories[i] if sentiment_histories is not None else None
            )
            
            forecast, metadata = self.refine(
                method=method,
                history=histories[i],
                dates=date_arrays[i],
                tsm_forecast=tsm_forecast,
                pred_len=pred_len,
                exogenous_summary=exo_summary,
                price_base=price_base,
                teaching_examples=examples,
                sentiment_history=sentiment_history
            )
            
            if forecast is not None:
                forecasts[i] = forecast
            elif tsm_forecasts is not None:
                # Fallback to TSM forecast
                forecasts[i] = tsm_forecasts[i]
                metadata["fallback"] = True
            else:
                # Fallback to persistence
                forecasts[i] = np.full(pred_len, histories[i, -1])
                metadata["fallback"] = True
            metadata["sample_key"] = sample_key
            _save_checkpoint(checkpoint_dir_path, sample_key, forecasts[i], metadata)
            metadata_list[i] = metadata
            
            if (i + 1) % 10 == 0:
                logger.info(
                    "Processed %d/%d samples (%d resumed from checkpoint)",
                    i + 1,
                    batch_size,
                    resumed_count,
                )

        if resumed_count:
            logger.info(
                "Resumed %d/%d LLM samples from checkpoint",
                resumed_count,
                batch_size,
            )

        return forecasts, [metadata or {} for metadata in metadata_list]


def add_noise_to_forecast(
    forecast: np.ndarray,
    noise_level: float,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Add noise to a forecast for robustness testing.
    
    Args:
        forecast: Original forecast (batch, pred_len) or (pred_len,)
        noise_level: Noise level as fraction of std
        seed: Random seed
        
    Returns:
        Noisy forecast
    """
    if seed is not None:
        np.random.seed(seed)
    
    std = np.std(forecast, axis=-1, keepdims=True)
    noise = np.random.normal(0, noise_level * std, forecast.shape)
    
    return forecast + noise
