"""LLM-based sentiment labeling and daily aggregation."""

from __future__ import annotations

import json
import time
import os
import re
from collections import Counter
from datetime import datetime
from statistics import median
from typing import List, Dict, Optional, Tuple

import pandas as pd

try:
    # When `src/` is on sys.path (e.g. `python -m src.run_experiment`),
    # modules are imported as top-level packages: `llm`, `news`, etc.
    from llm.cache import ResponseCache  # type: ignore
except ModuleNotFoundError:
    # When importing as a package (e.g. `from src.news.sentiment import ...`).
    from src.llm.cache import ResponseCache  # type: ignore


LABEL_MAP = {"YES": 1, "NO": -1, "UNKNOWN": 0}


def _extract_openai_message_text(message: object) -> str:
    """Extract only final assistant output text (ignore reasoning traces)."""
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


def _build_prompt(
    headline: str,
    scoring_mode: str = "effect",
    votes: int = 1,
    vote_method: str = "per_call",
) -> str:
    if scoring_mode == "effect_importance":
        if vote_method == "single_call_multi_vote":
            return (
                "You are a financial expert with carbon trading experience.\n"
                f"Produce {votes} independent assessments for this headline.\n"
                "Return ONLY valid JSON with this schema:\n"
                "{\"votes\":[{\"direction\":\"YES|NO|UNKNOWN\",\"importance\":0-10}, ...]}\n"
                "Do not include explanations.\n"
                "The votes array length must exactly match the requested count.\n"
                f"Headline: {headline}\n"
            )
        return (
            "You are a financial expert with carbon trading experience.\n"
            "For the headline below, output EXACTLY 3 lines:\n"
            "Line 1: YES or NO or UNKNOWN (short-term direction for EU ETS price)\n"
            "Line 2: importance as an integer from 0 to 10 (0=no plausible impact, 10=major plausible impact)\n"
            "Line 3: one short rationale sentence\n"
            f"Headline: {headline}\n"
        )
    return (
        "Forget all your previous instructions. Pretend you are a financial expert.\n"
        "You are a financial expert with carbon trading experience. Answer \"YES\" if good news,\n"
        "\"NO\" if bad news, or \"UNKNOWN\" if uncertain in the first line. Then elaborate with one\n"
        "short and concise sentence on the next line. Is this headline good or bad for the EU ETS\n"
        "carbon price in the short term?\n"
        f"Headline: {headline}\n"
    )


def _call_openai(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    system_message: str = "You output exactly two lines: label then one-sentence rationale.",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout_seconds: Optional[float] = None,
) -> str:
    from openai import OpenAI

    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    if timeout_seconds is not None:
        kwargs["timeout"] = float(timeout_seconds)
    client = OpenAI(**kwargs)
    request = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_message,
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    }
    if model.startswith("gpt-5"):
        request["max_completion_tokens"] = max_tokens
    else:
        request["max_tokens"] = max_tokens
    response = client.chat.completions.create(**request)
    message = response.choices[0].message
    content = _extract_openai_message_text(message)
    return content.strip()


def _parse_label(response: str) -> Tuple[str, str]:
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    label = lines[0].upper() if lines else "UNKNOWN"
    rationale = lines[1] if len(lines) > 1 else ""
    if label not in LABEL_MAP:
        label = "UNKNOWN"
    return label, rationale


def _parse_label_importance(response: str) -> Tuple[str, int, str]:
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    if not lines:
        return "UNKNOWN", 0, ""

    label = lines[0].split()[0].upper()
    if label not in LABEL_MAP:
        label = "UNKNOWN"

    importance = 0
    if len(lines) > 1:
        match = re.search(r"-?\d+(\.\d+)?", lines[1])
        if match:
            try:
                importance = int(round(float(match.group(0))))
            except ValueError:
                importance = 0
    importance = max(0, min(10, importance))

    rationale = lines[2] if len(lines) > 2 else (lines[1] if len(lines) > 1 else "")
    return label, importance, rationale


def _extract_json_obj(text: str) -> Optional[dict]:
    s = (text or "").strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    match = re.search(r"\{.*\}", s, re.DOTALL)
    if not match:
        return None
    snippet = match.group(0)
    try:
        obj = json.loads(snippet)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _parse_multi_votes(
    response: str,
    expected_votes: int,
) -> Tuple[List[str], List[int], List[str]]:
    obj = _extract_json_obj(response)
    labels: List[str] = []
    importances: List[int] = []
    rationales: List[str] = []
    if obj and isinstance(obj.get("votes"), list):
        for item in obj["votes"]:
            if not isinstance(item, dict):
                continue
            label = str(item.get("direction", item.get("label", "UNKNOWN"))).upper()
            if label not in LABEL_MAP:
                label = "UNKNOWN"
            imp_raw = item.get("importance", 0)
            try:
                imp = int(round(float(imp_raw)))
            except Exception:
                imp = 0
            imp = max(0, min(10, imp))
            rationale = str(item.get("rationale", "")).strip()
            labels.append(label)
            importances.append(imp)
            rationales.append(rationale)
    if not labels:
        label, imp, rat = _parse_label_importance(response)
        labels = [label]
        importances = [imp]
        rationales = [rat]
    if len(labels) < expected_votes:
        pad_n = expected_votes - len(labels)
        labels.extend([labels[-1]] * pad_n)
        importances.extend([importances[-1]] * pad_n)
        rationales.extend([rationales[-1]] * pad_n)
    if len(labels) > expected_votes:
        labels = labels[:expected_votes]
        importances = importances[:expected_votes]
        rationales = rationales[:expected_votes]
    return labels, importances, rationales


def label_headlines_with_llm(
    headlines: pd.DataFrame,
    headline_col: str = "title",
    model: str = "gpt-5.2",
    temperature: float = 0.0,
    max_tokens: int = 80,
    votes: int = 3,
    cache_dir: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout_seconds: Optional[float] = None,
    sleep_seconds: float = 0.2,
    scoring_mode: str = "effect",
    vote_cache_mode: str = "shared",
    importance_confidence_power: float = 1.0,
    vote_method: str = "per_call",
) -> pd.DataFrame:
    cache = ResponseCache(cache_dir) if cache_dir else None
    records = []
    base_url_value = base_url or os.getenv("OPENAI_BASE_URL") or ""
    if scoring_mode not in ("effect", "effect_importance"):
        raise ValueError(f"Unknown scoring_mode: {scoring_mode}")
    if vote_cache_mode not in ("shared", "independent"):
        raise ValueError(f"Unknown vote_cache_mode: {vote_cache_mode}")
    if vote_method not in ("per_call", "single_call_multi_vote"):
        raise ValueError(f"Unknown vote_method: {vote_method}")
    if scoring_mode != "effect_importance" and vote_method != "per_call":
        raise ValueError("vote_method=single_call_multi_vote is only valid for scoring_mode=effect_importance")

    if scoring_mode == "effect_importance":
        if vote_method == "single_call_multi_vote":
            system_message = (
                "Return only valid JSON matching: "
                "{\"votes\":[{\"direction\":\"YES|NO|UNKNOWN\",\"importance\":0-10}]}"
            )
        else:
            system_message = (
                "You output exactly 3 lines: direction label, integer importance 0-10, one short rationale."
            )
    else:
        system_message = "You output exactly two lines: label then one-sentence rationale."

    cache_context = {
        "provider": "openai",
        "base_url": base_url_value,
        "system_message": system_message,
        "max_tokens": max_tokens,
        "scoring_mode": scoring_mode,
        "vote_method": vote_method,
        "votes": int(votes),
    }

    for _, row in headlines.iterrows():
        headline = str(row.get(headline_col, "")).strip()
        if not headline:
            continue
        prompt = _build_prompt(
            headline,
            scoring_mode=scoring_mode,
            votes=votes,
            vote_method=vote_method,
        )
        labels = []
        importances: List[int] = []
        rationales = []
        if scoring_mode == "effect_importance" and vote_method == "single_call_multi_vote":
            cached = (
                cache.get(prompt, model, temperature, context=cache_context)
                if cache
                else None
            )
            if cached:
                response = cached.get("content", "")
            else:
                response = _call_openai(
                    prompt,
                    model,
                    temperature,
                    max_tokens,
                    system_message=system_message,
                    api_key=api_key or os.getenv("OPENAI_API_KEY"),
                    base_url=base_url_value,
                    timeout_seconds=timeout_seconds,
                )
                if cache:
                    cache.set(
                        prompt,
                        model,
                        temperature,
                        {"content": response},
                        context=cache_context,
                    )
            labels, importances, rationales = _parse_multi_votes(response, expected_votes=votes)
            time.sleep(sleep_seconds)
        else:
            for vote_idx in range(votes):
                vote_context = dict(cache_context)
                if vote_cache_mode == "independent":
                    vote_context["vote_index"] = int(vote_idx)
                cached = (
                    cache.get(prompt, model, temperature, context=vote_context)
                    if cache
                    else None
                )
                if cached:
                    response = cached.get("content", "")
                else:
                    response = _call_openai(
                        prompt,
                        model,
                        temperature,
                        max_tokens,
                        system_message=system_message,
                        api_key=api_key or os.getenv("OPENAI_API_KEY"),
                        base_url=base_url_value,
                        timeout_seconds=timeout_seconds,
                    )
                    if cache:
                        cache.set(
                            prompt,
                            model,
                            temperature,
                            {"content": response},
                            context=vote_context,
                        )
                if scoring_mode == "effect_importance":
                    label, importance, rationale = _parse_label_importance(response)
                    importances.append(importance)
                else:
                    label, rationale = _parse_label(response)
                labels.append(label)
                rationales.append(rationale)
                time.sleep(sleep_seconds)

        vote_counts = Counter(labels)
        vote = vote_counts.most_common(1)[0][0]
        score = LABEL_MAP.get(vote, 0)
        record = dict(row)
        if scoring_mode == "effect_importance":
            majority_count = int(vote_counts.get(vote, 0))
            agree_frac = majority_count / max(1, votes)
            imp_pool = [imp for imp, lbl in zip(importances, labels) if lbl == vote]
            if not imp_pool:
                imp_pool = importances
            imp_med = int(round(median(imp_pool))) if imp_pool else 0
            imp_med = max(0, min(10, imp_med))
            imp_scale = imp_med / 10.0
            score_imp_nopen = float(score) * imp_scale
            score_imp = score_imp_nopen * (agree_frac ** float(max(0.0, importance_confidence_power)))
            record.update(
                {
                    "llm_vote": vote,
                    "llm_score": score,
                    "llm_importance": imp_med,
                    "llm_vote_agreement": agree_frac,
                    "llm_score_importance_nopen": score_imp_nopen,
                    "llm_score_importance": score_imp,
                    "llm_votes": json.dumps(labels),
                    "llm_importance_votes": json.dumps(importances),
                    "llm_rationales": json.dumps(rationales),
                }
            )
        else:
            record.update(
                {
                    "llm_vote": vote,
                    "llm_score": score,
                    "llm_votes": json.dumps(labels),
                    "llm_rationales": json.dumps(rationales),
                }
            )
        records.append(record)

    return pd.DataFrame(records)


def aggregate_daily_sentiment(
    labeled: pd.DataFrame,
    date_col: str = "seendate",
    score_col: str = "llm_score",
    output_col: str = "sent_score",
) -> pd.DataFrame:
    df = labeled.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.date
    grouped = df.groupby(date_col, as_index=False)[score_col].mean()
    grouped.rename(columns={score_col: output_col}, inplace=True)
    grouped[date_col] = pd.to_datetime(grouped[date_col])
    return grouped
