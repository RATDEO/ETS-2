"""LLM-based sentiment labeling and daily aggregation."""

from __future__ import annotations

import json
import time
import os
from collections import Counter
from datetime import datetime
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


def _build_prompt(headline: str) -> str:
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
                "content": (
                    "You output exactly two lines: label then one-sentence rationale."
                ),
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
    content = message.content or getattr(message, "reasoning_content", None) or ""
    return content.strip()


def _parse_label(response: str) -> Tuple[str, str]:
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    label = lines[0].upper() if lines else "UNKNOWN"
    rationale = lines[1] if len(lines) > 1 else ""
    if label not in LABEL_MAP:
        label = "UNKNOWN"
    return label, rationale


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
) -> pd.DataFrame:
    cache = ResponseCache(cache_dir) if cache_dir else None
    records = []
    base_url_value = base_url or os.getenv("OPENAI_BASE_URL") or ""
    cache_context = {
        "provider": "openai",
        "base_url": base_url_value,
        "system_message": "You output exactly two lines: label then one-sentence rationale.",
        "max_tokens": max_tokens,
    }

    for _, row in headlines.iterrows():
        headline = str(row.get(headline_col, "")).strip()
        if not headline:
            continue
        prompt = _build_prompt(headline)
        labels = []
        rationales = []
        for _ in range(votes):
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
            label, rationale = _parse_label(response)
            labels.append(label)
            rationales.append(rationale)
            time.sleep(sleep_seconds)

        vote = Counter(labels).most_common(1)[0][0]
        score = LABEL_MAP.get(vote, 0)
        record = dict(row)
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
