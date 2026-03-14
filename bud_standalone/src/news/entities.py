"""Entity mapping helpers for carbon-market headlines."""

from __future__ import annotations

from typing import Iterable


EU_ETS_TERMS = [
    "eu ets",
    "european emissions trading system",
    "european union allowance",
    "eu allowance",
    "eua",
    "eua futures",
    "eex",
    "ice eua",
    "eua auction",
    "eu carbon",
    "eu carbon price",
    "eu emissions trading",
]

OTHER_CARBON_TERMS = [
    "uk ets",
    "china ets",
    "chinese ets",
    "korea ets",
    "korean ets",
    "california cap-and-trade",
    "cap-and-trade california",
    "rggi",
    "wci",
    "quebec cap-and-trade",
    "nz ets",
    "new zealand ets",
    "accu",
    "ccer",
]

GENERAL_CARBON_TERMS = [
    "carbon price",
    "carbon market",
    "carbon trading",
    "emissions trading",
    "carbon allowance",
    "carbon permit",
]


def _matches_any(text: str, terms: Iterable[str]) -> bool:
    return any(term in text for term in terms)


def classify_entity(title: str) -> str:
    """Classify headline into EU_ETS / OTHER_CARBON / GENERAL_CARBON / UNKNOWN."""
    text = title.lower().strip()
    if not text:
        return "UNKNOWN"
    if _matches_any(text, EU_ETS_TERMS):
        return "EU_ETS"
    if _matches_any(text, OTHER_CARBON_TERMS):
        return "OTHER_CARBON"
    if _matches_any(text, GENERAL_CARBON_TERMS):
        return "GENERAL_CARBON"
    return "UNKNOWN"
