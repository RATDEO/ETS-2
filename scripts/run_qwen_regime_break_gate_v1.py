#!/usr/bin/env python3
"""Use Qwen as a categorical regime-break abstention gate over nested forecasts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from openai import OpenAI
from scipy.stats import ttest_rel, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.panel import select_feature_columns
from data.windows import WindowConfig, make_windows
from eval.residual_event_model import summarize_prediction
from news.event_panel import build_event_panel

from run_qwen_scalar_ramp_nested_backtest_v1 import load_full_windows
from run_ridge_qwen_residual_stacker_v1 import load_yaml, resolve_path


DEFAULT_CONFIG = ROOT / "uk_ets" / "config" / "uk_ets_qwen_regime_break_gate_v1.yaml"
ALLOWED_DIRECTIONS = {"above_base", "near_base", "below_base"}
CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--nested-report", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def market_snapshot(panel: pd.DataFrame, origin_date: pd.Timestamp) -> dict[str, float]:
    work = panel.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    known = work.loc[work["date"] <= pd.Timestamp(origin_date).normalize()].sort_values("date")
    if known.empty:
        raise ValueError(f"No market rows are known by {origin_date}.")
    latest = known.iloc[-1]
    prices = pd.to_numeric(known["y"], errors="coerce").dropna()

    def change(period: int) -> float:
        if len(prices) <= period or float(prices.iloc[-period - 1]) == 0.0:
            return 0.0
        return 100.0 * (float(prices.iloc[-1]) / float(prices.iloc[-period - 1]) - 1.0)

    return {
        "spot_price": _safe_float(latest.get("y")),
        "price_change_5obs_pct": change(5),
        "price_change_20obs_pct": change(20),
        "price_change_60obs_pct": change(60),
        "momentum_5d": _safe_float(latest.get("y_momentum_5d")),
        "momentum_20d": _safe_float(latest.get("y_momentum_20d")),
        "volatility_20d": _safe_float(latest.get("y_vol_20d")),
        "uk_gas_return": _safe_float(latest.get("uk_gas_return")),
        "uk_power_return": _safe_float(latest.get("uk_power_return")),
        "brent_return": _safe_float(latest.get("brent_return")),
        "coal_return": _safe_float(latest.get("coal_return")),
    }


def recent_headline_records(
    headlines: pd.DataFrame,
    origin_date: pd.Timestamp,
    lookback_days: int,
    max_headlines: int,
) -> list[dict[str, Any]]:
    work = headlines.copy()
    work["seendate"] = pd.to_datetime(work["seendate"], errors="coerce").dt.normalize()
    cutoff = pd.Timestamp(origin_date).normalize()
    start = cutoff - pd.Timedelta(days=int(lookback_days))
    known = work.loc[(work["seendate"] >= start) & (work["seendate"] <= cutoff)].copy()
    known["llm_importance"] = pd.to_numeric(known.get("llm_importance"), errors="coerce").fillna(0.0)
    known["llm_score"] = pd.to_numeric(known.get("llm_score"), errors="coerce").fillna(0.0)
    known = known.sort_values(["llm_importance", "seendate"], ascending=[False, False]).head(max_headlines)
    known = known.sort_values("seendate")
    return [
        {
            "date": str(pd.Timestamp(row.seendate).date()),
            "title": str(row.title),
            "prior_sentiment_label": _safe_float(row.llm_score),
            "importance": _safe_float(row.llm_importance),
        }
        for row in known.itertuples(index=False)
    ]


def build_regime_prompt(
    origin_date: pd.Timestamp,
    market: dict[str, float],
    base_path: np.ndarray,
    event_snapshot: dict[str, float],
    headlines: list[dict[str, Any]],
) -> str:
    base_path = np.asarray(base_path, dtype=float)
    base_summary = {
        "h1": float(base_path[0]),
        "h5": float(base_path[4]),
        "h10": float(base_path[9]),
        "h20": float(base_path[19]),
        "h30": float(base_path[29]),
        "h1_to_h30_pct": 100.0 * (float(base_path[29]) / max(abs(float(base_path[0])), 1e-9) - 1.0),
    }
    evidence = {
        "origin_date": str(pd.Timestamp(origin_date).date()),
        "market_known_at_origin": market,
        "ridge_base_forecast": base_summary,
        "qwen_event_aggregates_known_at_origin": event_snapshot,
        "recent_headlines_published_by_origin": headlines,
    }
    return f"""/no_think
You are a regime-break detector for UK carbon allowance prices. Do not produce a numerical forecast or percentage correction.

The quantitative ridge path below is the base forecast. Decide whether the realized price path over roughly the next 20-30 trading observations is more likely to finish materially ABOVE the ridge base, remain NEAR it, or finish materially BELOW it. Focus on structural policy, allowance supply/demand, energy switching, macro, and geopolitical evidence. Distinguish carbon-price direction from generic positive/negative prose.

Use only the supplied evidence, all timestamped on or before the origin. If evidence is mixed, stale, or weak, choose near_base or low confidence. Do not assume later events.

Evidence:
{json.dumps(evidence, indent=2, sort_keys=True)}

Return only one JSON object with exactly these fields:
{{
  "direction": "above_base" | "near_base" | "below_base",
  "confidence": "low" | "medium" | "high",
  "policy_or_supply_break": true | false,
  "energy_or_macro_break": true | false,
  "evidence_dates": ["YYYY-MM-DD"],
  "reason": "one concise sentence grounded in supplied evidence"
}}
"""


def parse_decision(content: str) -> dict[str, Any]:
    text = str(content).strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    payload = json.loads(text)
    direction = str(payload.get("direction", "")).strip().lower()
    confidence = str(payload.get("confidence", "")).strip().lower()
    if direction not in ALLOWED_DIRECTIONS:
        raise ValueError(f"Invalid regime direction: {direction!r}")
    if confidence not in CONFIDENCE_RANK:
        raise ValueError(f"Invalid regime confidence: {confidence!r}")
    payload["direction"] = direction
    payload["confidence"] = confidence
    return payload


def apply_regime_gate(
    base_pred: np.ndarray,
    refined_pred: np.ndarray,
    directions: np.ndarray,
    confidences: np.ndarray,
    min_confidence: str,
    activation_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    base_pred = np.asarray(base_pred, dtype=float)
    refined_pred = np.asarray(refined_pred, dtype=float)
    correction = refined_pred[:, -1] / np.maximum(np.abs(base_pred[:, -1]), 1e-9) - 1.0
    required_rank = CONFIDENCE_RANK[min_confidence]
    trusted = np.asarray([CONFIDENCE_RANK[str(value)] >= required_rank for value in confidences], dtype=bool)
    if activation_mask is not None:
        activation_mask = np.asarray(activation_mask, dtype=bool)
        if activation_mask.shape != trusted.shape:
            raise ValueError("activation_mask must match the number of prediction rows.")
        trusted &= activation_mask
    conflict = (
        ((directions == "above_base") & (correction < 0.0))
        | ((directions == "below_base") & (correction > 0.0))
        | (directions == "near_base")
    )
    reject = trusted & conflict & (np.abs(correction) > 1e-12)
    gated = refined_pred.copy()
    gated[reject] = base_pred[reject]
    return gated, reject


def main() -> int:
    args = parse_args()
    cfg = load_yaml(resolve_path(args.config))
    nested_report = resolve_path(args.nested_report or cfg["nested_report"])
    output_dir = (
        ROOT / "reports" / "qwen_regime_break_gate_v1" / datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.output_dir is None
        else resolve_path(args.output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "qwen_decisions"
    cache_dir.mkdir(parents=True, exist_ok=True)

    nested_audit = json.loads((nested_report / "audit.json").read_text(encoding="utf-8"))
    base_run = resolve_path(nested_audit["base_run"])
    events_path = resolve_path(nested_audit["event_source"])
    panel, _, _, _, _, full_dates, full_origin_dates = load_full_windows(base_run)
    headlines = pd.read_csv(events_path)
    event_cfg = cfg.get("event_features", {}) or {}
    event_panel = build_event_panel(
        events_path,
        calendar=panel["date"],
        rolling_windows=event_cfg.get("rolling_windows", [3, 7]),
        taxonomy_version=str(event_cfg.get("taxonomy_version", "v2")),
    )
    event_panel = event_panel.set_index(pd.to_datetime(event_panel["date"]).dt.normalize())

    predictions = np.load(nested_report / "predictions.npz", allow_pickle=True)
    window_indices = np.asarray(predictions["window_indices"], dtype=int)
    base_pred = np.asarray(predictions["ridge_base"], dtype=float)
    y_true = np.asarray(predictions["y_true"], dtype=float)
    decision_block_size = int(cfg.get("decision_block_size", 30))
    lookback_days = int(cfg.get("headline_lookback_days", 30))
    max_headlines = int(cfg.get("max_headlines", 18))

    endpoint_cfg = cfg.get("qwen", {}) or {}
    client = OpenAI(
        base_url=str(endpoint_cfg.get("base_url", "http://192.168.68.140:9881/v1")),
        api_key=str(endpoint_cfg.get("api_key", "deo")),
        timeout=float(endpoint_cfg.get("timeout_seconds", 180)),
        max_retries=int(endpoint_cfg.get("max_retries", 1)),
    )
    model = str(endpoint_cfg.get("model", "qwen3.6-27b-q4-k-m"))

    row_directions = np.empty(len(window_indices), dtype=object)
    row_confidences = np.empty(len(window_indices), dtype=object)
    row_activation = np.zeros(len(window_indices), dtype=bool)
    decision_rows: list[dict[str, Any]] = []
    gate_mode = str(cfg.get("gate_mode", "any_conflict"))
    cache_source = cfg.get("decision_cache_source")
    cache_source_dir = None if cache_source is None else resolve_path(cache_source) / "qwen_decisions"
    for block_start in range(0, len(window_indices), decision_block_size):
        block_end = min(block_start + decision_block_size, len(window_indices))
        window_index = int(window_indices[block_start])
        origin_date = pd.Timestamp(full_origin_dates[window_index]).normalize()
        market = market_snapshot(panel, origin_date)
        headline_records = recent_headline_records(
            headlines, origin_date, lookback_days=lookback_days, max_headlines=max_headlines
        )
        event_row = event_panel.reindex([origin_date]).fillna(0.0).iloc[0]
        event_names = [
            "evt_news_count_7d_sum",
            "evt_weighted_sent_sum_7d_sum",
            "evt_importance_sum_7d_sum",
            "evt_eu_ets_strong_count_7d_sum",
            "evt_energy_driver_count_7d_sum",
            "evt2_policy_count_7d_sum",
            "evt2_policy_weighted_sent_sum_7d_sum",
            "evt2_energy_count_7d_sum",
            "evt2_energy_weighted_sent_sum_7d_sum",
        ]
        event_snapshot = {name: _safe_float(event_row.get(name)) for name in event_names}
        prompt = build_regime_prompt(
            origin_date,
            market,
            base_pred[block_start],
            event_snapshot,
            headline_records,
        )
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = cache_dir / f"{str(origin_date.date())}_{prompt_hash[:12]}.json"
        source_cache_path = (
            None
            if cache_source_dir is None
            else cache_source_dir / f"{str(origin_date.date())}_{prompt_hash[:12]}.json"
        )
        readable_cache_path = cache_path if cache_path.exists() else source_cache_path
        if readable_cache_path is not None and readable_cache_path.exists():
            cache_payload = json.loads(readable_cache_path.read_text(encoding="utf-8"))
            decision = cache_payload["decision"]
            raw_content = cache_payload["raw_content"]
            reasoning_content = cache_payload.get("reasoning_content")
            if not cache_path.exists():
                cache_path.write_text(json.dumps(cache_payload, indent=2), encoding="utf-8")
        else:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "Return strict JSON. Do not reveal hidden reasoning. Follow the requested evidence cutoff.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=int(endpoint_cfg.get("max_tokens", 384)),
                response_format={"type": "json_object"},
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            message = response.choices[0].message
            raw_content = str(message.content)
            reasoning_content = getattr(message, "reasoning_content", None)
            decision = parse_decision(raw_content)
            cache_payload = {
                "origin_date": str(origin_date.date()),
                "model": model,
                "thinking_enabled": False,
                "non_thinking_prompt_tag": "/no_think",
                "prompt_sha256": prompt_hash,
                "prompt": prompt,
                "raw_content": raw_content,
                "reasoning_content": reasoning_content,
                "decision": decision,
            }
            cache_path.write_text(json.dumps(cache_payload, indent=2), encoding="utf-8")
        decision = parse_decision(json.dumps(decision))
        if reasoning_content not in (None, ""):
            raise RuntimeError("Qwen returned reasoning content even though thinking was disabled.")
        row_directions[block_start:block_end] = decision["direction"]
        row_confidences[block_start:block_end] = decision["confidence"]
        price_change_20d = float(market["price_change_20obs_pct"])
        countertrend = (
            (decision["direction"] == "above_base" and price_change_20d < 0.0)
            or (decision["direction"] == "below_base" and price_change_20d > 0.0)
        )
        confirmed_countertrend_break = bool(
            decision.get("policy_or_supply_break", False)
            and decision.get("energy_or_macro_break", False)
            and countertrend
        )
        if gate_mode == "confirmed_countertrend_break":
            row_activation[block_start:block_end] = confirmed_countertrend_break
        elif gate_mode == "any_conflict":
            row_activation[block_start:block_end] = True
        else:
            raise ValueError(f"Unknown gate_mode: {gate_mode}")
        decision_rows.append(
            {
                "block_start_row": block_start,
                "block_end_row": block_end,
                "window_index": window_index,
                "forecast_start": str(pd.Timestamp(full_dates[window_index]).date()),
                "origin_date": str(origin_date.date()),
                "direction": decision["direction"],
                "confidence": decision["confidence"],
                "policy_or_supply_break": bool(decision.get("policy_or_supply_break", False)),
                "energy_or_macro_break": bool(decision.get("energy_or_macro_break", False)),
                "reason": str(decision.get("reason", "")),
                "price_change_20obs_pct": price_change_20d,
                "countertrend_direction": countertrend,
                "confirmed_countertrend_break": confirmed_countertrend_break,
                "gate_mode_active": bool(row_activation[block_start]),
                "evidence_dates": ";".join(str(value) for value in decision.get("evidence_dates", [])),
                "thinking_enabled": False,
                "reasoning_content_present": reasoning_content not in (None, ""),
                "prompt_sha256": prompt_hash,
            }
        )

    min_confidence = str(cfg.get("min_confidence", "medium"))
    variants: dict[str, np.ndarray] = {"ridge_base": base_pred}
    gate_rows: list[dict[str, Any]] = []
    for source_name in ("selected", "best_qwen"):
        source_pred = np.asarray(predictions[source_name], dtype=float)
        gated, reject = apply_regime_gate(
            base_pred,
            source_pred,
            row_directions,
            row_confidences,
            min_confidence=min_confidence,
            activation_mask=row_activation,
        )
        variants[source_name] = source_pred
        variants[f"{source_name}_qwen_regime_gate"] = gated
        gate_rows.append(
            {
                "source_variant": source_name,
                "rejected_rows": int(reject.sum()),
                "rejected_share": float(reject.mean()),
                "applied_rows": int((~reject & (np.max(np.abs(source_pred - base_pred), axis=1) > 1e-12)).sum()),
            }
        )

    base_mse = float(np.mean((y_true - base_pred) ** 2))
    metric_rows: list[dict[str, Any]] = []
    for name, prediction in variants.items():
        row = summarize_prediction(name, y_true, prediction)
        row["improvement_vs_base_pct"] = 100.0 * (base_mse - float(row["path_mse"])) / base_mse
        metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows).sort_values("path_mse").reset_index(drop=True)

    gated_error = np.mean((y_true - variants["selected_qwen_regime_gate"]) ** 2, axis=1)
    base_error = np.mean((y_true - base_pred) ** 2, axis=1)
    t_result = ttest_rel(base_error, gated_error)
    try:
        w_result = wilcoxon(base_error, gated_error)
        w_stat, w_p = float(w_result.statistic), float(w_result.pvalue)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    significance = pd.DataFrame(
        [
            {"test": "paired_t", "statistic": float(t_result.statistic), "p_value": float(t_result.pvalue)},
            {"test": "wilcoxon", "statistic": w_stat, "p_value": w_p},
        ]
    )

    decision_table = pd.DataFrame(decision_rows)
    gate_table = pd.DataFrame(gate_rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    decision_table.to_csv(output_dir / "qwen_regime_decisions.csv", index=False)
    gate_table.to_csv(output_dir / "gate_actions.csv", index=False)
    significance.to_csv(output_dir / "significance_tests.csv", index=False)
    np.savez_compressed(
        output_dir / "predictions.npz",
        window_indices=window_indices,
        dates=np.asarray(predictions["dates"]),
        y_true=y_true,
        directions=row_directions.astype(str),
        confidences=row_confidences.astype(str),
        gate_activation=row_activation,
        **variants,
    )
    (output_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    selected_row = metrics.loc[metrics["variant"] == "selected"].iloc[0]
    gated_row = metrics.loc[metrics["variant"] == "selected_qwen_regime_gate"].iloc[0]
    qwen_row = metrics.loc[metrics["variant"] == "best_qwen"].iloc[0]
    gated_qwen_row = metrics.loc[metrics["variant"] == "best_qwen_qwen_regime_gate"].iloc[0]
    decision_lines = "\n".join(
        f"| {row.origin_date} | {row.direction} | {row.confidence} | {row.reason} |"
        for row in decision_table.itertuples(index=False)
    )
    summary = f"""# Qwen regime-break gate v1

## Aggregate result

- Nested adaptive stacker: {float(selected_row['path_mse']):.6f} MSE ({float(selected_row['improvement_vs_base_pct']):+.3f}% vs ridge)
- Adaptive stacker + Qwen regime gate: {float(gated_row['path_mse']):.6f} MSE ({float(gated_row['improvement_vs_base_pct']):+.3f}%)
- Best-Qwen comparator: {float(qwen_row['path_mse']):.6f} MSE ({float(qwen_row['improvement_vs_base_pct']):+.3f}%)
- Best-Qwen comparator + regime gate: {float(gated_qwen_row['path_mse']):.6f} MSE ({float(gated_qwen_row['improvement_vs_base_pct']):+.3f}%)
- Gated-policy p-values: paired t-test {float(t_result.pvalue):.6f}; Wilcoxon {w_p:.6f}
- Thinking enabled: no (`enable_thinking=false`, `/no_think`, no reasoning content returned)
- Gate mode: `{gate_mode}`

## Qwen decisions

| Origin | Direction relative to ridge | Confidence | Evidence summary |
|---|---|---|---|
{decision_lines}

The gate is one-way: it can replace a conflicting refinement with the ridge base, but cannot flip the correction or create a numerical forecast. Decisions use only market rows and headlines dated on or before each origin.

This remains a retrospective experiment. Although Qwen never received realized outcomes, the regime-gate architecture was introduced after inspecting failures in the first nested backtest and therefore needs a newly acquired period for clean prospective confirmation. The `confirmed_countertrend_break` rule, when used, is explicitly exploratory because it was formulated after reviewing the v1 block failures.
"""
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "metrics": metrics.to_dict(orient="records"), "decisions": decision_rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
