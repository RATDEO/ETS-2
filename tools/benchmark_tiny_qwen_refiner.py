#!/usr/bin/env python3
"""CPU-only tiny-Qwen smoke benchmark for the ETS2 refinement plumbing.

This is deliberately *not* the final LLM benchmark. It checks that a small
OpenAI-style instruction model can consume a strictly as-of prompt, emit a
bounded structured adjustment, be parsed deterministically, and fall back to
the frozen base forecast on failure. The final quality evaluation belongs on
the owner's Qwen 27B server.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HORIZONS = (1, 5, 20, 30)
DEFAULT_RUN = Path("runs/20260320_182015_da9e9d")
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _json_object(text: str) -> dict[str, Any] | None:
    """Extract the last balanced JSON object from model text."""
    candidates: list[str] = []
    depth = 0
    start: int | None = None
    in_string = False
    escaped = False
    for index, char in enumerate(text):
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
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start : index + 1])
                start = None
    for candidate in reversed(candidates):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _parse_adjustments(text: str, bound_pct: float) -> tuple[dict[int, float], str]:
    payload = _json_object(text)
    if payload is None:
        return {1: 0.0, 5: 0.0, 20: 0.0, 30: 0.0}, "no_json"
    result = {1: 0.0}
    try:
        for horizon in (5, 20, 30):
            raw = payload.get(f"h{horizon}_pct", 0.0)
            value = float(raw)
            if not np.isfinite(value):
                raise ValueError("non-finite adjustment")
            result[horizon] = float(np.clip(value, -bound_pct, bound_pct))
    except (TypeError, ValueError):
        return {1: 0.0, 5: 0.0, 20: 0.0, 30: 0.0}, "invalid_values"
    return result, "ok"


def _apply(base: np.ndarray, anchors: dict[int, float]) -> tuple[np.ndarray, np.ndarray]:
    steps = np.arange(1, len(base) + 1, dtype=float)
    anchor_x = np.array(sorted(anchors), dtype=float)
    anchor_y = np.array([anchors[int(h)] for h in anchor_x], dtype=float)
    curve_pct = np.interp(steps, anchor_x, anchor_y)
    refined = np.asarray(base, dtype=float) * (1.0 + curve_pct / 100.0)
    return refined, curve_pct


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    result = {
        "mse_path": float(np.mean((actual - predicted) ** 2)),
        "mae_path": float(np.mean(np.abs(actual - predicted))),
    }
    for horizon in HORIZONS:
        if horizon <= actual.shape[1]:
            result[f"mse_h{horizon}"] = float(
                np.mean((actual[:, horizon - 1] - predicted[:, horizon - 1]) ** 2)
            )
    return result


def _prompt(history: np.ndarray, base: np.ndarray, bound_pct: float) -> str:
    anchors = {h: float(base[h - 1]) for h in HORIZONS}
    return f"""You are checking a frozen UK carbon allowance price forecast.
Use only the supplied historical prices and base forecast. Do not invent news.
Return one JSON object and no prose:
{{"h5_pct": number, "h20_pct": number, "h30_pct": number}}
Each number is a percentage adjustment to the base price at that horizon and
must be between {-bound_pct:.2f} and {bound_pct:.2f}. A value of 0 means keep
the base. Be conservative. Day 1 is frozen and must not be adjusted.

Last observed prices, oldest to newest:
{json.dumps([round(float(x), 3) for x in history])}

Frozen base prices:
{json.dumps({f'h{h}': round(v, 3) for h, v in anchors.items()})}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--bound-pct", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/agent_tiny_qwen_smoke")
    )
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    prediction_path = (
        args.run_dir
        / "predictions"
        / "TSM+LLM-COT-RF-HDELTA_pred_test_subset.npz"
    )
    with np.load(prediction_path, allow_pickle=True) as saved:
        dates = pd.to_datetime(saved["dates"])
        actual = np.asarray(saved["y_true"], dtype=float)
        base = np.asarray(saved["base_pred"], dtype=float)

    panel = pd.read_parquet(args.run_dir / "data" / "panel.parquet")
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel = panel.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    positions = {date: index for index, date in enumerate(panel["date"])}
    price = panel["y"].to_numpy(dtype=float)

    count = min(args.max_samples, len(dates))
    selected = np.unique(np.linspace(0, len(dates) - 1, count, dtype=int))

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float32,
        trust_remote_code=False,
    )
    model.eval()

    refined_rows: list[np.ndarray] = []
    response_rows: list[dict[str, Any]] = []
    for case_number, source_index in enumerate(selected):
        forecast_date = pd.Timestamp(dates[source_index]).normalize()
        panel_index = positions[forecast_date]
        history = price[max(0, panel_index - 18) : panel_index]
        if len(history) < 5:
            raise ValueError(f"Insufficient as-of history for {forecast_date.date()}")
        prompt = _prompt(history, base[source_index], args.bound_pct)
        messages = [
            {
                "role": "system",
                "content": "Return only valid JSON. Do not provide hidden reasoning or prose.",
            },
            {"role": "user", "content": prompt},
        ]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(rendered, return_tensors="pt")
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=96,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        latency = time.perf_counter() - started
        continuation = generated[0, inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(continuation, skip_special_tokens=True).strip()
        anchors, parse_status = _parse_adjustments(text, args.bound_pct)
        refined, curve = _apply(base[source_index], anchors)
        refined_rows.append(refined)
        response_rows.append(
            {
                "case": case_number,
                "source_index": int(source_index),
                "date": forecast_date.date().isoformat(),
                "parse_status": parse_status,
                "latency_seconds": latency,
                "response": text,
                "anchors_pct": anchors,
                "mean_abs_adjustment_pct": float(np.mean(np.abs(curve))),
            }
        )
        print(
            f"{case_number + 1}/{len(selected)} {forecast_date.date()} "
            f"parse={parse_status} latency={latency:.2f}s"
        )

    actual_subset = actual[selected]
    base_subset = base[selected]
    refined_subset = np.vstack(refined_rows)
    base_metrics = _metrics(actual_subset, base_subset)
    refined_metrics = _metrics(actual_subset, refined_subset)
    improvement = 100.0 * (
        base_metrics["mse_path"] - refined_metrics["mse_path"]
    ) / base_metrics["mse_path"]
    successful = sum(row["parse_status"] == "ok" for row in response_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "source_index": selected,
            "date": dates[selected].astype(str),
            "base_case_mse": np.mean((actual_subset - base_subset) ** 2, axis=1),
            "refined_case_mse": np.mean(
                (actual_subset - refined_subset) ** 2, axis=1
            ),
            "parse_status": [row["parse_status"] for row in response_rows],
            "latency_seconds": [row["latency_seconds"] for row in response_rows],
            "mean_abs_adjustment_pct": [
                row["mean_abs_adjustment_pct"] for row in response_rows
            ],
        }
    ).to_csv(args.output_dir / "per_case.csv", index=False)
    with (args.output_dir / "responses.jsonl").open("w", encoding="utf-8") as handle:
        for row in response_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "purpose": "CPU-only provisional LLM plumbing smoke test; not final model evidence",
        "model_id": args.model_id,
        "run_dir": str(args.run_dir),
        "sample_count": int(len(selected)),
        "selection": "evenly spaced deterministic origins",
        "bound_pct": args.bound_pct,
        "parse_success_count": successful,
        "parse_success_rate": successful / len(response_rows),
        "fallback_count": len(response_rows) - successful,
        "base": base_metrics,
        "refined": refined_metrics,
        "mse_path_improvement_pct": improvement,
        "median_latency_seconds": float(
            np.median([row["latency_seconds"] for row in response_rows])
        ),
        "mean_abs_adjustment_pct": float(
            np.mean([row["mean_abs_adjustment_pct"] for row in response_rows])
        ),
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    markdown = [
        "# Tiny-Qwen CPU refinement smoke test",
        "",
        "This run checks the refinement interface only. It is not a substitute for the owner's Qwen 27B server and is not a statistically powered performance claim.",
        "",
        f"- Model: `{args.model_id}`",
        f"- Frozen historical origins: **{len(selected)}**",
        f"- Parse success: **{successful}/{len(response_rows)}**",
        f"- Fallbacks: **{len(response_rows) - successful}**",
        f"- Base path MSE: **{base_metrics['mse_path']:.6f}**",
        f"- Refined path MSE: **{refined_metrics['mse_path']:.6f}**",
        f"- Provisional change: **{improvement:+.3f}%**",
        f"- Median CPU generation latency: **{summary['median_latency_seconds']:.2f}s/case**",
        f"- Mean absolute applied adjustment: **{summary['mean_abs_adjustment_pct']:.3f}%**",
        "",
        "Any final LLM-refinement conclusion must be rerun on newly frozen windows using the exact Qwen 27B model served by the owner's GPU server.",
    ]
    (args.output_dir / "summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
