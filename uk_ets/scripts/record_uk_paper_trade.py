#!/usr/bin/env python3
"""Record a frozen UK ETS forecast packet into a paper-trading ledger."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.return_metrics import assess_llm_training_cutoff


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def _load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_model(payload: dict[str, Any], model_name: str) -> dict[str, Any]:
    models = payload.get("models", [])
    for model in models:
        if model.get("model") == model_name:
            return model
    available = ", ".join(str(m.get("model")) for m in models)
    raise ValueError(f"Model {model_name!r} was not found in forecast payload. Available: {available}")


def _compute_signal(last_observed_price: float, forecast_points: list[dict[str, Any]], horizon: int, threshold: float) -> tuple[float, int, str]:
    point = next((row for row in forecast_points if int(row.get("horizon", -1)) == int(horizon)), None)
    if point is None:
        raise ValueError(f"Forecast payload does not include horizon {horizon}")
    predicted_price = float(point["predicted_price"])
    predicted_return = (predicted_price / float(last_observed_price)) - 1.0
    signal = 1 if predicted_return > threshold else -1 if predicted_return < -threshold else 0
    return predicted_return, signal, str(point["target_date"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Append a frozen forecast packet to a UK ETS paper-trading ledger.")
    parser.add_argument("--forecast-json", required=True, help="Path to current_forecast.json-style payload.")
    parser.add_argument("--model", required=True, help="Model key inside the forecast payload.")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--signal-threshold", type=float, default=0.0)
    parser.add_argument("--output-root", default="reports/uk_ets_paper_trading")
    parser.add_argument("--cutoff-date", default="2025-01-01")
    parser.add_argument("--execution-convention", default="enter_next_session_close_hold_h_days")
    args = parser.parse_args()

    forecast_path = _resolve_path(args.forecast_json)
    output_root = _resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    payload = _load_payload(forecast_path)
    model_payload = _find_model(payload, args.model)
    forecast_points = list(model_payload.get("forecast", []))
    if not forecast_points:
        raise ValueError(f"Model {args.model!r} has no forecast path in {forecast_path}")

    last_observed_price = float(payload["last_observed_price"])
    predicted_return, signal, target_date = _compute_signal(
        last_observed_price=last_observed_price,
        forecast_points=forecast_points,
        horizon=args.horizon,
        threshold=float(args.signal_threshold),
    )
    cutoff_note = assess_llm_training_cutoff(
        [row.get("target_date") for row in forecast_points],
        cutoff_date=args.cutoff_date,
    )

    packet = {
        "recorded_at_utc": datetime.utcnow().isoformat(),
        "source_forecast_json": str(forecast_path),
        "run_id": payload.get("run_id"),
        "origin_date": payload.get("origin_date"),
        "data_cutoff_date": payload.get("data_cutoff_date"),
        "model": args.model,
        "primary_horizon": int(args.horizon),
        "signal_threshold": float(args.signal_threshold),
        "last_observed_price": last_observed_price,
        "predicted_target_date": target_date,
        "predicted_cumulative_return": float(predicted_return),
        "signal": int(signal),
        "execution_convention": args.execution_convention,
        "llm_cutoff_status": cutoff_note["status"],
        "llm_cutoff_note": cutoff_note["note"],
    }

    packets_dir = output_root / "packets"
    packets_dir.mkdir(parents=True, exist_ok=True)
    packet_name = f"{packet['origin_date']}_{args.model}_h{args.horizon}.json".replace(":", "-")
    packet_path = packets_dir / packet_name
    packet_path.write_text(json.dumps(packet, indent=2), encoding="utf-8")

    ledger_path = output_root / "paper_trade_ledger.csv"
    ledger_row = pd.DataFrame([packet])
    if ledger_path.exists():
        existing = pd.read_csv(ledger_path)
        existing = existing[
            ~(
                (existing["origin_date"] == packet["origin_date"])
                & (existing["model"] == packet["model"])
                & (existing["primary_horizon"] == packet["primary_horizon"])
            )
        ]
        ledger_row = pd.concat([existing, ledger_row], ignore_index=True)
    ledger_row = ledger_row.sort_values(["origin_date", "model", "primary_horizon"]).reset_index(drop=True)
    ledger_row.to_csv(ledger_path, index=False)

    manifest = {
        "generated_at_utc": datetime.utcnow().isoformat(),
        "ledger_path": str(ledger_path),
        "packet_path": str(packet_path),
        "note": (
            "This ledger records frozen forecast packets before outcomes are known. "
            "Execution and realized PnL should be appended later by a separate reconciliation step."
        ),
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(packet_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
