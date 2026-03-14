"""
Build a paper manuscript from an existing run directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from .paper_writer import PaperWriter


def _is_alignment_row(line: str) -> bool:
    content = line.replace("|", "").strip()
    if not content:
        return True
    return set(content) <= set("-: ")


def read_markdown_table(path: Path) -> Optional[pd.DataFrame]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    table_lines = [line for line in lines if "|" in line]
    if not table_lines:
        return None

    rows = []
    for line in table_lines:
        if _is_alignment_row(line):
            continue
        parts = [part.strip() for part in line.split("|")]
        if parts and parts[0] == "":
            parts = parts[1:]
        if parts and parts[-1] == "":
            parts = parts[:-1]
        rows.append(parts)

    if not rows:
        return None

    header = rows[0]
    data = rows[1:]
    df = pd.DataFrame(data, columns=header)

    if df.columns[0] == "" or df.columns[0].lower().startswith("unnamed"):
        df = df.drop(columns=[df.columns[0]])

    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass

    for col in [c for c in df.columns if c.endswith("significant") or c.endswith("better")]:
        if df[col].dtype == object:
            df[col] = df[col].map(
                lambda v: True if str(v).strip().lower() == "true"
                else False if str(v).strip().lower() == "false"
                else v
            )

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a paper manuscript from a run directory.")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to the run directory.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output paper directory.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = repo_root / run_dir

    config_path = run_dir / "config_resolved.yaml"
    panel_schema_path = run_dir / "data" / "panel_schema.json"

    config = {}
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}

    panel_schema = {}
    if panel_schema_path.exists():
        panel_schema = json.loads(panel_schema_path.read_text())

    tables_dir = run_dir / "paper_snapshot" / "tables"
    metrics_df = read_markdown_table(tables_dir / "mse_by_horizon.md") if tables_dir.exists() else None
    trend_df = read_markdown_table(tables_dir / "trend_accuracy.md") if tables_dir.exists() else None
    sig_df = read_markdown_table(tables_dir / "significance_tests.md") if tables_dir.exists() else None
    noise_df = read_markdown_table(tables_dir / "noise_injection.md") if tables_dir.exists() else None

    output_dir = Path(args.output_dir) if args.output_dir else config.get("output", {}).get("paper_dir", "paper")
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir

    writer = PaperWriter(output_dir)
    writer.write_all_sections(
        panel_schema=panel_schema,
        config=config,
        metrics_by_horizon=metrics_df,
        trend_accuracy=trend_df,
        significance_tests=sig_df,
        noise_results=noise_df
    )
    writer.save()


if __name__ == "__main__":
    main()
