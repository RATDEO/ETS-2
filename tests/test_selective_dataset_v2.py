from __future__ import annotations

import ast
import types
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "uk_ets/scripts/build_selective_residual_helpful_case_dataset_v2.py"

# Test the pure alignment helper without importing the legacy project module.
source = SCRIPT.read_text(encoding="utf-8")
tree = ast.parse(source)
selected = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom)) and not (
    isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("uk_ets")
)]
selected += [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "attach_last_observed_features"]
module = types.ModuleType("dataset_v2_alignment")
exec(compile(ast.Module(body=selected, type_ignores=[]), str(SCRIPT), "exec"), module.__dict__)


def test_alignment_uses_strictly_previous_panel_row() -> None:
    frame = pd.DataFrame({"date": pd.to_datetime(["2024-01-03", "2024-01-05"])})
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-04", "2024-01-05"]),
            "y": [10.0, 99.0, 12.0, 88.0],
            "y_return": [0.0, 9.9, 0.2, 7.3],
        }
    )
    aligned = module.attach_last_observed_features(frame, panel, ["y", "y_return"])
    assert aligned["feature_date"].tolist() == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-04")]
    assert aligned["asof_y"].tolist() == [10.0, 12.0]
    assert (aligned["feature_date"] < aligned["forecast_date"]).all()
