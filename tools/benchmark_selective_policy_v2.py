#!/usr/bin/env python3
"""Causal, leakage-safe benchmark for UK ETS selective LLM policies.

The script evaluates only chronological holdouts. Candidate model and threshold
are selected from forward out-of-fold predictions on older windows, refit on
all older windows, then scored once on the next unseen window. With W4..W1,
W2 and W1 are genuine test windows.
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUTCOMES = {
    "base_path_mse", "llm_path_mse", "path_uplift_abs", "path_uplift_pct",
    "base_h1_mse", "llm_h1_mse", "h1_uplift_abs", "h1_uplift_pct",
    "base_h5_mse", "llm_h5_mse", "h5_uplift_abs", "h5_uplift_pct",
    "base_h20_mse", "llm_h20_mse", "h20_uplift_abs", "h20_uplift_pct",
    "base_h30_mse", "llm_h30_mse", "h30_uplift_abs", "h30_uplift_pct",
    "helpful_loose", "helpful_long_only", "helpful_strict", "harmful_strict",
    "base_bias_h20", "base_bias_h30", "llm_bias_h20", "llm_bias_h30",
}
PRECALL = (
    "asof_y", "asof_y_return", "asof_target_range_pct", "asof_target_volume",
    "asof_y_vol_20d", "asof_y_ma_5d", "asof_y_momentum_20d",
    "asof_is_auction_day", "asof_uk_icap_secondary_print_day",
    "asof_coal_brent_ratio", "asof_coal_brent_ratio_z20",
    "asof_uk_power_gas_vol_ratio_20d", "asof_uk_gas_hdd18_surprise_interaction",
    "asof_uk_hdd18_7d_ma", "asof_uka_brent_ratio", "asof_uka_brent_ratio_z20",
    "asof_uk_gas_vol_20d", "asof_uk_temp_mean_c",
    "base_move_h20_pct", "base_move_h30_pct",
)
OUTCOME_COLUMNS = OUTCOMES

POSTCALL = (
    "apply_prompt_length", "teaching_examples", "matched_teaching_count",
    "support_example_count", "positive_memory_count", "negative_memory_count",
    "dynamic_frozen_horizons", "adjust_h1", "adjust_h5", "adjust_h20", "adjust_h30",
)


def candidates(seed: int):
    yield "dummy_mean", 0, lambda: DummyRegressor(strategy="mean")
    for alpha in (0.1, 1.0, 10.0, 100.0):
        yield f"ridge_a{alpha:g}", 1, lambda a=alpha: make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=a)
        )
    for leaf, fraction in ((5, 0.7), (10, 0.7), (20, 1.0)):
        yield f"extra_trees_leaf{leaf}_mf{fraction:g}", 2, lambda l=leaf, f=fraction: make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(
                n_estimators=300, min_samples_leaf=l, max_features=f,
                random_state=seed, n_jobs=-1,
            ),
        )


def validate(frame: pd.DataFrame) -> pd.DataFrame:
    needed = {"window", "date", "base_path_mse", "llm_path_mse", "path_uplift_abs", "llm_applied"}
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="raise")
    out["window"] = out["window"].astype(str)
    out["llm_applied"] = out["llm_applied"].fillna(0).astype(int).clip(0, 1)
    if out.duplicated(["window", "date"]).any():
        raise ValueError("Duplicate (window, date) rows")
    error = np.max(np.abs(out["path_uplift_abs"] - (out["base_path_mse"] - out["llm_path_mse"])))
    if error > 1e-7:
        raise ValueError(f"Uplift identity failed: {error:g}")
    return out.sort_values(["date", "window"]).reset_index(drop=True)


def resolve_features(frame: pd.DataFrame, policy: str) -> list[str]:
    requested = PRECALL if policy == "precall" else POSTCALL if policy == "postcall" else ()
    if not requested:
        raise ValueError(f"Unknown policy: {policy}")
    features = [name for name in requested if name in frame.columns]
    if len(features) < 2:
        raise ValueError(f"Only {len(features)} safe {policy} features are available")
    forbidden = OUTCOMES & set(features)
    suspicious = [x for x in features if "bias" in x or "uplift" in x or x.endswith("_mse")]
    if forbidden or suspicious:
        raise AssertionError(f"Leaky feature list: {sorted(forbidden | set(suspicious))}")
    return features


def window_order(frame: pd.DataFrame) -> list[str]:
    order = frame.groupby("window", observed=True)["date"].min().sort_values().index.astype(str).tolist()
    if len(order) < 3:
        raise ValueError("At least three windows are required")
    return order


def fit_scores(factory, train: pd.DataFrame, score: pd.DataFrame, features: list[str]) -> pd.Series:
    train = train[train["llm_applied"] == 1]
    opportunities = score[score["llm_applied"] == 1]
    if len(train) < max(12, len(features) // 2):
        raise ValueError(f"Too few applied training rows: {len(train)}")
    model = factory()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(train[features], train["path_uplift_abs"])
    result = pd.Series(-np.inf, index=score.index, dtype=float)
    if len(opportunities):
        result.loc[opportunities.index] = model.predict(opportunities[features])
    return result


def metrics(frame: pd.DataFrame, scores: np.ndarray, threshold: float):
    applied = frame["llm_applied"].to_numpy(bool)
    keep = applied & (scores > threshold)
    selected = np.where(keep, frame["llm_path_mse"], frame["base_path_mse"])
    macro = pd.DataFrame({"window": frame["window"], "mse": selected}).groupby("window")["mse"].mean().mean()
    return float(macro), float(np.mean(selected)), float(np.mean(keep[applied])) if applied.any() else 0.0


def choose(frame: pd.DataFrame, older: list[str], features: list[str], seed: int):
    audit, best = [], None
    for name, complexity, factory in candidates(seed):
        parts, failure = [], ""
        for pos in range(1, len(older)):
            train = frame[frame["window"].isin(older[:pos])]
            val = frame[frame["window"] == older[pos]]
            try:
                scores = fit_scores(factory, train, val, features)
            except Exception as exc:
                failure = f"{type(exc).__name__}: {exc}"
                break
            part = val[["window", "date", "base_path_mse", "llm_path_mse", "llm_applied"]].copy()
            part["score"] = scores
            parts.append(part)
        if failure or not parts:
            audit.append({"candidate": name, "status": "failed", "error": failure or "no folds"})
            continue
        oof = pd.concat(parts).sort_values("date")
        finite = oof.loc[np.isfinite(oof["score"]), "score"].to_numpy()
        grid = np.unique(np.r_[[-np.inf, 0.0], np.quantile(finite, np.linspace(0, 1, 41)), [np.inf]])
        local = None
        for threshold in grid:
            macro, pooled, share = metrics(oof, oof["score"].to_numpy(), float(threshold))
            key = (macro, pooled, share, -threshold if np.isfinite(threshold) else 0.0)
            if local is None or key < local[0]:
                local = (key, float(threshold), macro, pooled, share)
        assert local is not None
        _, threshold, macro, pooled, share = local
        audit.append({
            "candidate": name, "status": "ok", "threshold": threshold,
            "validation_macro_mse": macro, "validation_pooled_mse": pooled,
            "validation_keep_share_applied": share, "n_validation_rows": len(oof),
        })
        key = (macro, pooled, complexity, share, name)
        if best is None or key < best[0]:
            best = (key, name, factory, threshold, macro)
    if best is None:
        raise RuntimeError("Every candidate failed")
    return best, pd.DataFrame(audit)


def pct(base: float, contender: float) -> float:
    return 100.0 * (base - contender) / base if abs(base) > 1e-12 else float("nan")


def run_policy(frame: pd.DataFrame, policy: str, seed: int = 42):
    frame, features = validate(frame), resolve_features(frame, policy)
    order = window_order(frame)
    result_rows, decisions, audits = [], [], []
    for pos in range(2, len(order)):
        older, test_window = order[:pos], order[pos]
        best, audit = choose(frame, older, features, seed)
        _, name, factory, threshold, validation_mse = best
        audit.insert(0, "policy", policy)
        audit.insert(1, "test_window", test_window)
        audit.insert(2, "prior_windows", "+".join(older))
        audits.append(audit)
        train = frame[frame["window"].isin(older)]
        test = frame[frame["window"] == test_window].copy()
        scores = fit_scores(factory, train, test, features)
        applied = test["llm_applied"].to_numpy(bool)
        keep = applied & (scores.to_numpy() > threshold)
        selected = np.where(keep, test["llm_path_mse"], test["base_path_mse"])
        base, always, chosen = float(test["base_path_mse"].mean()), float(test["llm_path_mse"].mean()), float(selected.mean())
        result_rows.append({
            "policy": policy, "test_window": test_window, "prior_windows": "+".join(older),
            "train_max_date": train["date"].max().date().isoformat(),
            "test_min_date": test["date"].min().date().isoformat(),
            "selected_candidate": name, "threshold": threshold,
            "validation_macro_mse": validation_mse, "n_test": len(test),
            "n_applied_opportunities": int(applied.sum()),
            "keep_share_applied": float(np.mean(keep[applied])) if applied.any() else 0.0,
            "base_mse": base, "always_accept_mse": always, "selected_mse": chosen,
            "improvement_vs_base_pct": pct(base, chosen),
            "improvement_vs_always_accept_pct": pct(always, chosen),
        })
        part = test[["window", "date", "base_path_mse", "llm_path_mse", "llm_applied"]].copy()
        part.insert(0, "policy", policy)
        part["score"], part["threshold"], part["accepted"] = scores, threshold, keep.astype(int)
        part["selected_path_mse"], part["selected_candidate"] = selected, name
        decisions.append(part)
    results = pd.DataFrame(result_rows)
    if not (pd.to_datetime(results["train_max_date"]) < pd.to_datetime(results["test_min_date"])).all():
        raise AssertionError("Training reaches into a test window")
    return results, pd.concat(decisions, ignore_index=True), pd.concat(audits, ignore_index=True), features


def block_bootstrap(decisions: pd.DataFrame, reps=2000, block=30, seed=42):
    rng, windows = np.random.default_rng(seed), sorted(decisions["window"].unique())
    versus_base, versus_always = [], []
    for _ in range(reps):
        b, a = [], []
        for window in windows:
            part = decisions[decisions["window"] == window].sort_values("date")
            n, width = len(part), min(block, len(part))
            starts = rng.integers(0, n, size=math.ceil(n / width))
            idx = ((starts[:, None] + np.arange(width)) % n).reshape(-1)[:n]
            sample = part.iloc[idx]
            b.append(float((sample["base_path_mse"] - sample["selected_path_mse"]).mean()))
            a.append(float((sample["llm_path_mse"] - sample["selected_path_mse"]).mean()))
        versus_base.append(np.mean(b)); versus_always.append(np.mean(a))
    out = {}
    for label, values in (("base", versus_base), ("always", versus_always)):
        arr = np.asarray(values)
        low, high = np.quantile(arr, [0.025, 0.975])
        out[f"vs_{label}_ci_low"], out[f"vs_{label}_ci_high"] = float(low), float(high)
        out[f"vs_{label}_conclusion"] = "positive" if low > 0 else "negative" if high < 0 else "inconclusive"
    out.update({"block_size": block, "n_bootstrap": reps})
    return out


def moving_block_bootstrap(decisions: pd.DataFrame, *, block_size=30, n_bootstrap=2000, seed=42):
    """Compatibility wrapper with explicit keyword names used by tests."""
    return block_bootstrap(decisions, n_bootstrap, block_size, seed)


def aggregate(results: pd.DataFrame):
    rows = []
    for policy, part in results.groupby("policy"):
        base, always, selected = part["base_mse"].mean(), part["always_accept_mse"].mean(), part["selected_mse"].mean()
        rows.append({
            "policy": policy, "n_test_windows": len(part), "macro_base_mse": base,
            "macro_always_accept_mse": always, "macro_selected_mse": selected,
            "improvement_vs_base_pct": pct(base, selected),
            "improvement_vs_always_accept_pct": pct(always, selected),
            "mean_keep_share_applied": part["keep_share_applied"].mean(),
        })
    return pd.DataFrame(rows)


def write_summary(path: Path, results, totals, bootstrap, feature_map):
    cols = ["policy", "test_window", "selected_candidate", "keep_share_applied", "base_mse", "always_accept_mse", "selected_mse", "improvement_vs_base_pct", "improvement_vs_always_accept_pct"]
    lines = [
        "# Leakage-safe selective policy benchmark v2", "", "## Protocol", "",
        "Model and threshold selection uses forward out-of-fold predictions from older windows only. The selected policy is refit on all older windows and evaluated once on W2 and W1.", "",
        "`precall` uses strictly last-observed features and can avoid inference. `postcall` uses response metadata and can veto an adjustment, but cannot save inference cost.", "",
        "## Results", "", results[cols].to_markdown(index=False, floatfmt=".6f"), "",
        "## Macro", "", totals.to_markdown(index=False, floatfmt=".6f"), "", "## Block bootstrap", "",
    ]
    for policy, item in bootstrap.items():
        lines += [f"- `{policy}` vs base: `{item['vs_base_ci_low']:.6f}` to `{item['vs_base_ci_high']:.6f}` (**{item['vs_base_conclusion']}**).",
                  f"- `{policy}` vs always accept: `{item['vs_always_ci_low']:.6f}` to `{item['vs_always_ci_high']:.6f}` (**{item['vs_always_conclusion']}**)."]
    lines += ["", "## Safe features", ""]
    lines += [f"- `{policy}`: {', '.join(f'`{x}`' for x in names)}" for policy, names in feature_map.items()]
    lines += ["", "## Limits", "",
              "- Thirty-step paths overlap; confidence intervals therefore use 30-origin circular blocks.",
              "- Only historical LLM-applied rows reveal the LLM counterfactual.",
              "- These windows end in 2025; current production claims require newly frozen rolling-origin tests."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("reports/agent_selective_dataset_v2/helpful_case_dataset_v2.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/agent_selective_policy_v2"))
    parser.add_argument("--mode", choices=("precall", "postcall", "both"), default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    frame, policies = pd.read_csv(args.input), ("precall", "postcall") if args.mode == "both" else (args.mode,)
    result_parts, decision_parts, audit_parts, feature_map = [], [], [], {}
    for policy in policies:
        result, decision, audit, features = run_policy(frame, policy, args.seed)
        result_parts.append(result); decision_parts.append(decision); audit_parts.append(audit); feature_map[policy] = features
    results, decisions, audits = pd.concat(result_parts), pd.concat(decision_parts), pd.concat(audit_parts)
    totals = aggregate(results)
    boot = {p: block_bootstrap(decisions[decisions["policy"] == p], args.bootstrap, 30, args.seed) for p in policies}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "results_by_window.csv", index=False)
    totals.to_csv(args.output_dir / "aggregate.csv", index=False)
    decisions.to_csv(args.output_dir / "case_decisions.csv", index=False)
    audits.to_csv(args.output_dir / "candidate_selection.csv", index=False)
    (args.output_dir / "bootstrap.json").write_text(json.dumps(boot, indent=2), encoding="utf-8")
    (args.output_dir / "manifest.json").write_text(json.dumps({"protocol": "causal expanding-window", "features": feature_map, "seed": args.seed}, indent=2), encoding="utf-8")
    write_summary(args.output_dir / "summary.md", results, totals, boot, feature_map)
    print(totals.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
