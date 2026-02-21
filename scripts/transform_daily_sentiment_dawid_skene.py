#!/usr/bin/env python3
"""Build daily sentiment via Dawid-Skene probabilistic vote aggregation."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd


LABELS = ["NO", "UNKNOWN", "YES"]
LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}
SCORES = np.array([-1.0, 0.0, 1.0], dtype=float)


def _parse_votes(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    parsed = None
    try:
        parsed = json.loads(text)
    except Exception:
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = None
    if isinstance(parsed, (list, tuple)):
        out = []
        for item in parsed:
            label = str(item).upper().strip()
            if label in LABEL_TO_IDX:
                out.append(label)
        return out
    label = text.upper().strip()
    return [label] if label in LABEL_TO_IDX else []


def _build_observation_matrix(
    df: pd.DataFrame,
    votes_col: str,
    fallback_vote_col: str | None,
    n_workers: int,
) -> np.ndarray:
    obs = -np.ones((len(df), n_workers), dtype=int)
    for i, row in enumerate(df.itertuples(index=False)):
        votes_raw = getattr(row, votes_col, None)
        votes = _parse_votes(votes_raw)
        if not votes and fallback_vote_col and fallback_vote_col in df.columns:
            votes = _parse_votes(getattr(row, fallback_vote_col, None))
        for j, label in enumerate(votes[:n_workers]):
            obs[i, j] = LABEL_TO_IDX[label]
    return obs


def _dawid_skene(
    obs: np.ndarray,
    n_classes: int,
    max_iter: int,
    tol: float,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, bool]:
    n_items, n_workers = obs.shape
    eps = 1e-12

    # Init posteriors by per-item majority vote (uniform when missing).
    q = np.full((n_items, n_classes), 1.0 / n_classes, dtype=float)
    for i in range(n_items):
        valid = obs[i] >= 0
        if not np.any(valid):
            continue
        labels = obs[i, valid]
        counts = np.bincount(labels, minlength=n_classes)
        major = int(np.argmax(counts))
        q[i, :] = 0.05 / max(1, n_classes - 1)
        q[i, major] = 0.95

    pi = q.mean(axis=0)
    theta = np.full((n_workers, n_classes, n_classes), 1.0 / n_classes, dtype=float)

    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        # M-step
        pi = q.mean(axis=0)
        pi = pi / max(pi.sum(), eps)
        for j in range(n_workers):
            valid = obs[:, j] >= 0
            for k in range(n_classes):
                denom = float(np.sum(q[valid, k]))
                if denom <= eps:
                    theta[j, k, :] = 1.0 / n_classes
                    continue
                numer = np.zeros(n_classes, dtype=float)
                for l in range(n_classes):
                    numer[l] = float(np.sum(q[(obs[:, j] == l), k]))
                theta[j, k, :] = (numer + alpha) / (denom + alpha * n_classes)

        # E-step
        q_prev = q.copy()
        log_theta = np.log(np.clip(theta, eps, 1.0))
        log_pi = np.log(np.clip(pi, eps, 1.0))
        for i in range(n_items):
            logp = log_pi.copy()
            for j in range(n_workers):
                l = int(obs[i, j])
                if l < 0:
                    continue
                logp += log_theta[j, :, l]
            m = float(np.max(logp))
            p = np.exp(logp - m)
            s = float(p.sum())
            if s <= eps:
                q[i, :] = 1.0 / n_classes
            else:
                q[i, :] = p / s

        delta = float(np.max(np.abs(q - q_prev)))
        if delta < tol:
            converged = True
            break

    return q, pi, theta, n_iter, converged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transform vote-level labels into daily sentiment using Dawid-Skene."
    )
    parser.add_argument(
        "--input",
        default="data/news/headlines_labeled_qwen_votes3_daily3.csv",
        help="Input labeled headlines CSV.",
    )
    parser.add_argument(
        "--output",
        default="data/news/daily_sentiment_qwen_votes3_daily3_dawidskene.csv",
        help="Output daily sentiment CSV.",
    )
    parser.add_argument("--date-col", default="seendate")
    parser.add_argument("--votes-col", default="llm_votes")
    parser.add_argument("--fallback-vote-col", default="llm_vote")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--posteriors-output",
        default="",
        help="Optional headline-level posteriors CSV path.",
    )
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.max_iter < 1:
        raise ValueError("--max-iter must be >= 1")
    if args.alpha < 0.0:
        raise ValueError("--alpha must be >= 0")

    df = pd.read_csv(args.input)
    for col in (args.date_col, args.votes_col):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.copy()
    df[args.date_col] = pd.to_datetime(df[args.date_col], errors="coerce").dt.normalize()
    df = df.dropna(subset=[args.date_col]).reset_index(drop=True)

    obs = _build_observation_matrix(
        df,
        votes_col=args.votes_col,
        fallback_vote_col=args.fallback_vote_col,
        n_workers=args.workers,
    )
    q, pi, theta, n_iter, converged = _dawid_skene(
        obs=obs,
        n_classes=len(LABELS),
        max_iter=args.max_iter,
        tol=args.tol,
        alpha=float(args.alpha),
    )

    exp_score = q @ SCORES
    df["_ds_score"] = exp_score
    daily = (
        df.groupby(args.date_col, as_index=False)["_ds_score"]
        .mean()
        .rename(columns={"_ds_score": "sent_score"})
        .sort_values(args.date_col)
    )
    daily[args.date_col] = pd.to_datetime(daily[args.date_col]).dt.strftime("%Y-%m-%d")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output_path, index=False)

    if args.posteriors_output:
        post = df[[args.date_col]].copy()
        post["p_no"] = q[:, LABEL_TO_IDX["NO"]]
        post["p_unknown"] = q[:, LABEL_TO_IDX["UNKNOWN"]]
        post["p_yes"] = q[:, LABEL_TO_IDX["YES"]]
        post["sent_score_ds"] = exp_score
        post_path = Path(args.posteriors_output)
        post_path.parent.mkdir(parents=True, exist_ok=True)
        post.to_csv(post_path, index=False)

    mean_conf = float(np.mean(np.max(q, axis=1)))
    print(f"Saved Dawid-Skene daily sentiment to {output_path}")
    print(
        "stats:",
        f"rows={len(daily)}",
        f"mean={daily['sent_score'].mean():.6f}",
        f"std={daily['sent_score'].std():.6f}",
        f"min={daily['sent_score'].min():.6f}",
        f"max={daily['sent_score'].max():.6f}",
    )
    print(
        "em:",
        f"iters={n_iter}",
        f"converged={converged}",
        f"class_priors={dict(zip(LABELS, [float(x) for x in pi]))}",
        f"mean_posterior_conf={mean_conf:.6f}",
    )
    # Print worker diagonal quality as quick diagnostics.
    for j in range(theta.shape[0]):
        diag = [float(theta[j, k, k]) for k in range(theta.shape[1])]
        print(f"worker_{j}_diag={diag}")


if __name__ == "__main__":
    main()

