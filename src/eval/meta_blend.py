"""
Lightweight per-horizon calibration layer for combining TSM and LLM forecasts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def _design_matrix(tsm_pred: np.ndarray, llm_pred: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            np.asarray(tsm_pred, dtype=float),
            np.asarray(llm_pred, dtype=float),
            np.ones(len(tsm_pred), dtype=float),
        ]
    )


def _fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    eye = np.eye(X.shape[1], dtype=float)
    eye[-1, -1] = 0.0  # do not regularize intercept
    return np.linalg.solve(X.T @ X + alpha * eye, X.T @ y)


def _blocked_fold_indices(n_samples: int, n_folds: int = 4) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_samples < max(n_folds, 8):
        split = max(1, n_samples // 2)
        return [(np.arange(split, dtype=int), np.arange(split, n_samples, dtype=int))]
    folds = np.array_split(np.arange(n_samples, dtype=int), n_folds)
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(1, len(folds)):
        train_idx = np.concatenate(folds[:i])
        val_idx = folds[i]
        if len(train_idx) and len(val_idx):
            result.append((train_idx, val_idx))
    return result


@dataclass
class HorizonBlendResult:
    horizon: int
    alpha: float
    weights: np.ndarray


@dataclass
class OuterFoldResult:
    fold_index: int
    train_size: int
    val_size: int
    path_mse_tsm: float
    path_mse_llm: float
    path_mse_blend: float


class MetaBlender:
    """Per-horizon ridge combiner for TSM and LLM forecasts."""

    def __init__(self, alphas: Iterable[float] | None = None, n_folds: int = 4):
        self.alphas = list(alphas or [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0])
        self.n_folds = int(n_folds)
        self.results_: list[HorizonBlendResult] = []

    def fit(self, tsm_pred: np.ndarray, llm_pred: np.ndarray, y_true: np.ndarray) -> "MetaBlender":
        tsm_pred = np.asarray(tsm_pred, dtype=float)
        llm_pred = np.asarray(llm_pred, dtype=float)
        y_true = np.asarray(y_true, dtype=float)
        if tsm_pred.shape != llm_pred.shape or tsm_pred.shape != y_true.shape:
            raise ValueError("All inputs must have matching shapes.")

        self.results_ = []
        folds = _blocked_fold_indices(tsm_pred.shape[0], self.n_folds)

        for h in range(tsm_pred.shape[1]):
            X = _design_matrix(tsm_pred[:, h], llm_pred[:, h])
            y = y_true[:, h]

            best_alpha = self.alphas[0]
            best_mse = float("inf")
            for alpha in self.alphas:
                fold_mses = []
                for train_idx, val_idx in folds:
                    w = _fit_ridge(X[train_idx], y[train_idx], alpha)
                    pred = X[val_idx] @ w
                    fold_mses.append(float(np.mean((y[val_idx] - pred) ** 2)))
                mse = float(np.mean(fold_mses))
                if mse < best_mse:
                    best_mse = mse
                    best_alpha = float(alpha)

            weights = _fit_ridge(X, y, best_alpha)
            self.results_.append(HorizonBlendResult(horizon=h + 1, alpha=best_alpha, weights=weights))
        return self

    def predict(self, tsm_pred: np.ndarray, llm_pred: np.ndarray) -> np.ndarray:
        if not self.results_:
            raise RuntimeError("MetaBlender must be fit before predict().")
        tsm_pred = np.asarray(tsm_pred, dtype=float)
        llm_pred = np.asarray(llm_pred, dtype=float)
        if tsm_pred.shape != llm_pred.shape:
            raise ValueError("Prediction inputs must have matching shapes.")

        yhat = np.zeros_like(tsm_pred, dtype=float)
        for result in self.results_:
            h = result.horizon - 1
            X = _design_matrix(tsm_pred[:, h], llm_pred[:, h])
            yhat[:, h] = X @ result.weights
        return yhat

    def summary_rows(self) -> list[dict[str, float]]:
        rows = []
        for result in self.results_:
            rows.append(
                {
                    "horizon": float(result.horizon),
                    "alpha": float(result.alpha),
                    "w_tsm": float(result.weights[0]),
                    "w_llm": float(result.weights[1]),
                    "bias": float(result.weights[2]),
                }
            )
        return rows


class StatefulMetaBlender:
    """Per-horizon ridge combiner with state-dependent interaction features."""

    FEATURE_NAMES = (
        "tsm",
        "llm",
        "div",
        "vol20",
        "sent3",
        "tsm_x_vol20",
        "llm_x_vol20",
        "tsm_x_sent3",
        "llm_x_sent3",
        "tsm_x_div",
        "llm_x_div",
        "bias",
    )

    def __init__(self, alphas: Iterable[float] | None = None, n_folds: int = 4):
        self.alphas = list(alphas or [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0])
        self.n_folds = int(n_folds)
        self.results_: list[HorizonBlendResult] = []
        self.scalers_: list[tuple[np.ndarray, np.ndarray]] = []

    @staticmethod
    def _build_features(tsm_pred: np.ndarray, llm_pred: np.ndarray, state_features: np.ndarray) -> np.ndarray:
        tsm_pred = np.asarray(tsm_pred, dtype=float)
        llm_pred = np.asarray(llm_pred, dtype=float)
        state = np.asarray(state_features, dtype=float)
        if state.ndim != 2 or state.shape[1] < 2:
            raise ValueError("state_features must have shape (n_samples, >=2) for vol20 and sent3.")
        vol20 = state[:, 0]
        sent3 = state[:, 1]
        div = np.abs(llm_pred - tsm_pred)
        return np.column_stack(
            [
                tsm_pred,
                llm_pred,
                div,
                vol20,
                sent3,
                tsm_pred * vol20,
                llm_pred * vol20,
                tsm_pred * sent3,
                llm_pred * sent3,
                tsm_pred * div,
                llm_pred * div,
                np.ones(len(tsm_pred), dtype=float),
            ]
        )

    @staticmethod
    def _standardize_train(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = np.mean(X[:, :-1], axis=0)
        std = np.std(X[:, :-1], axis=0)
        std[std < 1e-6] = 1.0
        Xs = X.copy()
        Xs[:, :-1] = (Xs[:, :-1] - mean) / std
        return Xs, mean, std

    @staticmethod
    def _standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        Xs = X.copy()
        Xs[:, :-1] = (Xs[:, :-1] - mean) / std
        return Xs

    def fit(
        self,
        tsm_pred: np.ndarray,
        llm_pred: np.ndarray,
        y_true: np.ndarray,
        state_features: np.ndarray,
    ) -> "StatefulMetaBlender":
        tsm_pred = np.asarray(tsm_pred, dtype=float)
        llm_pred = np.asarray(llm_pred, dtype=float)
        y_true = np.asarray(y_true, dtype=float)
        state_features = np.asarray(state_features, dtype=float)
        if tsm_pred.shape != llm_pred.shape or tsm_pred.shape != y_true.shape:
            raise ValueError("All prediction inputs must have matching shapes.")
        if state_features.shape[0] != tsm_pred.shape[0]:
            raise ValueError("state_features must have the same number of rows as predictions.")

        self.results_ = []
        self.scalers_ = []
        folds = _blocked_fold_indices(tsm_pred.shape[0], self.n_folds)

        for h in range(tsm_pred.shape[1]):
            X = self._build_features(tsm_pred[:, h], llm_pred[:, h], state_features)
            y = y_true[:, h]

            best_alpha = self.alphas[0]
            best_mse = float("inf")
            for alpha in self.alphas:
                fold_mses = []
                for train_idx, val_idx in folds:
                    X_train, mean, std = self._standardize_train(X[train_idx])
                    w = _fit_ridge(X_train, y[train_idx], alpha)
                    X_val = self._standardize_apply(X[val_idx], mean, std)
                    pred = X_val @ w
                    fold_mses.append(float(np.mean((y[val_idx] - pred) ** 2)))
                mse = float(np.mean(fold_mses))
                if mse < best_mse:
                    best_mse = mse
                    best_alpha = float(alpha)

            X_all, mean, std = self._standardize_train(X)
            weights = _fit_ridge(X_all, y, best_alpha)
            self.results_.append(HorizonBlendResult(horizon=h + 1, alpha=best_alpha, weights=weights))
            self.scalers_.append((mean, std))
        return self

    def predict(self, tsm_pred: np.ndarray, llm_pred: np.ndarray, state_features: np.ndarray) -> np.ndarray:
        if not self.results_:
            raise RuntimeError("StatefulMetaBlender must be fit before predict().")
        tsm_pred = np.asarray(tsm_pred, dtype=float)
        llm_pred = np.asarray(llm_pred, dtype=float)
        state_features = np.asarray(state_features, dtype=float)
        if tsm_pred.shape != llm_pred.shape:
            raise ValueError("Prediction inputs must have matching shapes.")
        if state_features.shape[0] != tsm_pred.shape[0]:
            raise ValueError("state_features must have the same number of rows as predictions.")

        yhat = np.zeros_like(tsm_pred, dtype=float)
        for result, (mean, std) in zip(self.results_, self.scalers_, strict=False):
            h = result.horizon - 1
            X = self._build_features(tsm_pred[:, h], llm_pred[:, h], state_features)
            Xs = self._standardize_apply(X, mean, std)
            yhat[:, h] = Xs @ result.weights
        return yhat

    def summary_rows(self) -> list[dict[str, float]]:
        rows = []
        for result in self.results_:
            row = {
                "horizon": float(result.horizon),
                "alpha": float(result.alpha),
            }
            for name, weight in zip(self.FEATURE_NAMES, result.weights, strict=False):
                row[name] = float(weight)
            rows.append(row)
        return rows


def rolling_meta_blend_cv(
    tsm_pred: np.ndarray,
    llm_pred: np.ndarray,
    y_true: np.ndarray,
    *,
    alphas: Iterable[float] | None = None,
    n_outer_folds: int = 4,
    n_inner_folds: int = 4,
) -> tuple[np.ndarray, list[OuterFoldResult]]:
    """Evaluate MetaBlender with outer rolling folds and inner alpha selection."""
    tsm_pred = np.asarray(tsm_pred, dtype=float)
    llm_pred = np.asarray(llm_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    if tsm_pred.shape != llm_pred.shape or tsm_pred.shape != y_true.shape:
        raise ValueError("All inputs must have matching shapes.")

    outer_folds = _blocked_fold_indices(tsm_pred.shape[0], n_outer_folds)
    preds = np.full_like(y_true, np.nan, dtype=float)
    results: list[OuterFoldResult] = []

    for fold_index, (train_idx, val_idx) in enumerate(outer_folds, start=1):
        blender = MetaBlender(alphas=alphas, n_folds=n_inner_folds)
        blender.fit(tsm_pred[train_idx], llm_pred[train_idx], y_true[train_idx])
        fold_pred = blender.predict(tsm_pred[val_idx], llm_pred[val_idx])
        preds[val_idx] = fold_pred

        fold_true = y_true[val_idx]
        tsm_mse = float(np.mean((fold_true - tsm_pred[val_idx]) ** 2))
        llm_mse = float(np.mean((fold_true - llm_pred[val_idx]) ** 2))
        blend_mse = float(np.mean((fold_true - fold_pred) ** 2))
        results.append(
            OuterFoldResult(
                fold_index=fold_index,
                train_size=int(len(train_idx)),
                val_size=int(len(val_idx)),
                path_mse_tsm=tsm_mse,
                path_mse_llm=llm_mse,
                path_mse_blend=blend_mse,
            )
        )

    return preds, results
