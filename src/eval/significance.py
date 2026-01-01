"""
Statistical significance testing for model comparison.

Implements paired t-test and Wilcoxon signed-rank test
as per the original paper methodology.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
import logging

logger = logging.getLogger(__name__)


def paired_t_test(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    alternative: str = "two-sided"
) -> Tuple[float, float]:
    """
    Perform paired t-test on squared errors.
    
    Tests if model A has significantly different errors than model B.
    
    Args:
        errors_a: Squared errors from model A
        errors_b: Squared errors from model B
        alternative: 'two-sided', 'less', or 'greater'
        
    Returns:
        Tuple of (t-statistic, p-value)
    """
    if len(errors_a) != len(errors_b):
        raise ValueError("Error arrays must have same length")
    
    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(errors_a, errors_b, alternative=alternative)
    
    return float(t_stat), float(p_value)


def wilcoxon_test(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    alternative: str = "two-sided"
) -> Tuple[float, float]:
    """
    Perform Wilcoxon signed-rank test on squared errors.
    
    Non-parametric alternative to paired t-test.
    
    Args:
        errors_a: Squared errors from model A
        errors_b: Squared errors from model B
        alternative: 'two-sided', 'less', or 'greater'
        
    Returns:
        Tuple of (test-statistic, p-value)
    """
    if len(errors_a) != len(errors_b):
        raise ValueError("Error arrays must have same length")
    
    # Compute differences
    diff = errors_a - errors_b
    
    # Remove zeros (ties)
    diff = diff[diff != 0]
    
    if len(diff) < 10:
        logger.warning(f"Too few non-zero differences for Wilcoxon test: {len(diff)}")
        return np.nan, np.nan
    
    try:
        stat, p_value = stats.wilcoxon(diff, alternative=alternative)
        return float(stat), float(p_value)
    except Exception as e:
        logger.warning(f"Wilcoxon test failed: {e}")
        return np.nan, np.nan


def diebold_mariano_test(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    h: int = 1,
    alternative: str = "two-sided"
) -> Tuple[float, float]:
    """
    Perform Diebold-Mariano test for predictive accuracy comparison.
    
    Args:
        errors_a: Squared errors from model A
        errors_b: Squared errors from model B
        h: Forecast horizon (for variance adjustment)
        alternative: 'two-sided', 'less', or 'greater'
        
    Returns:
        Tuple of (DM-statistic, p-value)
    """
    d = errors_a - errors_b
    n = len(d)
    
    # Mean and variance of loss differential
    d_mean = np.mean(d)
    
    # Newey-West variance estimator (for h-step ahead forecasts)
    d_var = np.var(d, ddof=1)
    
    # Add autocorrelation terms
    for k in range(1, h):
        if n - k > 0:
            gamma_k = np.mean((d[k:] - d_mean) * (d[:-k] - d_mean))
            d_var += 2 * gamma_k
    
    d_var = max(d_var, 1e-10)  # Prevent division by zero
    
    # DM statistic
    dm_stat = d_mean / np.sqrt(d_var / n)
    
    # P-value (asymptotically normal)
    if alternative == "two-sided":
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    elif alternative == "less":
        p_value = stats.norm.cdf(dm_stat)
    else:  # greater
        p_value = 1 - stats.norm.cdf(dm_stat)
    
    return float(dm_stat), float(p_value)


def run_significance_tests(
    errors_baseline: np.ndarray,
    errors_model: np.ndarray,
    horizon: int = 1,
    alpha: float = 0.05,
    model_name: str = "model",
    baseline_name: str = "baseline"
) -> Dict[str, any]:
    """
    Run all significance tests comparing a model to a baseline.
    
    Args:
        errors_baseline: Squared errors from baseline
        errors_model: Squared errors from model
        horizon: Forecast horizon
        alpha: Significance level
        model_name: Name of the model being tested
        baseline_name: Name of the baseline
        
    Returns:
        Dictionary with test results
    """
    # Mean errors
    mean_baseline = np.mean(errors_baseline)
    mean_model = np.mean(errors_model)
    
    # Tests
    t_stat, t_pval = paired_t_test(errors_model, errors_baseline)
    w_stat, w_pval = wilcoxon_test(errors_model, errors_baseline)
    dm_stat, dm_pval = diebold_mariano_test(errors_model, errors_baseline, h=horizon)
    
    # Determine if model is significantly better
    model_better = mean_model < mean_baseline
    
    result = {
        "horizon": horizon,
        "model": model_name,
        "baseline": baseline_name,
        "mean_error_model": mean_model,
        "mean_error_baseline": mean_baseline,
        "improvement_pct": (mean_baseline - mean_model) / mean_baseline * 100,
        "t_statistic": t_stat,
        "t_pvalue": t_pval,
        "t_significant": t_pval < alpha,
        "wilcoxon_statistic": w_stat,
        "wilcoxon_pvalue": w_pval,
        "wilcoxon_significant": w_pval < alpha if not np.isnan(w_pval) else False,
        "dm_statistic": dm_stat,
        "dm_pvalue": dm_pval,
        "dm_significant": dm_pval < alpha,
        "model_better": model_better
    }
    
    return result


def compare_all_models(
    errors_dict: Dict[str, np.ndarray],
    baseline_name: str,
    horizons: List[int] = [1, 5, 20, 30],
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Compare all models against a baseline.
    
    Args:
        errors_dict: Dictionary mapping model_name -> {horizon: errors_array}
        baseline_name: Name of baseline model
        horizons: Horizons to compare
        alpha: Significance level
        
    Returns:
        DataFrame with all comparison results
    """
    if baseline_name not in errors_dict:
        raise ValueError(f"Baseline '{baseline_name}' not found in errors_dict")
    
    results = []
    
    for model_name, model_errors in errors_dict.items():
        if model_name == baseline_name:
            continue
        
        for h in horizons:
            if h not in model_errors or h not in errors_dict[baseline_name]:
                continue
            
            result = run_significance_tests(
                errors_baseline=errors_dict[baseline_name][h],
                errors_model=model_errors[h],
                horizon=h,
                alpha=alpha,
                model_name=model_name,
                baseline_name=baseline_name
            )
            results.append(result)
    
    return pd.DataFrame(results)


def format_significance_table(
    results_df: pd.DataFrame,
    metric: str = "improvement_pct"
) -> pd.DataFrame:
    """
    Format significance results as a publication-ready table.
    
    Args:
        results_df: DataFrame from compare_all_models()
        metric: Metric to display
        
    Returns:
        Formatted DataFrame
    """
    # Pivot to wide format
    pivot = results_df.pivot(
        index="model",
        columns="horizon",
        values=metric
    )
    
    # Add significance markers
    for h in pivot.columns:
        mask = results_df["horizon"] == h
        for model in pivot.index:
            model_mask = mask & (results_df["model"] == model)
            if model_mask.any():
                row = results_df[model_mask].iloc[0]
                value = pivot.loc[model, h]
                
                if row["t_significant"]:
                    if row["wilcoxon_significant"]:
                        pivot.loc[model, h] = f"{value:.2f}**"
                    else:
                        pivot.loc[model, h] = f"{value:.2f}*"
                else:
                    pivot.loc[model, h] = f"{value:.2f}"
    
    return pivot
