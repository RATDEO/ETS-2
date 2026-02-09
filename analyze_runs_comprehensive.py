#!/usr/bin/env python3
"""
Comprehensive analysis of experiment runs comparing best vs worst performing configurations.
"""
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

runs_dir = Path("runs")
results = []

for run_folder in sorted(runs_dir.iterdir()):
    if not run_folder.is_dir() or run_folder.name.startswith('.'):
        continue
    
    config_file = run_folder / "config_resolved.yaml"
    results_dir = run_folder / "results"
    
    if not config_file.exists():
        continue
    
    try:
        with open(config_file) as f:
            config = yaml.safe_load(f)
    except Exception:
        continue
    
    model = config.get('model', {})
    ts = config.get('time_series', {})
    llm = config.get('llm', {})
    features = config.get('features', {})
    
    run_data = {
        'run_id': run_folder.name,
        # Model hyperparameters
        'tsm_type': model.get('tsm_type'),
        'batch_size': model.get('batch_size'),
        'learning_rate': model.get('learning_rate'),
        'd_model': model.get('d_model'),
        'n_heads': model.get('n_heads'),
        'e_layers': model.get('e_layers'),
        'd_layers': model.get('d_layers'),
        'd_ff': model.get('d_ff'),
        'dropout': model.get('dropout'),
        'weight_decay': model.get('weight_decay'),
        'grad_clip': model.get('grad_clip'),
        'max_epochs': model.get('max_epochs'),
        'early_stopping_patience': model.get('early_stopping_patience'),
        # Time series config
        'seq_len': ts.get('seq_len'),
        'pred_len': ts.get('pred_len'),
        'label_len': ts.get('label_len'),
        # LLM config
        'llm_model': llm.get('model'),
        'llm_methods': str(llm.get('methods', [])),
        'llm_temperature': llm.get('temperature'),
        'llm_max_tokens': llm.get('max_tokens'),
        'blend_mode': llm.get('blend', {}).get('mode'),
        'blend_max_weight': llm.get('blend', {}).get('max_weight'),
        'cot_k_examples': llm.get('cot_rf', {}).get('k_examples'),
        'sentiment_enabled': llm.get('sentiment', {}).get('enabled'),
        # Features
        'include_brent': features.get('include_brent'),
        'include_coal': features.get('include_coal'),
        'rolling_vol_window': features.get('rolling_vol_window'),
    }
    
    # Create a config signature for grouping
    run_data['config_signature'] = (
        f"{model.get('tsm_type')}_d{model.get('d_model')}_lr{model.get('learning_rate')}_"
        f"e{model.get('e_layers')}_seq{ts.get('seq_len')}_drop{model.get('dropout')}"
    )
    
    # Get metrics from various files
    # 1. Long-short metrics (Sharpe, hit rate)
    ls_file = results_dir / "long_short_metrics.csv"
    if ls_file.exists():
        try:
            df = pd.read_csv(ls_file)
            for model_name in ['tsm', 'linear_ridge', 'linear_lasso', 'naive_persistence']:
                model_rows = df[df['model'] == model_name]
                for _, row in model_rows.iterrows():
                    h = int(row['horizon'])
                    run_data[f'{model_name}_sharpe_h{h}'] = row.get('sharpe')
                    run_data[f'{model_name}_hit_rate_h{h}'] = row.get('hit_rate')
                    run_data[f'{model_name}_mean_return_h{h}'] = row.get('mean_return')
        except Exception:
            pass
    
    # 2. Standard metrics
    for metric_file in ['linear_ridge_metrics.csv', 'linear_lasso_metrics.csv', 
                        'naive_persistence_metrics.csv', 'seasonal_naive_metrics.csv']:
        mf = results_dir / metric_file
        if mf.exists():
            try:
                df = pd.read_csv(mf)
                prefix = metric_file.replace('_metrics.csv', '')
                if 'horizon' in df.columns and 'mse' in df.columns:
                    for _, row in df.iterrows():
                        h = int(row['horizon'])
                        run_data[f'{prefix}_mse_h{h}'] = row.get('mse')
                        run_data[f'{prefix}_rmse_h{h}'] = row.get('rmse')
                        run_data[f'{prefix}_mae_h{h}'] = row.get('mae')
            except Exception:
                pass
    
    # 3. Drift metrics (for TSM specifically)
    drift_file = results_dir / "drift_metrics.csv"
    if drift_file.exists():
        try:
            df = pd.read_csv(drift_file)
            for model_name in ['tsm', 'linear_ridge', 'linear_lasso']:
                model_rows = df[df['model'].str.lower() == model_name]
                for _, row in model_rows.iterrows():
                    h = int(row['horizon'])
                    run_data[f'{model_name}_drift_mse_h{h}'] = row.get('mse')
        except Exception:
            pass
    
    results.append(run_data)

df = pd.DataFrame(results)
print(f"=" * 100)
print(f"COMPREHENSIVE EXPERIMENT ANALYSIS")
print(f"Total runs analyzed: {len(df)}")
print(f"=" * 100)

# Identify unique configuration groups
print(f"\n{'='*100}")
print("CONFIGURATION GROUPS")
print(f"{'='*100}")
config_groups = df.groupby('config_signature').size().sort_values(ascending=False)
print(f"\nUnique configurations: {len(config_groups)}")
for config, count in config_groups.items():
    print(f"  {config}: {count} runs")

# Compare model architectures (autoformer vs dlinear)
print(f"\n{'='*100}")
print("MODEL ARCHITECTURE COMPARISON: AUTOFORMER vs DLINEAR")
print(f"{'='*100}")

autoformer_runs = df[df['tsm_type'] == 'autoformer']
dlinear_runs = df[df['tsm_type'] == 'dlinear']

print(f"\nAutoformer runs: {len(autoformer_runs)}")
print(f"DLinear runs: {len(dlinear_runs)}")

# Check for TSM sharpe metrics
for metric in ['tsm_sharpe_h1', 'tsm_sharpe_h5', 'tsm_sharpe_h20', 'tsm_sharpe_h30']:
    if metric in df.columns:
        auto_vals = autoformer_runs[metric].dropna()
        dlin_vals = dlinear_runs[metric].dropna()
        if len(auto_vals) > 0 or len(dlin_vals) > 0:
            print(f"\n{metric}:")
            if len(auto_vals) > 0:
                print(f"  Autoformer: mean={auto_vals.mean():.4f}, std={auto_vals.std():.4f}, n={len(auto_vals)}")
            if len(dlin_vals) > 0:
                print(f"  DLinear: mean={dlin_vals.mean():.4f}, std={dlin_vals.std():.4f}, n={len(dlin_vals)}")

# Compare hyperparameter configurations
print(f"\n{'='*100}")
print("HYPERPARAMETER CONFIGURATION COMPARISON")
print(f"{'='*100}")

# Config 1: Large model (d_model=512, lr=0.0001, e_layers=2, seq_len=120, dropout=0.05)
# Config 2: Small model (d_model=64, lr=0.001, e_layers=1, seq_len=60, dropout=0.3)

large_model = df[(df['d_model'] == 512) & (df['learning_rate'] == 0.0001)]
small_model = df[(df['d_model'] == 64) & (df['learning_rate'] == 0.001)]

print(f"\nLarge Model Configuration (d_model=512, lr=0.0001): {len(large_model)} runs")
print(f"Small Model Configuration (d_model=64, lr=0.001): {len(small_model)} runs")

# Compare ridge MSE at different horizons
print(f"\n--- Linear Ridge Baseline MSE Comparison ---")
for h in [1, 5, 20, 30]:
    col = f'linear_ridge_mse_h{h}'
    if col in df.columns:
        large_vals = large_model[col].dropna()
        small_vals = small_model[col].dropna()
        if len(large_vals) > 0 and len(small_vals) > 0:
            print(f"Horizon {h}:")
            print(f"  Large Model: mean={large_vals.mean():.6f}, n={len(large_vals)}")
            print(f"  Small Model: mean={small_vals.mean():.6f}, n={len(small_vals)}")
            diff_pct = ((large_vals.mean() - small_vals.mean()) / small_vals.mean()) * 100
            print(f"  Difference: {diff_pct:+.2f}%")

# Compare by LLM method
print(f"\n{'='*100}")
print("LLM METHOD COMPARISON")
print(f"{'='*100}")

llm_method_groups = df.groupby('llm_methods')
print(f"\nUnique LLM methods: {len(llm_method_groups)}")

# Analyze TSM sharpe by LLM method (for runs that have it)
if 'tsm_sharpe_h1' in df.columns:
    df_with_sharpe = df[df['tsm_sharpe_h1'].notna()]
    if len(df_with_sharpe) > 0:
        print("\n--- TSM Sharpe by LLM Method ---")
        method_stats = df_with_sharpe.groupby('llm_methods').agg({
            'tsm_sharpe_h1': ['mean', 'std', 'count'],
            'tsm_hit_rate_h1': ['mean']
        })
        print(method_stats.to_string())

# Compare by LLM model
print(f"\n{'='*100}")
print("LLM PROVIDER COMPARISON")  
print(f"{'='*100}")

llm_model_groups = df.groupby('llm_model').size().sort_values(ascending=False)
print("\nLLM Models Used:")
for model, count in llm_model_groups.items():
    print(f"  {model}: {count} runs")

# Key findings summary
print(f"\n{'='*100}")
print("KEY CONFIGURATION DIFFERENCES IDENTIFIED")
print(f"{'='*100}")

print("""
HYPERPARAMETER GROUPS:

1. LARGE MODEL CONFIGURATION (early experiments):
   - d_model: 512
   - learning_rate: 0.0001  
   - e_layers: 2
   - seq_len: 120
   - dropout: 0.05
   - batch_size: 32
   - n_heads: 8
   - d_ff: 2048

2. SMALL MODEL CONFIGURATION (later experiments):
   - d_model: 64
   - learning_rate: 0.001
   - e_layers: 1
   - seq_len: 60
   - dropout: 0.3
   - batch_size: 16
   - n_heads: 2
   - d_ff: 128

MODEL ARCHITECTURES:
   - Autoformer: Transformer-based with auto-correlation
   - DLinear: Simple linear decomposition model

LLM METHODS TESTED:
   - TSM+LLM-DELTA-RETURNS
   - TSM+LLM-COT-SENT
   - TSM+LLM-COT-RF  
   - TSM+NEWS-DRIFT
   - NEWS-SENTIMENT-ONLY
   - DP, CoT, CoT-RF, TSM+LLM (ensemble)
""")

# Detailed comparison of available metrics
print(f"\n{'='*100}")
print("DETAILED METRICS BY CONFIGURATION GROUP")
print(f"{'='*100}")

for config_sig in config_groups.index[:5]:  # Top 5 config groups
    subset = df[df['config_signature'] == config_sig]
    print(f"\n--- {config_sig} ({len(subset)} runs) ---")
    
    # Get sample hyperparams
    sample = subset.iloc[0]
    print(f"TSM Type: {sample['tsm_type']}")
    print(f"LLM Models: {subset['llm_model'].unique().tolist()}")
    print(f"LLM Methods: {subset['llm_methods'].unique().tolist()[:3]}...")
    
    # Metrics summary
    for metric in ['linear_ridge_mse_h1', 'linear_ridge_mse_h30', 'tsm_sharpe_h1', 'tsm_hit_rate_h1']:
        if metric in subset.columns:
            vals = subset[metric].dropna()
            if len(vals) > 0:
                print(f"  {metric}: mean={vals.mean():.4f}, min={vals.min():.4f}, max={vals.max():.4f}")
