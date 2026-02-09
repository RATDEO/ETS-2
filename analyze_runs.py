import os
import yaml
import pandas as pd
from pathlib import Path
import json
import numpy as np

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
    except Exception as e:
        continue
    
    # Look for various metrics files
    metrics_files = list(results_dir.glob("*_metrics.csv")) if results_dir.exists() else []
    
    run_data = {
        'run_id': run_folder.name,
        'tsm_type': config.get('model', {}).get('tsm_type', 'N/A'),
        'batch_size': config.get('model', {}).get('batch_size', 'N/A'),
        'learning_rate': config.get('model', {}).get('learning_rate', 'N/A'),
        'd_model': config.get('model', {}).get('d_model', 'N/A'),
        'n_heads': config.get('model', {}).get('n_heads', 'N/A'),
        'e_layers': config.get('model', {}).get('e_layers', 'N/A'),
        'd_layers': config.get('model', {}).get('d_layers', 'N/A'),
        'd_ff': config.get('model', {}).get('d_ff', 'N/A'),
        'dropout': config.get('model', {}).get('dropout', 'N/A'),
        'weight_decay': config.get('model', {}).get('weight_decay', 'N/A'),
        'grad_clip': config.get('model', {}).get('grad_clip', 'N/A'),
        'max_epochs': config.get('model', {}).get('max_epochs', 'N/A'),
        'early_stopping_patience': config.get('model', {}).get('early_stopping_patience', 'N/A'),
        'seq_len': config.get('time_series', {}).get('seq_len', 'N/A'),
        'pred_len': config.get('time_series', {}).get('pred_len', 'N/A'),
        'label_len': config.get('time_series', {}).get('label_len', 'N/A'),
        'llm_model': config.get('llm', {}).get('model', 'N/A'),
        'llm_methods': str(config.get('llm', {}).get('methods', 'N/A')),
        'include_brent': config.get('features', {}).get('include_brent', 'N/A'),
        'include_coal': config.get('features', {}).get('include_coal', 'N/A'),
        'rolling_vol_window': config.get('features', {}).get('rolling_vol_window', 'N/A'),
    }
    
    # Get primary metrics (from ridge or autoformer metrics)
    for mf in metrics_files:
        try:
            df = pd.read_csv(mf)
            prefix = mf.stem.replace('_metrics', '')
            if 'horizon' in df.columns and 'mse' in df.columns:
                for _, row in df.iterrows():
                    h = int(row['horizon'])
                    run_data[f'{prefix}_mse_h{h}'] = row['mse']
                    run_data[f'{prefix}_rmse_h{h}'] = row.get('rmse', None)
                    run_data[f'{prefix}_mae_h{h}'] = row.get('mae', None)
        except Exception as e:
            pass
    
    results.append(run_data)

df = pd.DataFrame(results)
print(f"Total runs analyzed: {len(df)}")
print(f"\n{'='*80}")
print("AVAILABLE METRIC COLUMNS:")
print('='*80)
mse_cols = [c for c in df.columns if 'mse' in c]
print(mse_cols[:30])

# Find a common metric to rank by
# Prefer linear_ridge as it's more comparable across runs
key_metric = None
for candidate in ['linear_ridge_mse_h1', 'linear_lasso_mse_h1', 'naive_persistence_mse_h1']:
    if candidate in df.columns:
        key_metric = candidate
        break

if key_metric:
    # Filter runs that have this metric
    df_valid = df[df[key_metric].notna()].copy()
    df_valid = df_valid.sort_values(key_metric)
    
    print(f"\n{'='*80}")
    print(f"RANKING BY: {key_metric}")
    print(f"Runs with valid metrics: {len(df_valid)}")
    print('='*80)
    
    # Top 10 best performers
    print("\n*** TOP 10 BEST PERFORMERS (lowest MSE h1) ***")
    best = df_valid.head(10)
    for i, (_, row) in enumerate(best.iterrows(), 1):
        print(f"\n{i}. Run: {row['run_id']}")
        print(f"   MSE h1: {row[key_metric]:.6f}")
        print(f"   Model: {row['tsm_type']}, LR: {row['learning_rate']}, d_model: {row['d_model']}")
        print(f"   Batch: {row['batch_size']}, Dropout: {row['dropout']}, Weight Decay: {row['weight_decay']}")
        print(f"   Layers (e/d): {row['e_layers']}/{row['d_layers']}, n_heads: {row['n_heads']}, d_ff: {row['d_ff']}")
        print(f"   Seq/Pred/Label len: {row['seq_len']}/{row['pred_len']}/{row['label_len']}")
        print(f"   LLM: {row['llm_model']}, Methods: {row['llm_methods'][:50]}...")
    
    # Bottom 10 worst performers
    print(f"\n\n*** TOP 10 WORST PERFORMERS (highest MSE h1) ***")
    worst = df_valid.tail(10).iloc[::-1]
    for i, (_, row) in enumerate(worst.iterrows(), 1):
        print(f"\n{i}. Run: {row['run_id']}")
        print(f"   MSE h1: {row[key_metric]:.6f}")
        print(f"   Model: {row['tsm_type']}, LR: {row['learning_rate']}, d_model: {row['d_model']}")
        print(f"   Batch: {row['batch_size']}, Dropout: {row['dropout']}, Weight Decay: {row['weight_decay']}")
        print(f"   Layers (e/d): {row['e_layers']}/{row['d_layers']}, n_heads: {row['n_heads']}, d_ff: {row['d_ff']}")
        print(f"   Seq/Pred/Label len: {row['seq_len']}/{row['pred_len']}/{row['label_len']}")
        print(f"   LLM: {row['llm_model']}, Methods: {row['llm_methods'][:50]}...")
    
    # Analysis of hyperparameter impact
    print(f"\n\n{'='*80}")
    print("HYPERPARAMETER IMPACT ANALYSIS")
    print('='*80)
    
    # Split into top/bottom quartiles
    q1 = df_valid[key_metric].quantile(0.25)
    q3 = df_valid[key_metric].quantile(0.75)
    top_quartile = df_valid[df_valid[key_metric] <= q1]
    bottom_quartile = df_valid[df_valid[key_metric] >= q3]
    
    print(f"\nTop quartile (best): {len(top_quartile)} runs with MSE <= {q1:.4f}")
    print(f"Bottom quartile (worst): {len(bottom_quartile)} runs with MSE >= {q3:.4f}")
    
    # Compare numeric parameters
    numeric_params = ['learning_rate', 'd_model', 'batch_size', 'dropout', 'weight_decay', 
                      'e_layers', 'd_layers', 'n_heads', 'd_ff', 'seq_len', 'pred_len', 'label_len',
                      'grad_clip', 'max_epochs', 'early_stopping_patience']
    
    print("\n| Parameter | Best (mean) | Worst (mean) | Diff |")
    print("|-----------|-------------|--------------|------|")
    
    for param in numeric_params:
        if param in df_valid.columns:
            top_vals = pd.to_numeric(top_quartile[param], errors='coerce')
            bot_vals = pd.to_numeric(bottom_quartile[param], errors='coerce')
            if top_vals.notna().sum() > 0 and bot_vals.notna().sum() > 0:
                top_mean = top_vals.mean()
                bot_mean = bot_vals.mean()
                diff = bot_mean - top_mean
                print(f"| {param:24} | {top_mean:11.4f} | {bot_mean:12.4f} | {diff:+.4f} |")
    
    # Categorical analysis
    print("\n\nCATEGORICAL PARAMETER DISTRIBUTION:")
    print("-" * 40)
    
    for param in ['tsm_type', 'llm_model']:
        if param in df_valid.columns:
            print(f"\n{param} in BEST quartile:")
            print(top_quartile[param].value_counts())
            print(f"\n{param} in WORST quartile:")
            print(bottom_quartile[param].value_counts())

# Also look at longer horizons if available
for h in [5, 20, 30]:
    h_metric = f'linear_ridge_mse_h{h}'
    if h_metric in df.columns:
        df_h = df[df[h_metric].notna()]
        if len(df_h) > 0:
            print(f"\n\n{'='*80}")
            print(f"HORIZON {h} ANALYSIS")
            print('='*80)
            print(f"Best MSE h{h}: {df_h[h_metric].min():.4f} (run: {df_h.loc[df_h[h_metric].idxmin(), 'run_id']})")
            print(f"Worst MSE h{h}: {df_h[h_metric].max():.4f} (run: {df_h.loc[df_h[h_metric].idxmax(), 'run_id']})")
            print(f"Mean MSE h{h}: {df_h[h_metric].mean():.4f}")
            print(f"Std MSE h{h}: {df_h[h_metric].std():.4f}")
