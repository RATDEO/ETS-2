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
        'blend_mode': config.get('llm', {}).get('blend', {}).get('mode', 'N/A'),
        'blend_max_weight': config.get('llm', {}).get('blend', {}).get('max_weight', 'N/A'),
        'cot_k_examples': config.get('llm', {}).get('cot_rf', {}).get('k_examples', 'N/A'),
        'sentiment_enabled': config.get('llm', {}).get('sentiment', {}).get('enabled', 'N/A'),
        'sentiment_history_points': config.get('llm', {}).get('sentiment', {}).get('history_points', 'N/A'),
    }
    
    # Get long_short metrics for TSM (Sharpe ratio is a good performance indicator)
    long_short_file = results_dir / "long_short_metrics.csv"
    if long_short_file.exists():
        try:
            df = pd.read_csv(long_short_file)
            tsm_rows = df[df['model'] == 'tsm']
            for _, row in tsm_rows.iterrows():
                h = int(row['horizon'])
                run_data[f'tsm_sharpe_h{h}'] = row.get('sharpe', None)
                run_data[f'tsm_hit_rate_h{h}'] = row.get('hit_rate', None)
                run_data[f'tsm_mean_return_h{h}'] = row.get('mean_return', None)
        except Exception as e:
            pass
    
    # Get drift metrics for TSM
    drift_file = results_dir / "drift_metrics.csv"
    if drift_file.exists():
        try:
            df = pd.read_csv(drift_file)
            # Look for TSM-specific or LLM-blended rows
            for model_name in ['tsm', 'tsm+llm', 'blended', 'llm']:
                model_rows = df[df['model'].str.lower().str.contains(model_name, na=False)]
                if len(model_rows) > 0:
                    for _, row in model_rows.iterrows():
                        h = int(row['horizon'])
                        prefix = model_name.replace('+', '_')
                        run_data[f'{prefix}_drift_mse_h{h}'] = row.get('mse', None)
                        run_data[f'{prefix}_drift_mae_h{h}'] = row.get('mae', None)
        except Exception as e:
            pass
    
    results.append(run_data)

df = pd.DataFrame(results)
print(f"Total runs analyzed: {len(df)}")

# Find runs with TSM sharpe ratio data
sharpe_cols = [c for c in df.columns if 'sharpe' in c]
hit_cols = [c for c in df.columns if 'hit_rate' in c]
return_cols = [c for c in df.columns if 'mean_return' in c]

print(f"\nSharpe columns: {sharpe_cols}")
print(f"Hit rate columns: {hit_cols}")
print(f"Return columns: {return_cols}")

# Analyze by TSM Sharpe at h1 (day-ahead)
if 'tsm_sharpe_h1' in df.columns:
    df_valid = df[df['tsm_sharpe_h1'].notna()].copy()
    df_valid = df_valid.sort_values('tsm_sharpe_h1', ascending=False)  # Higher Sharpe = better
    
    print(f"\n{'='*80}")
    print(f"RANKING BY: tsm_sharpe_h1 (Higher is Better)")
    print(f"Runs with valid TSM Sharpe metrics: {len(df_valid)}")
    print('='*80)
    
    if len(df_valid) > 0:
        # Top 10 best performers (highest Sharpe)
        print("\n*** TOP 10 BEST PERFORMERS (highest Sharpe h1) ***")
        best = df_valid.head(10)
        for i, (_, row) in enumerate(best.iterrows(), 1):
            print(f"\n{i}. Run: {row['run_id']}")
            print(f"   Sharpe h1: {row['tsm_sharpe_h1']:.4f}, Hit Rate: {row.get('tsm_hit_rate_h1', 'N/A'):.2%}")
            print(f"   Model: {row['tsm_type']}, LR: {row['learning_rate']}, d_model: {row['d_model']}")
            print(f"   Dropout: {row['dropout']}, Weight Decay: {row['weight_decay']}")
            print(f"   LLM Methods: {row['llm_methods']}")
            print(f"   Blend: {row['blend_mode']}, Max Weight: {row['blend_max_weight']}")
        
        # Bottom 10 worst performers (lowest Sharpe)
        print(f"\n\n*** TOP 10 WORST PERFORMERS (lowest Sharpe h1) ***")
        worst = df_valid.tail(10).iloc[::-1]
        for i, (_, row) in enumerate(worst.iterrows(), 1):
            print(f"\n{i}. Run: {row['run_id']}")
            print(f"   Sharpe h1: {row['tsm_sharpe_h1']:.4f}, Hit Rate: {row.get('tsm_hit_rate_h1', 'N/A'):.2%}")
            print(f"   Model: {row['tsm_type']}, LR: {row['learning_rate']}, d_model: {row['d_model']}")
            print(f"   Dropout: {row['dropout']}, Weight Decay: {row['weight_decay']}")
            print(f"   LLM Methods: {row['llm_methods']}")
            print(f"   Blend: {row['blend_mode']}, Max Weight: {row['blend_max_weight']}")
        
        # Statistics
        print(f"\n\n{'='*80}")
        print("TSM SHARPE STATISTICS BY HORIZON")
        print('='*80)
        for h in [1, 5, 20, 30]:
            col = f'tsm_sharpe_h{h}'
            if col in df_valid.columns:
                vals = df_valid[col].dropna()
                if len(vals) > 0:
                    print(f"\nHorizon {h}:")
                    print(f"  Mean: {vals.mean():.4f}, Std: {vals.std():.4f}")
                    print(f"  Min: {vals.min():.4f}, Max: {vals.max():.4f}")
                    print(f"  Median: {vals.median():.4f}")

# Check for variation in hyperparameters
print(f"\n\n{'='*80}")
print("HYPERPARAMETER VARIATION CHECK")
print('='*80)

numeric_params = ['learning_rate', 'd_model', 'batch_size', 'dropout', 'weight_decay', 
                  'e_layers', 'd_layers', 'n_heads', 'd_ff', 'seq_len', 'pred_len', 'label_len']

for param in numeric_params:
    if param in df.columns:
        vals = pd.to_numeric(df[param], errors='coerce').dropna()
        unique_vals = vals.unique()
        if len(unique_vals) > 1:
            print(f"{param}: {len(unique_vals)} unique values - {sorted(unique_vals)[:10]}")
        else:
            print(f"{param}: CONSTANT at {unique_vals[0] if len(unique_vals) > 0 else 'N/A'}")

# Check categorical variation
print("\nCategorical Parameters:")
for param in ['tsm_type', 'llm_model', 'llm_methods', 'blend_mode']:
    if param in df.columns:
        unique_vals = df[param].unique()
        print(f"{param}: {len(unique_vals)} unique values")
        for v in unique_vals[:5]:
            count = len(df[df[param] == v])
            print(f"  - {v}: {count} runs")

# If LLM methods vary, analyze performance by method
if 'llm_methods' in df.columns and 'tsm_sharpe_h1' in df.columns:
    df_valid = df[df['tsm_sharpe_h1'].notna()].copy()
    if len(df_valid) > 0:
        print(f"\n\n{'='*80}")
        print("PERFORMANCE BY LLM METHOD")
        print('='*80)
        grouped = df_valid.groupby('llm_methods').agg({
            'tsm_sharpe_h1': ['mean', 'std', 'count'],
            'tsm_hit_rate_h1': ['mean', 'std']
        }).round(4)
        print(grouped.to_string())
