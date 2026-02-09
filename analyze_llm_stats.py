#!/usr/bin/env python3
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import json

runs_dir = Path("runs")
results = []

for run_folder in sorted(runs_dir.iterdir()):
    if not run_folder.is_dir() or run_folder.name.startswith('.'):
        continue
    
    config_file = run_folder / "config_resolved.yaml"
    llm_logs = run_folder / "llm" / "logs" / "llm_calls.jsonl"
    
    if not config_file.exists():
        continue
    
    with open(config_file) as f:
        config = yaml.safe_load(f)
    
    run_data = {
        'run_id': run_folder.name,
        'llm_model': config.get('llm', {}).get('model'),
        'llm_methods': str(config.get('llm', {}).get('methods', [])),
        'tsm_type': config.get('model', {}).get('tsm_type'),
        'd_model': config.get('model', {}).get('d_model'),
    }
    
    # Parse LLM logs to get call statistics
    if llm_logs.exists():
        success_count = 0
        failure_count = 0
        total_calls = 0
        methods_used = set()
        
        with open(llm_logs) as f:
            for line in f:
                try:
                    call = json.loads(line.strip())
                    total_calls += 1
                    if call.get('metadata', {}).get('success', False):
                        success_count += 1
                    else:
                        failure_count += 1
                    methods_used.add(call.get('method'))
                except:
                    pass
        
        run_data['llm_total_calls'] = total_calls
        run_data['llm_success_rate'] = success_count / total_calls if total_calls > 0 else None
        run_data['llm_methods_used'] = str(methods_used)
    
    results.append(run_data)

df = pd.DataFrame(results)

# Filter for runs with LLM logs
df_llm = df[df['llm_total_calls'].notna()].copy()
print(f"Runs with LLM logs: {len(df_llm)}")

if len(df_llm) > 0:
    print("\n=== LLM CALL STATISTICS ===")
    print(f"Total LLM calls across all runs: {df_llm['llm_total_calls'].sum():.0f}")
    print(f"Avg calls per run: {df_llm['llm_total_calls'].mean():.1f}")
    print(f"Min calls: {df_llm['llm_total_calls'].min():.0f}, Max: {df_llm['llm_total_calls'].max():.0f}")
    
    print("\n=== BY LLM MODEL ===")
    for model in df_llm['llm_model'].unique():
        subset = df_llm[df_llm['llm_model'] == model]
        calls = subset['llm_total_calls'].sum()
        success = subset['llm_success_rate'].mean()
        print(f"{model}: {len(subset)} runs, {calls:.0f} total calls, {success:.1%} avg success rate")
    
    print("\n=== BY LLM METHOD ===")
    for method in df_llm['llm_methods'].unique()[:10]:
        subset = df_llm[df_llm['llm_methods'] == method]
        calls = subset['llm_total_calls'].sum()
        success = subset['llm_success_rate'].mean()
        print(f"{method[:60]}: {len(subset)} runs, {calls:.0f} calls, {success:.1%} success")
