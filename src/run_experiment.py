"""
Main experiment runner for EU ETS forecasting.

This script orchestrates the entire pipeline:
1. Load and standardize data
2. FX convert USD→EUR
3. Build panel + windows
4. Train baselines + TSM
5. Generate TSM predictions
6. Run LLM refinement methods
7. Evaluate + significance tests
8. Robustness suite
9. Write paper

Usage:
    python -m src.run_experiment --config src/config/default.yaml
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import load_config
from data import build_panel
from data.windows import make_windows, split_windows, StandardScaler, save_datasets, WindowConfig
from data.panel import get_coverage_report, plot_coverage_heatmap
from models.baselines import NaivePersistence, SeasonalNaive, LinearBaseline
from eval.metrics import compute_metrics_by_horizon, get_per_sample_errors
from eval.trend_classification import compute_trend_accuracy
from eval.significance import compare_all_models
from paper import PaperWriter
from utils import setup_logging, set_seed

logger = logging.getLogger(__name__)


def run_experiment(config_path: str = None, overrides: dict = None):
    """
    Run the full experiment pipeline.
    
    Args:
        config_path: Path to config YAML file
        overrides: Config overrides
    """
    # =========================================================================
    # Step 1: Load configuration
    # =========================================================================
    logger.info("=" * 70)
    logger.info("EU ETS FUTURES FORECASTING EXPERIMENT")
    logger.info("=" * 70)
    
    config = load_config(config_path, overrides)
    run_dir = config.setup_run_dir()
    
    # Setup logging to run directory
    setup_logging(run_dir)
    
    logger.info(f"Run ID: {config.run_id}")
    logger.info(f"Run directory: {run_dir}")
    
    # Set random seed
    set_seed(config.seed)
    
    # Save config
    config.save()
    config.save_reproducibility_info()
    
    # =========================================================================
    # Step 2: Build data panel
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Building data panel")
    logger.info("=" * 70)
    
    data_dir = Path(__file__).parent.parent / "Data"
    
    panel, schema = build_panel(
        data_dir=data_dir,
        config=config.raw,
        save_path=run_dir / "data"
    )
    
    # Generate coverage report
    coverage = get_coverage_report(panel)
    coverage.to_csv(run_dir / "data" / "coverage_report.csv", index=False)
    
    try:
        plot_coverage_heatmap(panel, run_dir / "paper_snapshot" / "fig_data_coverage.png")
    except Exception as e:
        logger.warning(f"Failed to create coverage plot: {e}")
    
    # =========================================================================
    # Step 3: Create windows and split data
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: Creating windows and splitting data")
    logger.info("=" * 70)
    
    # Get feature columns (exclude date and target)
    feature_cols = [c for c in panel.columns if c not in ['date', 'y']]
    
    window_config = WindowConfig(
        seq_len=config.seq_len,
        label_len=config.label_len,
        pred_len=config.pred_len,
        target_col='y',
        feature_cols=['y'] + feature_cols[:10]  # Limit features for now
    )
    
    X_enc, X_dec, y, dates = make_windows(panel, window_config, mode='MS')
    
    # Split data
    splits = split_windows(
        X_enc, X_dec, y, dates,
        train_end=config.split.get('train_end', '2022-12-31'),
        val_end=config.split.get('val_end', '2023-12-31')
    )
    
    # Fit scaler on training data
    scaler = StandardScaler()
    scaler.fit(splits['train']['X_enc'])
    
    # Save datasets
    save_datasets(splits, scaler, run_dir / "data" / "datasets")
    
    # =========================================================================
    # Step 4: Train and evaluate baselines
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: Training and evaluating baselines")
    logger.info("=" * 70)
    
    # Extract target only for baselines
    y_train_hist = splits['train']['X_enc'][:, :, 0]  # Target is first feature
    y_test_hist = splits['test']['X_enc'][:, :, 0]
    y_test_future = splits['test']['y']
    
    # Get base price (last known price for each sample)
    y_test_base = y_test_hist[:, -1]
    
    # Evaluate baselines
    baseline_results = {}
    horizons = config.horizons
    
    # Naive persistence
    naive = NaivePersistence(config.pred_len)
    naive.fit(y_train_hist.flatten())
    naive_pred = naive.predict(y_test_hist)
    
    baseline_results['naive_persistence'] = {
        'predictions': naive_pred,
        'metrics': compute_metrics_by_horizon(y_test_future, naive_pred, horizons),
        'errors': get_per_sample_errors(y_test_future, naive_pred, horizons)
    }
    
    # Seasonal naive
    seasonal = SeasonalNaive(config.pred_len, season_period=5)
    seasonal.fit(y_train_hist.flatten())
    seasonal_pred = seasonal.predict(y_test_hist)
    
    baseline_results['seasonal_naive'] = {
        'predictions': seasonal_pred,
        'metrics': compute_metrics_by_horizon(y_test_future, seasonal_pred, horizons),
        'errors': get_per_sample_errors(y_test_future, seasonal_pred, horizons)
    }
    
    # Print baseline results
    logger.info("\nBaseline Results (MSE by horizon):")
    for name, result in baseline_results.items():
        logger.info(f"\n{name}:")
        logger.info(result['metrics'])
    
    # Save baseline results
    for name, result in baseline_results.items():
        result['metrics'].to_csv(run_dir / "results" / f"{name}_metrics.csv")
    
    # =========================================================================
    # Step 5: TSM training (if PyTorch available)
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: TSM model training")
    logger.info("=" * 70)
    
    tsm_pred = None
    
    try:
        import torch
        from torch.utils.data import DataLoader
        from models.tsm import TSMForecaster
        from data.windows import TimeSeriesDataset
        
        # Create datasets
        train_dataset = TimeSeriesDataset(
            scaler.transform(splits['train']['X_enc']),
            scaler.transform(splits['train']['X_dec']),
            (splits['train']['y'] - scaler.mean_[0]) / scaler.std_[0]
        )
        
        val_dataset = TimeSeriesDataset(
            scaler.transform(splits['val']['X_enc']),
            scaler.transform(splits['val']['X_dec']),
            (splits['val']['y'] - scaler.mean_[0]) / scaler.std_[0]
        )
        
        test_dataset = TimeSeriesDataset(
            scaler.transform(splits['test']['X_enc']),
            scaler.transform(splits['test']['X_dec']),
            (splits['test']['y'] - scaler.mean_[0]) / scaler.std_[0]
        )
        
        # Update config with actual feature count
        config.raw['model']['enc_in'] = splits['train']['X_enc'].shape[-1]
        config.raw['model']['dec_in'] = splits['train']['X_dec'].shape[-1]
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        # Train model
        tsm = TSMForecaster(config.raw)
        
        history = tsm.fit(
            train_loader,
            val_loader,
            epochs=config.model.get('max_epochs', 100),
            patience=config.model.get('early_stopping_patience', 10),
            save_path=run_dir / "models" / "tsm_checkpoint.pt"
        )
        
        # Generate predictions
        tsm_pred_scaled, y_true_scaled = tsm.predict(test_loader)
        
        # Inverse transform
        tsm_pred = scaler.inverse_transform_target(tsm_pred_scaled)
        
        # Evaluate
        baseline_results['tsm'] = {
            'predictions': tsm_pred,
            'metrics': compute_metrics_by_horizon(y_test_future, tsm_pred, horizons),
            'errors': get_per_sample_errors(y_test_future, tsm_pred, horizons)
        }
        
        logger.info("\nTSM Results:")
        logger.info(baseline_results['tsm']['metrics'])
        
        # Save predictions
        import pandas as pd
        pred_df = pd.DataFrame({
            'date': splits['test']['dates']
        })
        for h in range(config.pred_len):
            pred_df[f'y_true_t_plus_{h+1}'] = y_test_future[:, h]
            pred_df[f'yhat_t_plus_{h+1}'] = tsm_pred[:, h]
        
        pred_df.to_parquet(run_dir / "predictions" / "tsm_pred_test.parquet")
        
    except ImportError:
        logger.warning("PyTorch not available - skipping TSM training")
    except Exception as e:
        logger.error(f"TSM training failed: {e}")
        import traceback
        traceback.print_exc()
    
    # =========================================================================
    # Step 6: LLM refinement (if API key available)
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 6: LLM refinement")
    logger.info("=" * 70)
    
    llm_results = {}
    
    try:
        import os
        if not os.environ.get('OPENAI_API_KEY'):
            logger.warning("OPENAI_API_KEY not set - skipping LLM refinement")
        else:
            from llm import LLMRefiner
            
            refiner = LLMRefiner(
                config.llm,
                cache_dir=run_dir / "llm" / "cache",
                log_dir=run_dir / "llm" / "logs"
            )
            
            # Prepare data
            test_dates = [list(map(str, d)) for d in splits['test']['dates']]
            
            # Run each method (limit samples for cost control)
            max_samples = min(50, len(y_test_hist))
            
            for method in config.llm.get('methods', ['TSM+LLM']):
                logger.info(f"Running method: {method}")
                
                tsm_forecast = tsm_pred[:max_samples] if tsm_pred is not None else None
                
                try:
                    predictions, metadata = refiner.refine_batch(
                        method=method,
                        histories=y_test_hist[:max_samples],
                        date_arrays=test_dates[:max_samples],
                        tsm_forecasts=tsm_forecast,
                        pred_len=config.pred_len
                    )
                    
                    llm_results[method] = {
                        'predictions': predictions,
                        'metrics': compute_metrics_by_horizon(
                            y_test_future[:max_samples], predictions, horizons
                        ),
                        'errors': get_per_sample_errors(
                            y_test_future[:max_samples], predictions, horizons
                        )
                    }
                    
                    logger.info(f"\n{method} Results:")
                    logger.info(llm_results[method]['metrics'])
                    
                except Exception as e:
                    logger.error(f"LLM method {method} failed: {e}")
    
    except Exception as e:
        logger.error(f"LLM refinement failed: {e}")
    
    # =========================================================================
    # Step 7: Statistical significance tests
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 7: Statistical significance tests")
    logger.info("=" * 70)
    
    # Combine all results
    all_errors = {name: result['errors'] for name, result in baseline_results.items()}
    all_errors.update({name: result['errors'] for name, result in llm_results.items()})
    
    if len(all_errors) > 1:
        significance_df = compare_all_models(
            all_errors,
            baseline_name='naive_persistence',
            horizons=horizons
        )
        
        significance_df.to_csv(run_dir / "results" / "significance_tests.csv", index=False)
        logger.info("\nSignificance test results saved")
    
    # =========================================================================
    # Step 8: Trend classification
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 8: Trend classification accuracy")
    logger.info("=" * 70)
    
    trend_results = {}
    
    for name, result in {**baseline_results, **llm_results}.items():
        pred = result['predictions']
        # Match dimensions
        if len(pred) != len(y_test_base):
            continue
            
        trend_acc = compute_trend_accuracy(
            y_test_future[:len(pred)],
            pred,
            y_test_base[:len(pred)],
            threshold=0.5,  # EUR
            horizons=horizons
        )
        trend_results[name] = trend_acc
        
        logger.info(f"\n{name} Trend Accuracy:")
        logger.info(trend_acc[['accuracy']])
    
    # =========================================================================
    # Step 9: Generate paper
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 9: Generating paper")
    logger.info("=" * 70)
    
    writer = PaperWriter(run_dir / "paper_snapshot")
    
    # Collect metrics for paper
    import pandas as pd
    all_metrics = []
    for name, result in {**baseline_results, **llm_results}.items():
        metrics = result['metrics'].copy()
        metrics['model'] = name
        all_metrics.append(metrics.reset_index())
    
    if all_metrics:
        metrics_df = pd.concat(all_metrics, ignore_index=True)
    else:
        metrics_df = None
    
    trend_df = None
    if trend_results:
        trend_dfs = []
        for name, df in trend_results.items():
            df = df.copy()
            df['model'] = name
            trend_dfs.append(df.reset_index())
        trend_df = pd.concat(trend_dfs, ignore_index=True)
    
    writer.write_all_sections(
        panel_schema=schema,
        config=config.raw,
        metrics_by_horizon=metrics_df,
        trend_accuracy=trend_df,
        significance_tests=significance_df if 'significance_df' in dir() else None
    )
    
    paper_path = writer.save()
    logger.info(f"Paper saved to: {paper_path}")
    
    # =========================================================================
    # Complete
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Results saved to: {run_dir}")
    
    return run_dir


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="EU ETS Futures Forecasting Experiment"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed override"
    )
    
    args = parser.parse_args()
    
    overrides = {}
    if args.seed:
        overrides["reproducibility"] = {"seed": args.seed}
    
    run_experiment(args.config, overrides)


if __name__ == "__main__":
    main()
