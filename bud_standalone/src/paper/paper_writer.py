"""
Paper writer for generating the scientific manuscript.

Automatically generates sections as pipeline stages complete,
including data descriptions, methods, results, and figures.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


class PaperWriter:
    """
    Incremental scientific paper generator.
    
    Writes markdown sections that can be compiled to PDF.
    """
    
    def __init__(
        self,
        output_dir: Path,
        title: str = "EU ETS Futures Forecasting with TSM + LLM Refinement",
        authors: List[str] = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)
        
        self.tables_dir = self.output_dir / "tables"
        self.tables_dir.mkdir(exist_ok=True)
        
        self.title = title
        self.authors = authors or ["Auto-generated"]
        
        self.sections = {}
        self.figure_count = 0
        self.table_count = 0
    
    def _write_section(self, name: str, content: str):
        """Write a section to the sections dict."""
        self.sections[name] = content
        logger.info(f"Wrote section: {name}")

    def _format_float(self, value: Optional[float], decimals: int = 3) -> str:
        """Format a float for narrative text."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "N/A"
        return f"{value:.{decimals}f}"

    def _summarize_results(
        self,
        metrics_by_horizon: Optional[pd.DataFrame],
        trend_accuracy: Optional[pd.DataFrame],
        significance_tests: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """Summarize key result signals for narrative sections."""
        summary: Dict[str, Any] = {
            "has_metrics": False,
            "has_trend": False,
            "has_significance": False,
            "has_llm": False,
            "llm_models": []
        }

        if metrics_by_horizon is not None and not metrics_by_horizon.empty:
            summary["has_metrics"] = True
            metrics = metrics_by_horizon.copy()
            if "model" in metrics.columns:
                metrics["model"] = metrics["model"].astype(str)
                model_names = sorted(metrics["model"].dropna().unique().tolist())
                llm_markers = ("llm", "cot", "dp", "prompt", "gpt")
                llm_models = [
                    name for name in model_names
                    if any(marker in name.lower() for marker in llm_markers)
                ]
                summary["has_llm"] = bool(llm_models)
                summary["llm_models"] = llm_models

            if "mse" in metrics.columns and "model" in metrics.columns:
                model_mse = metrics.groupby("model")["mse"].mean().sort_values()
                summary["model_mse"] = model_mse.to_dict()
                summary["best_mse_model"] = model_mse.index[0]
                summary["best_mse_value"] = float(model_mse.iloc[0])

                if "naive_persistence" in model_mse.index:
                    summary["naive_mse"] = float(model_mse["naive_persistence"])
                if "tsm" in model_mse.index:
                    summary["tsm_mse"] = float(model_mse["tsm"])

        if trend_accuracy is not None and not trend_accuracy.empty:
            summary["has_trend"] = True
            trend = trend_accuracy.copy()
            if "accuracy" in trend.columns and "model" in trend.columns:
                model_acc = trend.groupby("model")["accuracy"].mean().sort_values(ascending=False)
                summary["best_trend_model"] = model_acc.index[0]
                summary["best_trend_value"] = float(model_acc.iloc[0])
                best_row = trend.loc[trend["accuracy"].idxmax()]
                summary["best_trend_horizon"] = best_row.get("horizon")

        if significance_tests is not None and not significance_tests.empty:
            summary["has_significance"] = True
            if "model_better" in significance_tests.columns:
                summary["any_model_better"] = bool(significance_tests["model_better"].any())

        if summary.get("naive_mse") and summary.get("tsm_mse"):
            summary["tsm_vs_naive_ratio"] = summary["tsm_mse"] / summary["naive_mse"]

        return summary
    
    def write_abstract(self, summary_stats: Dict = None):
        """Write the abstract."""
        content = """## Abstract

This study reproduces and extends the methodology of recent research on using Large Language Models (LLMs) 
for carbon price forecasting, applying it to the European Union Emissions Trading System (EU ETS). 
We implement a hybrid approach combining time series models (TSM) with LLM-based forecast refinement, 
evaluating performance across multiple horizons (1, 5, 20, and 30 days ahead).

Our methodology follows a two-stage approach: first training a deep learning time series model to produce 
30-step forecast paths, then applying LLM refinement to improve predictions. We evaluate multiple prompting 
strategies including direct prompting (DP), chain-of-thought (CoT), and TSM+LLM refinement.

Results are evaluated using mean squared error (MSE) for price prediction accuracy and classification accuracy 
for trend direction prediction. Statistical significance is assessed using paired t-tests and Wilcoxon signed-rank tests.
"""
        
        if summary_stats:
            content += f"""
Key findings:
- Dataset spans {summary_stats.get('date_range', 'N/A')}
- {summary_stats.get('n_observations', 'N/A')} daily observations
- Best performing method: {summary_stats.get('best_method', 'TBD')}
"""
        
        self._write_section("abstract", content)
    
    def write_introduction(self):
        """Write the introduction section."""
        content = """## 1. Introduction

The European Union Emissions Trading System (EU ETS) is the world's largest carbon market, 
covering approximately 40% of the EU's greenhouse gas emissions. Accurate forecasting of 
EU ETS allowance prices is crucial for compliance planning, investment decisions, and 
policy analysis.

### 1.1 Background

Carbon markets have grown significantly since the establishment of the EU ETS in 2005. 
The price of EU Allowances (EUAs) is influenced by multiple factors including:
- Energy prices (particularly natural gas and coal)
- Economic activity and industrial output
- Regulatory decisions and policy announcements
- Market speculation and hedging activities
- Auction volumes and timing

### 1.2 Related Work

Recent advances in Large Language Models have shown promising results in financial forecasting. 
The work "Can Large Language Models forecast carbon price movements?" demonstrated that LLMs 
can effectively refine quantitative model forecasts for Chinese carbon markets.

### 1.3 Contributions

This study makes the following contributions:
1. Reproduction of the Meta-DTS methodology for EU ETS futures
2. Implementation of multiple prompting strategies for forecast refinement
3. Comprehensive evaluation at multiple forecast horizons
4. Robustness analysis including noise injection tests
"""
        self._write_section("introduction", content)
    
    def write_data_section(
        self,
        panel_schema: Dict,
        coverage_report: pd.DataFrame = None
    ):
        """Write the data description section."""
        content = f"""## 2. Data

### 2.1 Data Sources

This study uses the following data sources for EU ETS forecasting:

1. **Primary Market (Auctions)**: EEX emissions auction data from 2017-2025
2. **Secondary Market (ICAP)**: Daily EUA prices from the ICAP Allowance Price Explorer
3. **Carbon Market Indices**: ETF indices tracking carbon markets (KEUA, KRBN, GRN, KCCA, KSET)
4. **Energy Benchmarks**: Brent crude oil spot prices and Rotterdam coal futures
5. **Volatility Index**: VSTOXX for market volatility proxy
6. **Exchange Rates**: ECB EUR/USD reference rates for currency conversion

### 2.2 Target Variable

The primary target variable is the EU ETS secondary market price in EUR, derived from 
the ICAP allowance price explorer data. This represents actual trading prices rather 
than auction clearing prices.

### 2.3 Data Coverage

- **Date Range**: {panel_schema.get('date_range', {}).get('start', 'N/A')} to {panel_schema.get('date_range', {}).get('end', 'N/A')}
- **Total Observations**: {panel_schema.get('n_rows', 'N/A'):,}
- **Number of Features**: {len(panel_schema.get('columns', {})) - 1}

### 2.4 Preprocessing

Key preprocessing steps:
1. Currency conversion: All USD-denominated series converted to EUR using ECB rates
2. Calendar alignment: All features aligned to the EUA trading calendar
3. Missing value handling: Forward-fill up to 5 days for market holidays
4. Feature engineering: Returns, rolling statistics, and momentum indicators
"""
        
        if coverage_report is not None:
            content += "\n### 2.5 Feature Coverage\n\n"
            content += "See Figure 1 for the coverage heatmap across features and time.\n"
        
        self._write_section("data", content)
    
    def write_methods_section(self, config: Dict = None):
        """Write the methods section."""
        config = config or {}
        ts_config = config.get("time_series", {})
        model_config = config.get("model", {})
        
        content = f"""## 3. Methods

### 3.1 Problem Formulation

Given a sequence of daily EUA prices $y_{{t-L+1}}, ..., y_t$ and exogenous features 
$X_{{t-L+1}}, ..., X_t$, we aim to forecast the next $H$ days:

$$\\hat{{y}}_{{t+1}}, ..., \\hat{{y}}_{{t+H}}$$

where $L = {ts_config.get('seq_len', 120)}$ (lookback window) and $H = {ts_config.get('pred_len', 30)}$ (forecast horizon).

### 3.2 Baseline Models

We implement several baseline models for comparison:

1. **Naive Persistence**: $\\hat{{y}}_{{t+h}} = y_t$ for all horizons
2. **Seasonal Naive**: $\\hat{{y}}_{{t+h}} = y_{{t+h-5}}$ (weekly seasonality)
3. **Linear Regression**: Ridge regression on lagged features
4. **ARIMA**: Autoregressive integrated moving average model

### 3.3 Time Series Model (TSM)

The primary TSM uses an Autoformer-style architecture with:
- **Model dimension**: {model_config.get('d_model', 512)}
- **Attention heads**: {model_config.get('n_heads', 8)}
- **Encoder layers**: {model_config.get('e_layers', 2)}
- **Feed-forward dimension**: {model_config.get('d_ff', 2048)}
- **Dropout**: {model_config.get('dropout', 0.05)}

The model uses series decomposition to separate trend and seasonal components, 
with auto-correlation attention for efficient long-range dependency modeling.

### 3.4 LLM Refinement

We implement four prompting strategies:

1. **Direct Prompting (DP)**: LLM forecasts directly from historical data
2. **Chain-of-Thought (CoT)**: Includes step-by-step reasoning
3. **CoT with Refinement (CoT-RF)**: CoT followed by self-critique and refinement
4. **TSM+LLM**: LLM refines the TSM model's forecast path

The TSM+LLM method is the primary approach, where the LLM receives:
- Recent price history (30 days)
- Summary statistics (mean, volatility, trend)
- The TSM's 30-day forecast
- Market context (optional)

### 3.5 Evaluation Metrics

**Price Prediction (Regression)**:
- Mean Squared Error (MSE): $\\text{{MSE}} = \\frac{{1}}{{n}} \\sum_{{i=1}}^n (y_i - \\hat{{y}}_i)^2$
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)

**Trend Classification**:
- Three-way classification: Up, Flat, Down
- Threshold based on recent volatility: $\\tau = 0.25 \\times \\sigma_{{20d}}$
- Classification accuracy at each horizon

### 3.6 Statistical Tests

- **Paired t-test**: For comparing mean squared errors between methods
- **Wilcoxon signed-rank test**: Non-parametric alternative
- Significance level: $\\alpha = 0.05$
"""
        self._write_section("methods", content)
    
    def write_results_section(
        self,
        metrics_by_horizon: pd.DataFrame = None,
        trend_accuracy: pd.DataFrame = None,
        significance_tests: pd.DataFrame = None
    ):
        """Write the results section."""
        content = """## 4. Results

### 4.1 MSE by Horizon

Table 1 presents the Mean Squared Error at forecast horizons of 1, 5, 20, and 30 days.
"""
        
        if metrics_by_horizon is not None:
            self.table_count += 1
            table_path = self.tables_dir / "mse_by_horizon.md"
            metrics_by_horizon.to_markdown(table_path)
            content += f"\n{metrics_by_horizon.to_markdown()}\n"
        
        content += """
### 4.2 Trend Classification Accuracy

Table 2 shows the accuracy of predicting price direction (up/flat/down).
"""
        
        if trend_accuracy is not None:
            self.table_count += 1
            table_path = self.tables_dir / "trend_accuracy.md"
            trend_accuracy.to_markdown(table_path)
            content += f"\n{trend_accuracy.to_markdown()}\n"
        
        content += """
### 4.3 Statistical Significance

Table 3 reports the results of significance tests comparing methods.
"""
        
        if significance_tests is not None:
            self.table_count += 1
            table_path = self.tables_dir / "significance_tests.md"
            significance_tests.to_markdown(table_path)
            content += f"\n{significance_tests.to_markdown()}\n"
        
        self._write_section("results", content)
    
    def write_robustness_section(
        self,
        noise_results: pd.DataFrame = None,
        ablation_results: pd.DataFrame = None,
        subperiod_results: pd.DataFrame = None
    ):
        """Write the robustness analysis section."""
        content = """## 5. Robustness Analysis

### 5.1 Noise Injection Test

To assess robustness, we inject noise into forecast paths at levels of 5%, 10%, 20%, 
and 30% of the forecast standard deviation.
"""
        
        if noise_results is not None:
            self.table_count += 1
            table_path = self.tables_dir / "noise_injection.md"
            noise_results.to_markdown(table_path, index=False)
            content += f"\n{noise_results.to_markdown()}\n"
        
        content += """
### 5.2 Ablation Study

We compare the contribution of different components:
- TSM only (no LLM refinement)
- LLM only (direct prompting)
- TSM + LLM (main method)
- With/without exogenous features
"""
        
        if ablation_results is not None:
            self.table_count += 1
            table_path = self.tables_dir / "ablation_results.md"
            ablation_results.to_markdown(table_path, index=False)
            content += f"\n{ablation_results.to_markdown()}\n"
        
        content += """
### 5.3 Temporal Stability

We evaluate performance across different market regimes:
"""
        
        if subperiod_results is not None:
            self.table_count += 1
            table_path = self.tables_dir / "subperiod_results.md"
            subperiod_results.to_markdown(table_path, index=False)
            content += f"\n{subperiod_results.to_markdown()}\n"
        elif noise_results is None and ablation_results is None:
            content += "\nNo robustness suite results were generated for this run.\n"
        
        self._write_section("robustness", content)
    
    def write_discussion(self, summary: Optional[Dict[str, Any]] = None):
        """Write the discussion section."""
        summary = summary or {}
        key_findings = []
        if summary.get("best_mse_model"):
            key_findings.append(
                f"- **Price accuracy**: {summary['best_mse_model']} has the lowest average MSE "
                f"({self._format_float(summary.get('best_mse_value'))})."
            )
        if summary.get("tsm_vs_naive_ratio") is not None:
            ratio = summary["tsm_vs_naive_ratio"]
            key_findings.append(
                f"- **TSM vs naive**: TSM MSE is {ratio:.1f}x the naive baseline on average."
            )
        if summary.get("best_trend_model"):
            key_findings.append(
                f"- **Directional accuracy**: {summary['best_trend_model']} has the highest average trend accuracy "
                f"({self._format_float(summary.get('best_trend_value'))})."
            )
        if summary.get("has_metrics") and summary.get("has_llm") is False:
            key_findings.append(
                "- **LLM refinements**: No LLM results are available for this run, so LLM comparisons remain pending."
            )
        if summary.get("has_significance") and summary.get("any_model_better") is False:
            key_findings.append(
                "- **Significance**: No model shows a statistically significant MSE improvement over the naive baseline."
            )
        if not key_findings:
            key_findings.append("- **Key findings**: Results summary not available for this run.")

        content = """## 6. Discussion

### 6.1 Key Findings

""" + "\n".join(key_findings) + """

### 6.2 Comparison to Original Paper

This study applies the Meta-DTS methodology to the EU ETS market. Key similarities 
and differences with the original Chinese carbon market study:

- **Similarities**: Same evaluation metrics (MSE, trend accuracy), similar horizons
- **Differences**: Different market dynamics, regulatory environment, and data sources

### 6.3 Limitations

1. **Data limitations**: Limited secondary market data availability
2. **LLM costs**: API costs limit extensive hyperparameter tuning
3. **Market regime changes**: EU ETS underwent significant regulatory changes
4. **Comparison scope**: Direct comparison to the original paper is limited by market differences

### 6.4 Future Work

- Incorporate news sentiment analysis
- Extend to other carbon markets (California, UK ETS)
- Investigate ensemble approaches
- Explore fine-tuned LLMs for carbon market analysis
"""
        self._write_section("discussion", content)
    
    def write_conclusion(self, summary: Optional[Dict[str, Any]] = None):
        """Write the conclusion section."""
        summary = summary or {}
        conclusion_lines = [
            "## 7. Conclusion",
            "",
            "This run summarizes a reproducible pipeline for EU ETS carbon price forecasting."
        ]

        if summary.get("best_mse_model") and summary.get("best_trend_model"):
            conclusion_lines.append(
                f"Price accuracy is best for {summary['best_mse_model']}, while directional accuracy is highest for "
                f"{summary['best_trend_model']}."
            )
        elif summary.get("best_mse_model"):
            conclusion_lines.append(
                f"Price accuracy is best for {summary['best_mse_model']}."
            )

        if summary.get("has_metrics") and summary.get("has_llm") is False:
            conclusion_lines.append(
                "LLM refinement results are not included in this run, so the hybrid TSM+LLM comparison remains pending."
            )

        conclusion_lines.append(
            "The framework remains suitable for future runs with full LLM and robustness evaluations enabled."
        )
        content = "\n".join(conclusion_lines) + "\n"
        self._write_section("conclusion", content)
    
    def write_reproducibility_appendix(self, config: Dict = None):
        """Write the reproducibility appendix."""
        config = config or {}
        compute = config.get("compute", {})
        
        content = f"""## Appendix A: Reproducibility

### A.1 Configuration

```yaml
# Key configuration parameters
Random Seed: {config.get('reproducibility', {}).get('seed', 42)}
Prediction Length: {config.get('time_series', {}).get('pred_len', 30)}
Sequence Length: {config.get('time_series', {}).get('seq_len', 120)}
TSM Type: {config.get('model', {}).get('tsm_type', 'autoformer')}
LLM Model: {config.get('llm', {}).get('model', 'gpt-5.2')}
```

### A.2 Environment

- Python: 3.10+
- Key packages: torch, pytorch-forecasting, openai, pandas, numpy

### A.3 Compute Resources

- Device preference: {compute.get("device", "auto")}
- Data loader workers: {compute.get("num_workers", "N/A")}
- Pin memory: {compute.get("pin_memory", "N/A")}

### A.4 Data Availability

Raw data sources:
- ICAP Allowance Price Explorer
- EEX Emissions Auctions
- ECB Exchange Rates
- EIA Energy Prices
"""
        self._write_section("appendix", content)
    
    def compile(self) -> str:
        """Compile all sections into a single markdown document."""
        section_order = [
            "abstract",
            "introduction",
            "data",
            "methods",
            "results",
            "robustness",
            "discussion",
            "conclusion",
            "appendix"
        ]
        
        document = f"""# {self.title}

**Authors**: {', '.join(self.authors)}

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

"""
        
        for section in section_order:
            if section in self.sections:
                document += self.sections[section] + "\n\n"
        
        return document
    
    def save(self, filename: str = "manuscript.md") -> Path:
        """Save the compiled document."""
        document = self.compile()
        output_path = self.output_dir / filename
        
        with open(output_path, "w") as f:
            f.write(document)
        
        logger.info(f"Saved paper to {output_path}")
        return output_path
    
    def write_all_sections(
        self,
        panel_schema: Dict = None,
        config: Dict = None,
        metrics_by_horizon: pd.DataFrame = None,
        trend_accuracy: pd.DataFrame = None,
        significance_tests: pd.DataFrame = None,
        noise_results: pd.DataFrame = None,
        ablation_results: pd.DataFrame = None,
        subperiod_results: pd.DataFrame = None
    ):
        """Write all sections at once."""
        summary = self._summarize_results(metrics_by_horizon, trend_accuracy, significance_tests)
        summary_stats = None
        if panel_schema:
            date_range = panel_schema.get("date_range", {})
            date_range_str = f"{date_range.get('start', 'N/A')} to {date_range.get('end', 'N/A')}"
            summary_stats = {
                "date_range": date_range_str,
                "n_observations": panel_schema.get("n_rows", "N/A"),
                "best_method": summary.get("best_mse_model", "TBD")
            }

        self.write_abstract(summary_stats=summary_stats)
        self.write_introduction()
        self.write_data_section(panel_schema or {})
        self.write_methods_section(config)
        self.write_results_section(metrics_by_horizon, trend_accuracy, significance_tests)
        self.write_robustness_section(
            noise_results=noise_results,
            ablation_results=ablation_results,
            subperiod_results=subperiod_results
        )
        self.write_discussion(summary)
        self.write_conclusion(summary)
        self.write_reproducibility_appendix(config)
