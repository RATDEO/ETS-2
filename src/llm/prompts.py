"""
Prompt templates for LLM-based forecast refinement.

Implements multiple prompting strategies:
- DP: Direct prompting
- CoT: Chain-of-thought
- CoT-RF: Chain-of-thought with refinement
- TSM+LLM: TSM forecast refinement
"""

from typing import Dict, List, Optional, Union
import numpy as np
import json


class PromptTemplate:
    """Base class for prompt templates."""
    
    def __init__(self, name: str):
        self.name = name
    
    def format(self, **kwargs) -> str:
        """Format the prompt with given parameters."""
        raise NotImplementedError
    
    def get_system_message(self) -> str:
        """Get the system message for the LLM."""
        return """You are an expert financial analyst specializing in carbon markets and EU ETS (Emissions Trading System) price forecasting. 
Your task is to analyze carbon credit price data and provide accurate price forecasts.
Always respond with valid JSON containing your predictions."""


class DirectPromptTemplate(PromptTemplate):
    """Direct prompting: LLM forecasts from history alone."""
    
    def __init__(self):
        super().__init__("DP")
    
    def format(
        self,
        history: np.ndarray,
        dates: List[str],
        pred_len: int = 30,
        currency: str = "EUR",
        **kwargs
    ) -> str:
        """
        Format direct prompt.
        
        Args:
            history: Historical price values
            dates: Corresponding dates
            pred_len: Number of days to forecast
            currency: Price currency
        """
        # Compress history to recent values
        if len(history) > 60:
            history = history[-60:]
            dates = dates[-60:]
        
        history_str = ", ".join([f"{v:.2f}" for v in history])
        
        prompt = f"""Analyze the following EU ETS carbon credit price history and forecast the next {pred_len} trading days.

## Historical Data
Currency: {currency}
Period: {dates[0]} to {dates[-1]}
Prices (last {len(history)} days): [{history_str}]

## Summary Statistics
- Current price: {history[-1]:.2f} {currency}
- 20-day average: {np.mean(history[-20:]):.2f} {currency}
- 20-day volatility (std): {np.std(history[-20:]):.2f} {currency}
- 5-day change: {(history[-1] - history[-5]):.2f} {currency} ({(history[-1]/history[-5] - 1)*100:.1f}%)
- 20-day change: {(history[-1] - history[-20]):.2f} {currency} ({(history[-1]/history[-20] - 1)*100:.1f}%)

## Task
Forecast the EU ETS price for the next {pred_len} trading days.

Respond with a JSON object containing exactly {pred_len} price predictions:
{{"yhat": [day1_price, day2_price, ..., day{pred_len}_price]}}

Your forecast (JSON only):"""
        
        return prompt


class ChainOfThoughtTemplate(PromptTemplate):
    """Chain-of-thought prompting with reasoning."""
    
    def __init__(self):
        super().__init__("CoT")
    
    def format(
        self,
        history: np.ndarray,
        dates: List[str],
        pred_len: int = 30,
        currency: str = "EUR",
        exogenous_summary: Optional[Dict] = None,
        **kwargs
    ) -> str:
        """
        Format chain-of-thought prompt.
        """
        if len(history) > 60:
            history = history[-60:]
            dates = dates[-60:]
        
        history_str = ", ".join([f"{v:.2f}" for v in history])
        
        # Trend analysis
        trend_5d = "upward" if history[-1] > history[-5] else "downward"
        trend_20d = "upward" if history[-1] > history[-20] else "downward"
        
        exo_section = ""
        if exogenous_summary:
            exo_section = "\n## Related Market Data\n"
            for key, value in exogenous_summary.items():
                exo_section += f"- {key}: {value}\n"
        
        prompt = f"""Analyze the following EU ETS carbon credit price history and forecast the next {pred_len} trading days.

## Historical Data
Currency: {currency}
Period: {dates[0]} to {dates[-1]}
Prices (last {len(history)} days): [{history_str}]

## Summary Statistics
- Current price: {history[-1]:.2f} {currency}
- 20-day average: {np.mean(history[-20:]):.2f} {currency}
- 20-day volatility (std): {np.std(history[-20:]):.2f} {currency}
- 5-day trend: {trend_5d} ({(history[-1]/history[-5] - 1)*100:.1f}%)
- 20-day trend: {trend_20d} ({(history[-1]/history[-20] - 1)*100:.1f}%)
- 20-day high: {np.max(history[-20:]):.2f} {currency}
- 20-day low: {np.min(history[-20:]):.2f} {currency}
{exo_section}
## Task
Think step by step:
1. Analyze the recent price trend and momentum
2. Consider the volatility regime
3. Identify any potential support/resistance levels
4. Factor in typical carbon market seasonality
5. Generate your {pred_len}-day forecast

Provide your reasoning followed by a JSON forecast.

Your analysis:
[Think through the factors affecting EU ETS prices]

Based on your analysis, provide your forecast as JSON:
{{"yhat": [day1_price, day2_price, ..., day{pred_len}_price]}}

Response:"""
        
        return prompt


class RefinementTemplate(PromptTemplate):
    """TSM+LLM: Refine a model's forecast."""
    
    def __init__(self):
        super().__init__("TSM+LLM")
    
    def format(
        self,
        history: np.ndarray,
        dates: List[str],
        tsm_forecast: np.ndarray,
        pred_len: int = 30,
        currency: str = "EUR",
        exogenous_summary: Optional[Dict] = None,
        **kwargs
    ) -> str:
        """
        Format refinement prompt.
        
        Args:
            history: Historical price values
            dates: Corresponding dates
            tsm_forecast: Time series model forecast to refine
            pred_len: Number of days
            currency: Price currency
            exogenous_summary: Summary of exogenous variables
        """
        if len(history) > 60:
            history = history[-60:]
            dates = dates[-60:]
        
        history_str = ", ".join([f"{v:.2f}" for v in history[-30:]])
        forecast_str = ", ".join([f"{v:.2f}" for v in tsm_forecast])
        
        exo_section = ""
        if exogenous_summary:
            exo_section = "\n## Related Market Data\n"
            for key, value in exogenous_summary.items():
                exo_section += f"- {key}: {value}\n"
        
        prompt = f"""You are refining a quantitative time series model's EU ETS carbon price forecast.

## Historical Data (last 30 days)
Currency: {currency}
End date: {dates[-1]}
Prices: [{history_str}]

## Current Statistics
- Current price: {history[-1]:.2f} {currency}
- 20-day average: {np.mean(history[-20:]):.2f} {currency}
- 20-day volatility: {np.std(history[-20:]):.2f} {currency}
- Recent trend: {(history[-1]/history[-5] - 1)*100:.1f}% (5-day)
{exo_section}
## Model Forecast (to be refined)
The quantitative model predicts the following {pred_len}-day path:
[{forecast_str}]

Model forecast summary:
- Start: {tsm_forecast[0]:.2f} {currency}
- End: {tsm_forecast[-1]:.2f} {currency}
- Predicted change: {(tsm_forecast[-1]/tsm_forecast[0] - 1)*100:.1f}%
- Predicted volatility: {np.std(tsm_forecast):.2f} {currency}

## Task
Review the model's forecast and provide a refined version. Consider:
1. Is the predicted trend direction reasonable given recent momentum?
2. Is the magnitude of change realistic given historical volatility?
3. Should any short-term corrections be expected?
4. Are there any mean-reversion or breakout patterns to consider?

Provide your refined {pred_len}-day forecast as JSON:
{{"yhat": [day1_price, day2_price, ..., day{pred_len}_price]}}

Your refined forecast (JSON only):"""
        
        return prompt


class CoTRefinementTemplate(PromptTemplate):
    """Chain-of-thought with self-refinement."""
    
    def __init__(self):
        super().__init__("CoT-RF")
    
    def format(
        self,
        history: np.ndarray,
        dates: List[str],
        initial_forecast: np.ndarray,
        pred_len: int = 30,
        currency: str = "EUR",
        **kwargs
    ) -> str:
        """
        Format self-refinement prompt.
        """
        history_str = ", ".join([f"{v:.2f}" for v in history[-30:]])
        forecast_str = ", ".join([f"{v:.2f}" for v in initial_forecast])
        
        prompt = f"""Review and refine this EU ETS price forecast.

## Context
Current price: {history[-1]:.2f} {currency}
Historical volatility: {np.std(history[-20:]):.2f} {currency}
Recent prices: [{history_str}]

## Initial Forecast
[{forecast_str}]

## Refinement Task
Critically evaluate the forecast:
1. Check for unrealistic jumps or trends
2. Verify consistency with historical volatility patterns
3. Consider mean-reversion tendencies
4. Adjust for any overconfidence in trend predictions

Provide your refined {pred_len}-day forecast as JSON:
{{"yhat": [day1_price, day2_price, ..., day{pred_len}_price]}}

Refined forecast (JSON only):"""
        
        return prompt


def get_template(method: str) -> PromptTemplate:
    """Get prompt template by method name."""
    templates = {
        "DP": DirectPromptTemplate(),
        "CoT": ChainOfThoughtTemplate(),
        "TSM+LLM": RefinementTemplate(),
        "CoT-RF": CoTRefinementTemplate(),
    }
    
    if method not in templates:
        raise ValueError(f"Unknown method: {method}. Available: {list(templates.keys())}")
    
    return templates[method]
