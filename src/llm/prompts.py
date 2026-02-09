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


class SentimentRefinementTemplate(PromptTemplate):
    """TSM+LLM-COT-SENT: Refine a model forecast using news sentiment."""

    def __init__(self):
        super().__init__("TSM+LLM-COT-SENT")

    def format(
        self,
        history: np.ndarray,
        dates: List[str],
        tsm_forecast: np.ndarray,
        sentiment_history: np.ndarray,
        pred_len: int = 30,
        currency: str = "EUR",
        **kwargs
    ) -> str:
        if len(history) > 60:
            history = history[-60:]
            dates = dates[-60:]

        history_str = ", ".join([f"{v:.2f}" for v in history[-30:]])
        forecast_str = ", ".join([f"{v:.2f}" for v in tsm_forecast])

        sentiment_history = sentiment_history[-30:] if len(sentiment_history) > 30 else sentiment_history
        sentiment_str = ", ".join([f"{v:.2f}" for v in sentiment_history])

        prompt = f"""You are refining a quantitative EU ETS price forecast using recent news sentiment.

## Historical Data (last 30 days)
Currency: {currency}
End date: {dates[-1]}
Prices: [{history_str}]

## News Sentiment (aligned to history)
Values: [{sentiment_str}]
Interpretation: +1 positive, 0 neutral, -1 negative for short-term price direction.

## Model Forecast (to be refined)
The quantitative model predicts the following {pred_len}-day path:
[{forecast_str}]

## Task
Adjust the forecast using sentiment signals while respecting recent volatility.
Avoid extreme revisions; sentiment should nudge, not override, the model.

Provide your refined {pred_len}-day forecast as JSON.
Return ONLY JSON with exactly {pred_len} numeric values and no code fences:
{{"yhat": [day1_price, day2_price, ..., day{pred_len}_price]}}

JSON only:"""

        return prompt


class NormDeltaRefinementTemplate(PromptTemplate):
    """TSM+LLM-NORM-DELTA: Provide bounded deltas in normalized space."""

    def __init__(self):
        super().__init__("TSM+LLM-NORM-DELTA")

    def format(
        self,
        history: np.ndarray,
        dates: List[str],
        tsm_forecast: np.ndarray,
        pred_len: int = 30,
        currency: str = "EUR",
        history_mean: Optional[float] = None,
        history_std: Optional[float] = None,
        max_delta: float = 0.5,
        **kwargs
    ) -> str:
        """
        Format normalized delta refinement prompt.
        """
        if len(history) > 60:
            history = history[-60:]
            dates = dates[-60:]

        history_str = ", ".join([f"{v:.3f}" for v in history[-30:]])
        forecast_str = ", ".join([f"{v:.3f}" for v in tsm_forecast])

        mean_str = f"{history_mean:.3f}" if history_mean is not None else "N/A"
        std_str = f"{history_std:.3f}" if history_std is not None else "N/A"

        prompt = f"""You are refining a model forecast in normalized (z-score) space.

## Normalization
Prices were normalized using history mean={mean_str} and std={std_str}.
All values below are in normalized units (z-scores).

## Normalized History (last 30 days)
End date: {dates[-1]}
History_z: [{history_str}]

## Normalized Model Forecast (to be adjusted)
Forecast_z ({pred_len}d): [{forecast_str}]

## Task
Provide a small additive adjustment delta_z for each day.
- Each delta_z is in normalized units and will be added to the forecast_z.
- Keep |delta_z| <= {max_delta:.2f} for every day.
- Favor small early-horizon adjustments unless clearly justified.

Return ONLY JSON with exactly {pred_len} values:
{{"delta_z": [d1, d2, ..., d{pred_len}]}}

JSON only:"""

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


class CoTRFReflectionTemplate(PromptTemplate):
    """Chain-of-thought reflection to derive correction rules."""

    def __init__(self):
        super().__init__("CoT-RF-REFLECT")

    def get_system_message(self) -> str:
        return (
            "You are an expert financial analyst specializing in carbon markets and EU ETS price forecasting.\n"
            "Your job is to analyze teaching examples of past forecasts vs true outcomes and derive concise\n"
            "correction rules/heuristics.\n"
            "Respond with plain text rules only (no JSON, no code fences)."
        )

    def format(
        self,
        examples: List[Dict],
        pred_len: int = 30,
        currency: str = "EUR",
        history_points: int = 30,
        sentiment_points: int = 30,
        **kwargs
    ) -> str:
        if not examples:
            return "No teaching examples available."

        blocks = []
        for idx, ex in enumerate(examples, start=1):
            history = ex.get("history", [])
            forecast = ex.get("forecast", [])
            truth = ex.get("truth", [])
            history_str = ", ".join([f"{v:.2f}" for v in history[-30:]])
            forecast_str = ", ".join([f"{v:.2f}" for v in forecast])
            truth_str = ", ".join([f"{v:.2f}" for v in truth])
            ex_date = ex.get("date", "")

            blocks.append(
                f"""Example {idx} (date={ex_date})
History (last 30 days): [{history_str}]
Model forecast ({pred_len}d): [{forecast_str}]
True outcome ({pred_len}d): [{truth_str}]"""
            )

        examples_text = "\n\n".join(blocks)

        prompt = f"""You are analyzing past forecast errors to derive correction rules.

## Teaching Examples
{examples_text}

## Task
Identify systematic deviations between the model forecasts and the true outcomes.
Output a short set of correction rules/heuristics (free text, concise).
Focus on bias, lag, overshoot after spikes, mean reversion, or horizon-specific errors.

Rules:"""

        return prompt


class CoTRFApplyTemplate(PromptTemplate):
    """Apply reflection rules to refine a TSM forecast."""

    def __init__(self):
        super().__init__("CoT-RF-APPLY")

    def format(
        self,
        history: np.ndarray,
        dates: List[str],
        tsm_forecast: np.ndarray,
        rules_text: str,
        pred_len: int = 30,
        currency: str = "EUR",
        **kwargs
    ) -> str:
        history_str = ", ".join([f"{v:.2f}" for v in history[-30:]])
        forecast_str = ", ".join([f"{v:.2f}" for v in tsm_forecast])
        rules_text = rules_text.strip() if rules_text else ""

        prompt = f"""Refine the model forecast using the provided rules.

## Context
Current price: {history[-1]:.2f} {currency}
Recent prices: [{history_str}]

## Model Forecast (to be refined)
[{forecast_str}]

## Correction Rules
{rules_text}

## Task
Apply the rules to adjust the forecast.
Provide your refined {pred_len}-day forecast as JSON:
{{"yhat": [day1_price, day2_price, ..., day{pred_len}_price]}}

Refined forecast (JSON only):"""

        return prompt


class CoTSentRFReflectionTemplate(PromptTemplate):
    """Chain-of-thought reflection to derive correction rules with sentiment."""

    def __init__(self):
        super().__init__("CoT-SENT-RF-REFLECT")

    def get_system_message(self) -> str:
        return (
            "You are an expert financial analyst specializing in carbon markets and EU ETS price forecasting.\n"
            "Your job is to analyze teaching examples of past forecasts vs true outcomes, using sentiment\n"
            "history as additional context, and derive concise correction rules/heuristics.\n"
            "Respond with plain text rules only (no JSON, no code fences)."
        )

    def format(
        self,
        examples: List[Dict],
        pred_len: int = 30,
        currency: str = "EUR",
        history_points: int = 30,
        sentiment_points: int = 30,
        **kwargs
    ) -> str:
        if not examples:
            return "No teaching examples available."

        blocks = []
        for idx, ex in enumerate(examples, start=1):
            history = ex.get("history", [])
            forecast = ex.get("forecast", [])
            truth = ex.get("truth", [])
            sent_hist = ex.get("sentiment_history", [])
            history_str = ", ".join([f"{v:.2f}" for v in history[-history_points:]])
            forecast_str = ", ".join([f"{v:.2f}" for v in forecast])
            truth_str = ", ".join([f"{v:.2f}" for v in truth])
            sent_str = ", ".join([f"{v:.2f}" for v in sent_hist[-sentiment_points:]])
            ex_date = ex.get("date", "")

            blocks.append(
                f"""Example {idx} (date={ex_date})
History (last {history_points} days): [{history_str}]
Sentiment history: [{sent_str}]
Model forecast ({pred_len}d): [{forecast_str}]
True outcome ({pred_len}d): [{truth_str}]"""
            )

        examples_text = "\n\n".join(blocks)

        prompt = f"""You are analyzing past forecast errors to derive correction rules.

## Teaching Examples
{examples_text}

## Task
Identify systematic deviations between the model forecasts and the true outcomes,
considering the sentiment history. Output a short set of correction rules/heuristics
(free text, concise). Focus on bias, lag, overshoot after spikes, mean reversion,
or horizon-specific errors in relation to sentiment.

Rules:"""

        return prompt


class CoTSentRFApplyTemplate(PromptTemplate):
    """Apply reflection rules to refine a TSM forecast with sentiment."""

    def __init__(self):
        super().__init__("CoT-SENT-RF-APPLY")

    def format(
        self,
        history: np.ndarray,
        dates: List[str],
        tsm_forecast: np.ndarray,
        sentiment_history: np.ndarray,
        rules_text: str,
        pred_len: int = 30,
        currency: str = "EUR",
        history_points: int = 30,
        sentiment_points: int = 30,
        **kwargs
    ) -> str:
        history_str = ", ".join([f"{v:.2f}" for v in history[-history_points:]])
        forecast_str = ", ".join([f"{v:.2f}" for v in tsm_forecast])
        sent_str = ", ".join([f"{v:.2f}" for v in sentiment_history[-sentiment_points:]])
        rules_text = rules_text.strip() if rules_text else ""

        prompt = f"""Refine the model forecast using the provided rules.

## Context
Current price: {history[-1]:.2f} {currency}
Recent prices: [{history_str}]
Sentiment history (+1 positive, 0 neutral, -1 negative): [{sent_str}]

## Model Forecast (to be refined)
[{forecast_str}]

## Correction Rules
{rules_text}

## Task
Apply the rules to adjust the forecast.
Provide your refined {pred_len}-day forecast as JSON:
{{"yhat": [day1_price, day2_price, ..., day{pred_len}_price]}}

Refined forecast (JSON only):"""

        return prompt


class CoTSentRFDeltaApplyTemplate(PromptTemplate):
    """Apply reflection rules by proposing bounded deltas to a TSM forecast."""

    def __init__(self):
        super().__init__("CoT-SENT-RF-DELTA-APPLY")

    def format(
        self,
        history: np.ndarray,
        dates: List[str],
        tsm_forecast: np.ndarray,
        sentiment_history: np.ndarray,
        rules_text: str,
        pred_len: int = 30,
        currency: str = "EUR",
        max_delta: Optional[float] = None,
        history_points: int = 30,
        sentiment_points: int = 30,
        **kwargs
    ) -> str:
        history_str = ", ".join([f"{v:.2f}" for v in history[-history_points:]])
        forecast_str = ", ".join([f"{v:.2f}" for v in tsm_forecast])
        sent_str = ", ".join([f"{v:.2f}" for v in sentiment_history[-sentiment_points:]])
        rules_text = rules_text.strip() if rules_text else ""
        max_delta_str = f"{max_delta:.4f}" if max_delta is not None else "N/A"

        prompt = f"""Refine the model forecast using the provided rules by proposing bounded price deltas.

## Context
Current price: {history[-1]:.2f} {currency}
Recent prices: [{history_str}]
Sentiment history (+1 positive, 0 neutral, -1 negative): [{sent_str}]

## Model Forecast (to be refined)
[{forecast_str}]

## Correction Rules
{rules_text}

## Task
Provide a per-day additive adjustment (delta) to the model forecast.
- Delta is in absolute price units ({currency}).
- Keep each delta within ±{max_delta_str} when possible.
- Small corrections are preferred; avoid large rewrites.

Return ONLY JSON with exactly {pred_len} values:
{{"delta": [d1, d2, ..., d{pred_len}]}}

JSON only:"""

        return prompt


def get_template(method: str) -> PromptTemplate:
    """Get prompt template by method name."""
    templates = {
        "DP": DirectPromptTemplate(),
        "CoT": ChainOfThoughtTemplate(),
        "TSM+LLM": RefinementTemplate(),
        "TSM+LLM-NORM-DELTA": NormDeltaRefinementTemplate(),
        "TSM+LLM-COT-SENT": SentimentRefinementTemplate(),
        "CoT-RF": CoTRefinementTemplate(),
        "CoT-RF-REFLECT": CoTRFReflectionTemplate(),
        "CoT-RF-APPLY": CoTRFApplyTemplate(),
        "CoT-SENT-RF-REFLECT": CoTSentRFReflectionTemplate(),
        "CoT-SENT-RF-APPLY": CoTSentRFApplyTemplate(),
        "CoT-SENT-RF-DELTA-APPLY": CoTSentRFDeltaApplyTemplate(),
    }
    
    if method not in templates:
        raise ValueError(f"Unknown method: {method}. Available: {list(templates.keys())}")
    
    return templates[method]
