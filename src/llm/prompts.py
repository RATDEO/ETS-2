"""
Prompt templates for LLM-based forecast refinement.

Implements multiple prompting strategies:
- DP: Direct prompting
- CoT: Chain-of-thought
- CoT-RF: Chain-of-thought with refinement
- TSM+LLM: TSM forecast refinement
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import json


def _format_context_block(
    title: str,
    summary: Optional[Dict],
) -> str:
    if not summary:
        return ""
    lines = [f"## {title}"]
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    return "\n" + "\n".join(lines) + "\n"


def _format_bullet_block(
    title: str,
    lines: Optional[List[str]],
) -> str:
    if not lines:
        return ""
    cleaned = [str(line).strip() for line in lines if str(line).strip()]
    if not cleaned:
        return ""
    block = [f"## {title}"]
    block.extend([f"- {line}" for line in cleaned])
    return "\n" + "\n".join(block) + "\n"


class PromptTemplate:
    """Base class for prompt templates."""
    
    def __init__(self, name: str):
        self.name = name
    
    def format(self, **kwargs) -> str:
        """Format the prompt with given parameters."""
        raise NotImplementedError
    
    def get_system_message(self) -> str:
        """Get the system message for the LLM."""
        return """You are an expert financial analyst specializing in carbon allowance market price forecasting.
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
            "You are an expert financial analyst specializing in carbon allowance market price forecasting.\n"
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
        exogenous_summary: Optional[Dict] = None,
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


class CoTRFReflectionStrictJSONTemplate(PromptTemplate):
    """Strict JSON reflection prompt for deriving correction rules without sentiment."""

    def __init__(self):
        super().__init__("CoT-RF-REFLECT-STRICTJSON")

    def get_system_message(self) -> str:
        return (
            "You are an expert financial analyst specializing in carbon allowance market price forecasting.\n"
            "Analyze the examples and return ONLY strict JSON with this schema:\n"
            "{\"rules\": [\"rule 1\", \"rule 2\", ...]}\n"
            "No markdown, no prose outside JSON, no code fences.\n"
            "Do not show your reasoning. Return the final JSON directly and keep it under 160 tokens."
        )

    def format(
        self,
        examples: List[Dict],
        pred_len: int = 30,
        currency: str = "EUR",
        history_points: int = 30,
        sentiment_points: int = 30,
        exogenous_summary: Optional[Dict] = None,
        **kwargs
    ) -> str:
        if not examples:
            return (
                "Return ONLY JSON:\n"
                '{"rules": ["No teaching examples available. Keep base forecast conservative."]}'
            )

        blocks = []
        for idx, ex in enumerate(examples, start=1):
            history = ex.get("history", [])
            forecast = ex.get("forecast", [])
            truth = ex.get("truth", [])
            exo_summary = ex.get("exogenous_summary")
            history_str = ", ".join([f"{v:.2f}" for v in history[-history_points:]])
            forecast_str = ", ".join([f"{v:.2f}" for v in forecast])
            truth_str = ", ".join([f"{v:.2f}" for v in truth])
            ex_date = ex.get("date", "")
            exo_block = _format_context_block("Official / Exogenous Context", exo_summary).rstrip()

            blocks.append(
                f"""Example {idx} (date={ex_date})
History (last {history_points} days): [{history_str}]
Model forecast ({pred_len}d): [{forecast_str}]
True outcome ({pred_len}d): [{truth_str}]
{exo_block}"""
            )

        examples_text = "\n\n".join(blocks)
        current_exo_block = _format_context_block("Current Official / Exogenous Context", exogenous_summary)

        prompt = f"""Analyze the examples and derive correction rules.

## Teaching Examples
{examples_text}
{current_exo_block}

## Output Contract (MANDATORY)
Return EXACTLY one JSON object with key "rules":
{{"rules": ["short actionable rule 1", "short actionable rule 2", ...]}}

Constraints:
- 4 to 8 rules.
- Each rule must be one short sentence.
- Mention forecast bias/lag/overshoot/mean-reversion and horizon-specific behavior when relevant.
- No extra keys.
- No text outside JSON.

JSON only:"""

        return prompt


class CoTRFHorizonDeltaReflectionStrictJSONTemplate(PromptTemplate):
    """Compact strict-JSON reflection prompt for horizon-delta methods."""

    def __init__(self):
        super().__init__("CoT-RF-HDELTA-REFLECT-STRICTJSON")

    def get_system_message(self) -> str:
        return (
            "You are an expert financial analyst specializing in carbon allowance market price forecasting.\n"
            "Analyze the examples and return ONLY strict JSON with this schema:\n"
            "{\"rules\": [\"rule 1\", \"rule 2\", ...]}\n"
            "No markdown, no prose outside JSON, no code fences.\n"
            "Do not show your reasoning. Return the final JSON directly and keep it under 120 tokens."
        )

    def format(
        self,
        examples: List[Dict],
        pred_len: int = 30,
        currency: str = "EUR",
        history_points: int = 18,
        exogenous_summary: Optional[Dict] = None,
        current_case_summary: Optional[Dict] = None,
        **kwargs
    ) -> str:
        if not examples:
            return (
                "Return ONLY JSON:\n"
                '{"rules": ["No teaching examples available. Keep the base forecast unchanged."]}'
            )

        key_idx = [(1, 0), (5, 4), (20, 19), (30, 29)]
        blocks = []
        for idx, ex in enumerate(examples, start=1):
            history = np.asarray(ex.get("history", []), dtype=float)
            forecast = np.asarray(ex.get("forecast", []), dtype=float)
            truth = np.asarray(ex.get("truth", []), dtype=float)
            ex_date = ex.get("date", "")
            match_rank = ex.get("match_rank")
            match_distance = ex.get("match_distance")
            case_summary = ex.get("case_summary") or {}
            anchor_error_pct = ex.get("anchor_error_pct") or {}
            hist_window = history[-history_points:] if len(history) >= history_points else history
            last_price = float(hist_window[-1]) if len(hist_window) else 0.0
            change_5 = 0.0
            if len(hist_window) >= 5 and abs(hist_window[-5]) > 1e-8:
                change_5 = float((hist_window[-1] / hist_window[-5] - 1.0) * 100.0)
            vol_20 = float(np.std(hist_window[-20:])) if len(hist_window) else 0.0

            anchor_parts = []
            for horizon, arr_idx in key_idx:
                if arr_idx >= len(forecast) or arr_idx >= len(truth):
                    continue
                f_val = float(forecast[arr_idx])
                t_val = float(truth[arr_idx])
                err_val = float(f_val - t_val)
                anchor_parts.append(
                    f"h{horizon}: forecast={f_val:.2f}, true={t_val:.2f}, err={err_val:+.2f}"
                )
            pct_parts = []
            for horizon, err_pct in sorted(anchor_error_pct.items()):
                pct_parts.append(f"h{int(horizon)}={float(err_pct):+.2f}%")
            extra_lines = []
            if match_rank is not None:
                extra_lines.append(f"- match_rank: {int(match_rank)}")
            if match_distance is not None:
                extra_lines.append(f"- match_distance: {float(match_distance):.2f}")
            for key, value in case_summary.items():
                extra_lines.append(f"- {key}: {value}")
            extra_block = ("\n" + "\n".join(extra_lines)) if extra_lines else ""

            blocks.append(
                f"""Example {idx} (date={ex_date})
- last_price: {last_price:.2f} {currency}
- change_5d_pct: {change_5:+.2f}
- vol_20d: {vol_20:.2f}
- anchors: {"; ".join(anchor_parts)}
- anchor_error_pct: {"; ".join(pct_parts)}{extra_block}"""
            )

        current_exo_block = _format_context_block("Current Official / Exogenous Context", exogenous_summary)
        current_case_block = _format_context_block("Current Case", current_case_summary)
        examples_text = "\n\n".join(blocks)

        prompt = f"""Analyze the teaching examples and derive short correction rules for the current case only.

{current_case_block}

## Teaching Examples
{examples_text}
{current_exo_block}

## Output Contract (MANDATORY)
Return EXACTLY one JSON object with key "rules":
{{"rules": ["short actionable rule 1", "short actionable rule 2", ...]}}

Constraints:
- 3 to 5 rules.
- Each rule must be one short sentence.
- Focus on the current case, using only the matched-example evidence above.
- Mention bias, lag, overshoot, mean reversion, or horizons that should stay near zero when evidence is weak.
- If the matched examples disagree, include a rule to keep the affected horizon unchanged.
- No extra keys.
- No text outside JSON.

JSON only:"""

        return prompt


class CoTRFHorizonDeltaStructuredReflectionStrictJSONTemplate(PromptTemplate):
    """Structured strict-JSON reflection prompt for horizon-delta methods."""

    def __init__(self):
        super().__init__("CoT-RF-HDELTA-REFLECT-STRUCTURED-STRICTJSON")

    def get_system_message(self) -> str:
        return (
            "You are an expert financial analyst specializing in carbon allowance market price forecasting.\n"
            "Analyze the matched examples and return ONLY strict JSON with this schema:\n"
            "{\"horizons\": {\"h1\": {\"mode\": \"freeze|adjust\", \"preferred_sign\": \"positive|negative|zero\", "
            "\"confidence\": \"low|medium|high\", \"magnitude\": \"zero|tiny|small|medium\", \"reason\": \"...\"}, ...}}\n"
            "No markdown, no prose outside JSON, no code fences.\n"
            "Do not show your reasoning. Keep reasons short and concrete.\n"
            "Make one pass through the evidence, avoid self-critique loops, and stop after the first complete horizon plan."
        )

    def format(
        self,
        examples: List[Dict],
        pred_len: int = 30,
        currency: str = "EUR",
        history_points: int = 18,
        exogenous_summary: Optional[Dict] = None,
        current_case_summary: Optional[Dict] = None,
        key_horizons: Optional[List[int]] = None,
        **kwargs
    ) -> str:
        reasoning_style = str(kwargs.get("reasoning_style", "default")).strip().lower()
        compact_reasoning = bool(kwargs.get("compact_reasoning", False))
        reflection_memory = kwargs.get("reflection_memory") or []
        horizons = [int(h) for h in (key_horizons or [1, 5, 20, 30]) if 1 <= int(h) <= int(pred_len)]
        if not examples:
            horizon_entries = ", ".join(
                [
                    f'"h{h}": {{"mode": "freeze", "preferred_sign": "zero", "confidence": "high", '
                    f'"magnitude": "zero", "reason": "No matched examples available."}}'
                    for h in horizons
                ]
            )
            return "Return ONLY JSON:\n" + '{"horizons": {' + horizon_entries + "}}"

        blocks = []
        for idx, ex in enumerate(examples, start=1):
            history = np.asarray(ex.get("history", []), dtype=float)
            forecast = np.asarray(ex.get("forecast", []), dtype=float)
            truth = np.asarray(ex.get("truth", []), dtype=float)
            ex_date = ex.get("date", "")
            match_rank = ex.get("match_rank")
            match_distance = ex.get("match_distance")
            case_summary = ex.get("case_summary") or {}
            anchor_error_pct = ex.get("anchor_error_pct") or {}
            hist_window = history[-history_points:] if len(history) >= history_points else history
            last_price = float(hist_window[-1]) if len(hist_window) else 0.0
            change_5 = 0.0
            if len(hist_window) >= 5 and abs(hist_window[-5]) > 1e-8:
                change_5 = float((hist_window[-1] / hist_window[-5] - 1.0) * 100.0)
            vol_20 = float(np.std(hist_window[-20:])) if len(hist_window) else 0.0
            anchor_parts = []
            for horizon in horizons:
                arr_idx = int(horizon) - 1
                if arr_idx >= len(forecast) or arr_idx >= len(truth):
                    continue
                f_val = float(forecast[arr_idx])
                t_val = float(truth[arr_idx])
                err_pct = float(anchor_error_pct.get(int(horizon), 0.0))
                anchor_parts.append(
                    f"h{int(horizon)}: forecast={f_val:.2f}, true={t_val:.2f}, err_pct={err_pct:+.2f}%"
                )
            extra_lines = []
            if match_rank is not None:
                extra_lines.append(f"- match_rank: {int(match_rank)}")
            if match_distance is not None:
                extra_lines.append(f"- match_distance: {float(match_distance):.2f}")
            for key, value in case_summary.items():
                extra_lines.append(f"- {key}: {value}")
            extra_block = ("\n" + "\n".join(extra_lines)) if extra_lines else ""
            blocks.append(
                f"""Example {idx} (date={ex_date})
- last_price: {last_price:.2f} {currency}
- change_5d_pct: {change_5:+.2f}
- vol_20d: {vol_20:.2f}
- anchors: {"; ".join(anchor_parts)}{extra_block}"""
            )

        current_exo_block = _format_context_block("Current Official / Exogenous Context", exogenous_summary)
        current_case_block = _format_context_block("Current Case", current_case_summary)
        memory_block = _format_bullet_block("Persistent UK Error Memory", reflection_memory)
        example_block = "\n\n".join(blocks)
        horizon_keys = ", ".join([f'"h{int(h)}"' for h in horizons])

        reason_constraint_lines: List[str] = []

        if reasoning_style == "least_to_most":
            task_lines = [
                "1. Decide h1 first. Keep it frozen unless the matched examples strongly agree on a correction.",
                "2. Decide h5 next, conditioned on the h1 decision and the short-horizon evidence.",
                "3. Decide h20 only after h5, using the dominant medium-horizon pattern across examples.",
                "4. Decide h30 last. Only adjust it if the long-horizon evidence is both consistent and stronger than the shorter-horizon evidence.",
                "5. If later horizons would contradict earlier supported horizons, freeze the later horizon instead.",
            ]
        elif reasoning_style == "self_refine":
            task_lines = [
                "1. Draft a tentative decision for each horizon from the matched examples.",
                "2. Critique that draft for unsupported sign flips, over-large h20 or h30 moves, and contradictions with the examples.",
                "3. Revise the draft and return only the final per-horizon JSON.",
                "4. If the critique finds weak evidence, downgrade the horizon to freeze.",
            ]
        elif reasoning_style == "chain_of_verification":
            task_lines = [
                "1. For each horizon, verify the proposed direction against the matched examples before deciding.",
                "2. Treat a horizon as unverified if the examples are mixed, sparse, or only support a different horizon.",
                "3. Freeze any unverified horizon instead of carrying over a nearby-horizon bias.",
                "4. Use adjust only for horizons with verified support in the examples.",
            ]
            reason_constraint_lines.append("If possible, state whether the horizon was verified by the matched examples.")
        elif reasoning_style == "step_back":
            task_lines = [
                "1. First infer the higher-level market regime from the matched examples: undershoot, overshoot, trend persistence, or mean reversion.",
                "2. Then map that regime to each horizon without skipping directly to numeric adjustments.",
                "3. Freeze horizons where the regime does not clearly imply a move.",
                "4. Prefer one coherent regime-level explanation over disconnected horizon tweaks.",
            ]
        elif reasoning_style == "analogical_step_back":
            task_lines = [
                "1. Identify the one or two matched examples that are most analogous to the current case.",
                "2. Step back and infer the higher-level regime those analogies imply for the current case.",
                "3. Transfer only the reusable regime pattern, not the exact path, into the horizon decisions.",
                "4. Freeze horizons where the analogy does not clearly transfer.",
            ]
            reason_constraint_lines.append("When possible, note which matched example the analogy came from.")
        elif reasoning_style == "thought_propagation":
            task_lines = [
                "1. Extract the reusable reasoning pattern from the strongest matched examples.",
                "2. Propagate that pattern forward into the current case horizon by horizon.",
                "3. Keep only the parts of the pattern that stay consistent across the matched examples.",
                "4. Do not propagate a correction into a horizon unless the example pattern still supports it there.",
            ]
            reason_constraint_lines.append("When possible, say which matched-example pattern is being propagated.")
        elif reasoning_style == "self_discover":
            task_lines = [
                "1. First choose the reasoning modules that fit this case: regime abstraction, example comparison, contradiction check, and bounded sizing.",
                "2. Apply those modules in the order that best fits the current case before deciding the horizons.",
                "3. Freeze horizons when the chosen modules do not converge on a consistent action.",
                "4. Return only the final JSON after the reasoning structure has been selected internally.",
            ]
        elif reasoning_style == "self_verification":
            task_lines = [
                "1. Build a candidate regime-level horizon plan from the matched examples.",
                "2. Verify that plan backward: would each horizon decision still be justified if you explained it from the final action back to the examples?",
                "3. If backward verification fails for a horizon, freeze it.",
                "4. Keep only the decisions that survive both forward reasoning and backward verification.",
            ]
            reason_constraint_lines.append("Each reason should make clear that the final decision survived verification or that evidence was mixed.")
        elif reasoning_style == "skeleton":
            task_lines = [
                "1. First decide only the action skeleton: which horizons freeze and which adjust.",
                "2. Then fill in sign, confidence, and magnitude for the adjustable horizons.",
                "3. Keep the skeleton conservative if adding an extra adjustable horizon would create unsupported drift.",
                "4. If the skeleton is uncertain, prefer fewer adjustments.",
            ]
        elif reasoning_style == "self_ask":
            task_lines = [
                "1. Internally ask: does the evidence justify moving h1, h5, h20, and h30?",
                "2. Answer each question from the matched examples before deciding the JSON fields.",
                "3. Freeze any horizon whose answer is uncertain, mixed, or only indirectly supported.",
                "4. Do not let one strong horizon answer force changes at the others.",
            ]
        elif reasoning_style == "react_evidence":
            task_lines = [
                "1. Observe the matched examples and identify the most relevant evidence for each horizon.",
                "2. Reason about what that evidence implies for base-model bias at each horizon.",
                "3. Act only where the evidence clearly supports a bounded correction.",
                "4. If you cannot connect evidence to action, freeze the horizon.",
            ]
            reason_constraint_lines.append("When possible, anchor the reason in a specific matched-example pattern.")
        elif reasoning_style == "rarr_attribution":
            task_lines = [
                "1. Ground every horizon decision in explicit matched-example evidence, not generic market intuition.",
                "2. If you cannot attribute a horizon move to the examples, freeze it.",
                "3. Prefer evidence-supported freezes over unsupported adjustments.",
                "4. Keep h20 and h30 especially strict about attribution to real examples.",
            ]
            reason_constraint_lines.append("Each reason should cite example rank or say that the matched examples are mixed.")
        elif reasoning_style == "self_rag":
            task_lines = [
                "1. Retrieve the strongest matched-example evidence for each horizon from the examples shown.",
                "2. Critique whether that evidence is current, consistent, and sufficient for the horizon.",
                "3. Only after that critique, decide freeze or adjust.",
                "4. If retrieved evidence is stale, contradictory, or too weak, freeze the horizon.",
            ]
            reason_constraint_lines.append("Each reason should make clear whether the retrieved matched-example evidence was sufficient or mixed.")
        elif reasoning_style == "tree_of_thought":
            task_lines = [
                "1. Consider multiple candidate horizon plans internally: freeze-heavy, medium-adjustment, and long-horizon-adjustment.",
                "2. Compare those plans against the matched examples and reject any plan that creates unsupported drift.",
                "3. Choose the plan with the best evidence fit and smallest unnecessary change.",
                "4. Return only the final selected plan as JSON.",
            ]
        elif reasoning_style == "debate":
            task_lines = [
                "1. Internally consider an optimistic corrector that wants to adjust and a skeptical verifier that wants to freeze.",
                "2. Keep only the horizon decisions both sides could defend using the matched examples.",
                "3. If the two views would disagree materially, freeze that horizon.",
                "4. Prefer conservative consensus over aggressive path rewriting.",
            ]
        elif reasoning_style == "plan_and_solve":
            task_lines = [
                "1. First make a short internal plan: which horizons should freeze, which should adjust, and why.",
                "2. Then convert that plan into the required per-horizon JSON decisions.",
                "3. Prefer small, explicit decisions over broad path rewrites.",
                "4. If the evidence is mixed, freeze the horizon rather than improvising a correction.",
            ]
        elif reasoning_style == "harmonized":
            task_lines = [
                "1. Harmonize the matched examples into one consistent reasoning pattern for the current case.",
                "2. Ignore one-off outliers if most examples agree on the horizon direction.",
                "3. Prefer the dominant sign and confidence pattern across examples over example-specific noise.",
                "4. Freeze horizons where no single pattern clearly dominates.",
            ]
        elif reasoning_style == "minimal":
            task_lines = [
                "1. Choose the smallest justified action for each horizon.",
                "2. Freeze any horizon with weak or mixed evidence.",
                "3. Do not invent long-horizon drift when the examples are inconclusive.",
            ]
        else:
            task_lines = [
                "For each horizon, decide whether to freeze or adjust based only on the matched-example evidence.",
                "Use adjust only when the examples consistently point in one direction.",
                "Use freeze when evidence is weak, mixed, or horizon coupling would create a new drift.",
            ]
        task_block = "\n".join([f"- {line}" for line in task_lines])

        prompt = f"""Analyze the matched examples and produce per-horizon guidance for the current case only.

{current_case_block}
{memory_block}

## Teaching Examples
{example_block}
{current_exo_block}

## Task
For each horizon in {horizon_keys}, decide:
- `mode`: `freeze` or `adjust`
- `preferred_sign`: `positive`, `negative`, or `zero`
- `confidence`: `low`, `medium`, or `high`
- `magnitude`: `zero`, `tiny`, `small`, or `medium`
- `reason`: one short sentence grounded in the matched examples

Decision protocol:
{task_block}

## Output Contract (MANDATORY)
Return EXACTLY one JSON object:
{{"horizons": {{{", ".join([f'"h{int(h)}": {{"mode": "...", "preferred_sign": "...", "confidence": "...", "magnitude": "...", "reason": "..."}}' for h in horizons])}}}}}

Constraints:
- No extra keys.
- No text outside JSON.
- Keep each reason short.
- Make one pass through the evidence and stop after the first complete horizon plan.
- Do not re-open earlier horizon decisions once chosen unless the matched examples directly contradict them.
- Keep internal reasoning compact; avoid long self-critique loops.
{chr(10).join([f"- {line}" for line in reason_constraint_lines]) if reason_constraint_lines else ""}

JSON only:"""

        if compact_reasoning:
            prompt += (
                "\n\nAdditional instruction for reasoning models:\n"
                "- Decide the regime once.\n"
                "- Convert it to the four horizon decisions once.\n"
                "- Do not revisit or debate alternative plans after that."
            )

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
            "You are an expert financial analyst specializing in carbon allowance market price forecasting.\n"
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
        exogenous_summary: Optional[Dict] = None,
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
            exo_summary = ex.get("exogenous_summary")
            history_str = ", ".join([f"{v:.2f}" for v in history[-history_points:]])
            forecast_str = ", ".join([f"{v:.2f}" for v in forecast])
            truth_str = ", ".join([f"{v:.2f}" for v in truth])
            sent_str = ", ".join([f"{v:.2f}" for v in sent_hist[-sentiment_points:]])
            ex_date = ex.get("date", "")
            exo_block = _format_context_block("Official / Exogenous Context", exo_summary).rstrip()

            blocks.append(
                f"""Example {idx} (date={ex_date})
History (last {history_points} days): [{history_str}]
Sentiment history: [{sent_str}]
Model forecast ({pred_len}d): [{forecast_str}]
True outcome ({pred_len}d): [{truth_str}]
{exo_block}"""
            )

        examples_text = "\n\n".join(blocks)
        current_exo_block = _format_context_block("Current Official / Exogenous Context", exogenous_summary)

        prompt = f"""You are analyzing past forecast errors to derive correction rules.

## Teaching Examples
{examples_text}
{current_exo_block}

## Task
Identify systematic deviations between the model forecasts and the true outcomes,
considering the sentiment history and current official event context when available. Output a short set of correction rules/heuristics
(free text, concise). Focus on bias, lag, overshoot after spikes, mean reversion,
or horizon-specific errors in relation to sentiment.

Rules:"""

        return prompt


class CoTSentRFReflectionStrictJSONTemplate(PromptTemplate):
    """Strict JSON reflection prompt for deriving correction rules with sentiment."""

    def __init__(self):
        super().__init__("CoT-SENT-RF-REFLECT-STRICTJSON")

    def get_system_message(self) -> str:
        return (
            "You are an expert financial analyst specializing in carbon allowance market price forecasting.\n"
            "Analyze the examples and return ONLY strict JSON with this schema:\n"
            "{\"rules\": [\"rule 1\", \"rule 2\", ...]}\n"
            "No markdown, no prose outside JSON, no code fences.\n"
            "Do not show your reasoning. Return the final JSON directly and keep it under 160 tokens."
        )

    def format(
        self,
        examples: List[Dict],
        pred_len: int = 30,
        currency: str = "EUR",
        history_points: int = 30,
        sentiment_points: int = 30,
        exogenous_summary: Optional[Dict] = None,
        **kwargs
    ) -> str:
        if not examples:
            return (
                "Return ONLY JSON:\n"
                '{"rules": ["No teaching examples available. Keep base forecast conservative."]}'
            )

        blocks = []
        for idx, ex in enumerate(examples, start=1):
            history = ex.get("history", [])
            forecast = ex.get("forecast", [])
            truth = ex.get("truth", [])
            sent_hist = ex.get("sentiment_history", [])
            exo_summary = ex.get("exogenous_summary")
            history_str = ", ".join([f"{v:.2f}" for v in history[-history_points:]])
            forecast_str = ", ".join([f"{v:.2f}" for v in forecast])
            truth_str = ", ".join([f"{v:.2f}" for v in truth])
            sent_str = ", ".join([f"{v:.2f}" for v in sent_hist[-sentiment_points:]])
            ex_date = ex.get("date", "")
            exo_block = _format_context_block("Official / Exogenous Context", exo_summary).rstrip()

            blocks.append(
                f"""Example {idx} (date={ex_date})
History (last {history_points} days): [{history_str}]
Sentiment history: [{sent_str}]
Model forecast ({pred_len}d): [{forecast_str}]
True outcome ({pred_len}d): [{truth_str}]
{exo_block}"""
            )

        examples_text = "\n\n".join(blocks)
        current_exo_block = _format_context_block("Current Official / Exogenous Context", exogenous_summary)

        prompt = f"""Analyze the examples and derive correction rules.

## Teaching Examples
{examples_text}
{current_exo_block}

## Output Contract (MANDATORY)
Return EXACTLY one JSON object with key "rules":
{{"rules": ["short actionable rule 1", "short actionable rule 2", ...]}}

Constraints:
- 4 to 8 rules.
- Each rule must be one short sentence.
- Mention forecast bias/lag/overshoot/mean-reversion and sentiment interaction when relevant.
- No extra keys.
- No text outside JSON.

JSON only:"""

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
        exogenous_summary: Optional[Dict] = None,
        **kwargs
    ) -> str:
        history_str = ", ".join([f"{v:.2f}" for v in history[-history_points:]])
        forecast_str = ", ".join([f"{v:.2f}" for v in tsm_forecast])
        sent_str = ", ".join([f"{v:.2f}" for v in sentiment_history[-sentiment_points:]])
        rules_text = rules_text.strip() if rules_text else ""
        exo_block = _format_context_block("Official / Exogenous Context", exogenous_summary)

        prompt = f"""Refine the model forecast using the provided rules.

## Context
Current price: {history[-1]:.2f} {currency}
Recent prices: [{history_str}]
Sentiment history (+1 positive, 0 neutral, -1 negative): [{sent_str}]
{exo_block}

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


class CoTSentRFApplyStrictJSONTemplate(PromptTemplate):
    """Strict JSON apply prompt for refining a TSM forecast with sentiment."""

    def __init__(self):
        super().__init__("CoT-SENT-RF-APPLY-STRICTJSON")

    def get_system_message(self) -> str:
        return (
            "Return ONLY strict JSON with this schema:\n"
            '{"yhat": [n1, n2, ..., n30]}\n'
            "No markdown, no explanation, no code fences, no extra keys."
        )

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
        exogenous_summary: Optional[Dict] = None,
        **kwargs
    ) -> str:
        history_str = ", ".join([f"{v:.2f}" for v in history[-history_points:]])
        forecast_str = ", ".join([f"{v:.2f}" for v in tsm_forecast])
        sent_str = ", ".join([f"{v:.2f}" for v in sentiment_history[-sentiment_points:]])
        rules_text = rules_text.strip() if rules_text else ""
        exo_block = _format_context_block("Official / Exogenous Context", exogenous_summary)

        prompt = f"""Refine the model forecast using the provided rules.

## Context
Current price: {history[-1]:.2f} {currency}
Recent prices: [{history_str}]
Sentiment history (+1 positive, 0 neutral, -1 negative): [{sent_str}]
{exo_block}

## Model Forecast (to be refined)
[{forecast_str}]

## Correction Rules
{rules_text}

## Output Contract (MANDATORY)
Return EXACTLY one JSON object:
{{"yhat": [day1_price, day2_price, ..., day{pred_len}_price]}}

Hard constraints:
- Exactly {pred_len} numeric values in yhat.
- Keep values in realistic EU ETS range (positive, non-zero).
- No code fences, no markdown, no extra keys, no explanation text.
- First character must be '{{' and last character must be '}}'.

JSON only:"""

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


class CoTSentRFHorizonDeltaApplyTemplate(PromptTemplate):
    """Apply reflection rules by proposing bounded percentage deltas at key horizons."""

    def __init__(self):
        super().__init__("CoT-SENT-RF-HDELTA-APPLY")

    def format(
        self,
        history: np.ndarray,
        dates: List[str],
        tsm_forecast: np.ndarray,
        sentiment_history: np.ndarray,
        rules_text: str,
        pred_len: int = 30,
        currency: str = "EUR",
        key_horizons: Optional[List[int]] = None,
        max_adjustment_pct: float = 3.0,
        frozen_horizons: Optional[List[int]] = None,
        sentiment_secondary: bool = False,
        history_points: int = 30,
        sentiment_points: int = 30,
        exogenous_summary: Optional[Dict] = None,
        **kwargs
    ) -> str:
        history_str = ", ".join([f"{v:.2f}" for v in history[-history_points:]])
        forecast_str = ", ".join([f"{v:.2f}" for v in tsm_forecast])
        sent_str = ", ".join([f"{v:.2f}" for v in sentiment_history[-sentiment_points:]])
        rules_text = rules_text.strip() if rules_text else ""
        exo_block = _format_context_block("Official / Exogenous Context", exogenous_summary)
        horizons = key_horizons or [1, 5, 20, 30]
        horizons = [int(h) for h in horizons if 1 <= int(h) <= int(pred_len)]
        frozen = sorted({int(h) for h in (frozen_horizons or []) if int(h) in horizons})
        horizon_keys = ", ".join([f"\"h{h}\"" for h in horizons])
        horizon_lines = []
        for h in horizons:
            if h in frozen:
                horizon_lines.append(f"- h{h}: MUST be 0.0 exactly. This horizon is frozen to the base TSM forecast.")
            else:
                horizon_lines.append(f"- h{h}: percentage change to apply to day {h} of the model forecast.")
        horizon_bullets = "\n".join(horizon_lines)
        sentiment_guidance = (
            "- Treat sentiment as secondary evidence only. Use it to modulate a correction implied by price action; do not create or reverse a correction from sentiment alone.\n"
            "- If price action and sentiment conflict, prioritize price action and keep the adjustment small."
            if sentiment_secondary
            else "- Use sentiment and price action jointly when forming corrections."
        )

        prompt = f"""Refine the model forecast using the provided rules by proposing small percentage adjustments at key horizons only.

## Context
Current price: {history[-1]:.2f} {currency}
Recent prices: [{history_str}]
Sentiment history (+1 positive, 0 neutral, -1 negative): [{sent_str}]
{exo_block}

## Model Forecast (to be refined)
[{forecast_str}]

## Correction Rules
{rules_text}

## Task
Provide percentage adjustments for these key horizons only:
{horizon_bullets}

Constraints:
- Each adjustment is a percentage, not a price level.
- Positive means increase the model forecast, negative means decrease it.
- Keep each adjustment within +/-{max_adjustment_pct:.2f}%.
- Use small corrections. If uncertain, prefer values near 0.
- Intermediate days will be interpolated automatically.
{sentiment_guidance}

Return ONLY one JSON object:
{{"adjustments": {{{horizon_keys}}}}}

Example:
{{"adjustments": {{"h1": 0.3, "h5": -0.5, "h20": 0.8, "h30": -0.2}}}}

JSON only:"""

        return prompt


class CoTRFHorizonDeltaApplyTemplate(PromptTemplate):
    """Apply reflection rules by proposing bounded percentage deltas at key horizons."""

    def __init__(self):
        super().__init__("CoT-RF-HDELTA-APPLY")

    def format(
        self,
        history: np.ndarray,
        dates: List[str],
        tsm_forecast: np.ndarray,
        rules_text: str,
        pred_len: int = 30,
        currency: str = "EUR",
        key_horizons: Optional[List[int]] = None,
        max_adjustment_pct: float = 3.0,
        per_horizon_max_adjustment_pct: Optional[Dict[int, float]] = None,
        frozen_horizons: Optional[List[int]] = None,
        exogenous_summary: Optional[Dict] = None,
        history_points: int = 30,
        current_case_summary: Optional[Dict] = None,
        matched_examples_summary: Optional[List[str]] = None,
        horizon_guidance_summary: Optional[List[str]] = None,
        structured_horizon_guidance: Optional[Dict[int, Dict[str, str]]] = None,
        numeric_tool_enabled: bool = False,
        numeric_tool_name: str = "get_numeric_analysis",
        case_retrieval_tool_enabled: bool = False,
        case_retrieval_tool_name: str = "get_structured_case_retrieval",
        delta_verifier_tool_enabled: bool = False,
        delta_verifier_tool_name: str = "verify_hdelta_adjustments",
        market_microstructure_tool_enabled: bool = False,
        market_microstructure_tool_name: str = "get_market_microstructure_state",
        freeze_counterexample_tool_enabled: bool = False,
        freeze_counterexample_tool_name: str = "get_freeze_counterexample",
        **kwargs
    ) -> str:
        apply_style = str(kwargs.get("apply_style", "default")).strip().lower()
        reflection_memory = kwargs.get("reflection_memory") or []
        history_str = ", ".join([f"{v:.2f}" for v in history[-history_points:]])
        forecast_str = ", ".join([f"{v:.2f}" for v in tsm_forecast])
        rules_text = rules_text.strip() if rules_text else ""
        exo_block = _format_context_block("Official / Exogenous Context", exogenous_summary)
        current_case_block = _format_context_block("Current Case", current_case_summary)
        memory_block = _format_bullet_block("Persistent UK Error Memory", reflection_memory)
        horizons = key_horizons or [1, 5, 20, 30]
        horizons = [int(h) for h in horizons if 1 <= int(h) <= int(pred_len)]
        frozen = sorted({int(h) for h in (frozen_horizons or []) if int(h) in horizons})
        per_horizon_max = {
            int(h): float((per_horizon_max_adjustment_pct or {}).get(int(h), max_adjustment_pct))
            for h in horizons
        }
        horizon_keys = ", ".join([f"\"h{h}\"" for h in horizons])
        horizon_lines = []
        for h in horizons:
            if h in frozen:
                horizon_lines.append(f"- h{h}: MUST be 0.0 exactly. This horizon is frozen to the base forecast.")
            else:
                horizon_lines.append(
                    f"- h{h}: percentage change to apply to day {h} of the base forecast, keep within +/-{per_horizon_max[int(h)]:.2f}%."
                )
        horizon_bullets = "\n".join(horizon_lines)
        matched_block = ""
        if matched_examples_summary:
            matched_lines = [f"- {line}" for line in matched_examples_summary]
            matched_block = "\n## Matched Example Evidence\n" + "\n".join(matched_lines) + "\n"
        guidance_block = ""
        if horizon_guidance_summary:
            guidance_lines = [f"- {line}" for line in horizon_guidance_summary]
            guidance_block = "\n## Horizon Guidance\n" + "\n".join(guidance_lines) + "\n"
        structured_block = ""
        if structured_horizon_guidance:
            magnitude_caps = {"zero": "0%", "tiny": "25%", "small": "50%", "medium": "75%"}
            structured_lines = []
            for h in horizons:
                payload = (structured_horizon_guidance or {}).get(int(h)) or {}
                mode = str(payload.get("mode", "")).strip().lower()
                sign = str(payload.get("preferred_sign", "")).strip().lower()
                confidence = str(payload.get("confidence", "")).strip().lower()
                magnitude = str(payload.get("magnitude", "")).strip().lower()
                reason = str(payload.get("reason", "")).strip()
                if not mode:
                    continue
                structured_lines.append(
                    f"- h{int(h)}: mode={mode}; sign={sign}; confidence={confidence}; "
                    f"magnitude={magnitude} (keep within about {magnitude_caps.get(magnitude, '100%')} of the horizon bound); "
                    f"reason={reason}"
                )
            if structured_lines:
                structured_block = "\n## Structured Horizon Decisions\n" + "\n".join(structured_lines) + "\n"
        tool_block = ""
        tool_lines = []
        if case_retrieval_tool_enabled:
            tool_lines.extend(
                [
                    f"- `{case_retrieval_tool_name}` returns structured matched UK ETS cases with true historical forecast errors and hindsight feedback.",
                    f"- Call `{case_retrieval_tool_name}` before deciding direction when matched-example evidence is not already explicit in the prompt.",
                ]
            )
        if market_microstructure_tool_enabled:
            tool_lines.extend(
                [
                    f"- `{market_microstructure_tool_name}` returns structured UK auction and ICAP market-state features for the current case.",
                    f"- Call `{market_microstructure_tool_name}` before judging whether current microstructure supports a non-zero move.",
                ]
            )
        if freeze_counterexample_tool_enabled:
            tool_lines.extend(
                [
                    f"- `{freeze_counterexample_tool_name}` returns one similar low-error historical counterexample that argues for shrinking unsupported moves.",
                    f"- Call `{freeze_counterexample_tool_name}` before finalizing any medium or long-horizon adjustment.",
                ]
            )
        if numeric_tool_enabled:
            tool_lines.extend(
                [
                    f"- `{numeric_tool_name}` returns exact current momentum, volatility, base horizon drifts, and allowed bounds.",
                    f"- Call `{numeric_tool_name}` before finalizing adjustment size.",
                ]
            )
        if delta_verifier_tool_enabled:
            tool_lines.extend(
                [
                    f"- `{delta_verifier_tool_name}` verifies tentative horizon deltas against bounds and structured guidance, and returns `verified_adjustments`.",
                    f"- Call `{delta_verifier_tool_name}` after drafting tentative adjustments; final JSON should normally match `verified_adjustments`, or move closer to 0.0 without changing sign.",
                ]
            )
        if tool_lines:
            tool_block = "\n## Available Tools\n" + "\n".join(tool_lines) + "\n"
            if sum(
                int(flag)
                for flag in (
                    case_retrieval_tool_enabled,
                    market_microstructure_tool_enabled,
                    freeze_counterexample_tool_enabled,
                    numeric_tool_enabled,
                    delta_verifier_tool_enabled,
                )
            ) > 1:
                tool_block += (
                    "- Call each available tool exactly once before returning final JSON.\n"
                    "- Recommended order: retrieval / market-state / counterexample -> numeric analysis -> verifier.\n"
                )

        if apply_style == "program_of_thought":
            style_lines = [
                "Before emitting JSON, internally compute for each horizon: base anchor, allowed bound, preferred sign, and bounded adjustment.",
                "Use arithmetic and bounds, not narrative intuition.",
                "Shrink any long-horizon move that is larger than the matched-example evidence supports.",
                "If a later-horizon move would create new unsupported drift, reduce it toward 0.0.",
            ]
        elif apply_style == "verifier_program":
            style_lines = [
                "Start from a tentative bounded adjustment for each actionable horizon, then verify it against the structured guidance before returning it.",
                "If the sign, confidence, or cited evidence does not support the tentative move, shrink it toward 0.0 or set it to 0.0.",
                "Treat h20 and h30 as requiring stricter support than h5.",
                "Use arithmetic and bounds, not narrative intuition.",
            ]
        elif apply_style == "citation_bounded":
            style_lines = [
                "Only keep a non-zero adjustment when the structured guidance gives a clear sign and the reason is evidence-grounded.",
                "If the guidance says the evidence is mixed, sparse, or uncertain, return 0.0 for that horizon.",
                "Even when adjusting, stay close to 0 and remain well within the stated bound.",
                "Do not create a new long-horizon drift from weakly supported reasons.",
            ]
        elif apply_style == "minimal":
            style_lines = [
                "Choose the smallest useful change at each horizon.",
                "Do not create new h30 drift unless the evidence is unusually strong.",
                "When uncertain, return 0.0.",
            ]
        else:
            style_lines = [
                "Use small corrections. If uncertain, prefer values near 0.",
                "Do not rewrite the full 30-day path.",
                "If the matched examples conflict for a horizon, return 0.0 for that horizon.",
            ]
        style_block = "\n".join([f"- {line}" for line in style_lines])

        prompt = f"""Refine the model forecast using the provided rules by proposing small percentage adjustments at key horizons only.

## Context
Current price: {history[-1]:.2f} {currency}
Recent prices: [{history_str}]
{exo_block}
{current_case_block}{memory_block}{matched_block}{guidance_block}{structured_block}
{tool_block}

## Base Model Forecast
[{forecast_str}]

## Correction Rules
{rules_text}

## Task
Provide percentage adjustments for these key horizons only:
{horizon_bullets}

Constraints:
- Each adjustment is a percentage, not a price level.
- Positive means increase the base forecast, negative means decrease it.
- Keep each adjustment within its stated horizon-specific bound.
- Intermediate days will be interpolated automatically.
- Decision procedure:
{style_block}
- For any non-frozen horizon marked actionable, use the preferred sign from the guidance unless the current case clearly contradicts it.
- If structured horizon decisions are provided, treat them as the primary contract for sign, confidence, and size.
- Respect the structured `mode`: `freeze` means 0.0, `adjust` means make a bounded move.
- Respect the structured `magnitude`: `zero` means 0.0, `tiny` means very close to 0, `small` means well below the bound, `medium` means still below the bound.
- Do not return all zeros unless every actionable horizon is genuinely unsupported by the guidance.
- If tools are available, use tool outputs as the source of truth for matched cases, arithmetic, bounds, and verified adjustments.

Return ONLY one JSON object:
{{"adjustments": {{{horizon_keys}}}}}

Example:
{{"adjustments": {{"h1": 0.3, "h5": -0.5, "h20": 0.8, "h30": -0.2}}}}

JSON only:"""

        return prompt


class CoTSentRFHorizonPriceApplyTemplate(PromptTemplate):
    """Apply reflection rules by proposing bounded absolute anchor prices at key horizons."""

    def __init__(self):
        super().__init__("CoT-SENT-RF-HPRICE-APPLY")

    def format(
        self,
        history: np.ndarray,
        dates: List[str],
        tsm_forecast: np.ndarray,
        sentiment_history: np.ndarray,
        rules_text: str,
        pred_len: int = 30,
        currency: str = "EUR",
        key_horizons: Optional[List[int]] = None,
        max_adjustment_pct: float = 1.0,
        frozen_horizons: Optional[List[int]] = None,
        sentiment_secondary: bool = False,
        history_points: int = 30,
        sentiment_points: int = 30,
        exogenous_summary: Optional[Dict] = None,
        **kwargs
    ) -> str:
        history_str = ", ".join([f"{v:.2f}" for v in history[-history_points:]])
        forecast_str = ", ".join([f"{v:.2f}" for v in tsm_forecast])
        sent_str = ", ".join([f"{v:.2f}" for v in sentiment_history[-sentiment_points:]])
        rules_text = rules_text.strip() if rules_text else ""
        exo_block = _format_context_block("Official / Exogenous Context", exogenous_summary)
        horizons = key_horizons or [1, 10, 20, 30]
        horizons = [int(h) for h in horizons if 1 <= int(h) <= int(pred_len)]
        frozen = sorted({int(h) for h in (frozen_horizons or []) if int(h) in horizons})
        anchor_lines = []
        for h in horizons:
            base_price = float(tsm_forecast[h - 1])
            if h in frozen:
                anchor_lines.append(
                    f'- "h{h}": MUST equal {base_price:.2f} exactly (frozen to the base model forecast).'
                )
            else:
                lower = base_price * (1.0 - max_adjustment_pct / 100.0)
                upper = base_price * (1.0 + max_adjustment_pct / 100.0)
                anchor_lines.append(
                    f'- "h{h}": absolute price for day {h}, keep within [{lower:.2f}, {upper:.2f}] around the base forecast {base_price:.2f}.'
                )
        anchor_bullets = "\n".join(anchor_lines)
        horizon_keys = ", ".join([f"\"h{h}\"" for h in horizons])
        example_items = []
        for idx, h in enumerate(horizons):
            base_price = float(tsm_forecast[h - 1])
            if h in frozen:
                example_value = base_price
            else:
                direction = 1.0 if idx % 2 == 0 else -1.0
                example_value = base_price * (1.0 + direction * min(max_adjustment_pct, 1.0) / 200.0)
            example_items.append(f"\"h{h}\": {example_value:.2f}")
        example_json = ", ".join(example_items)
        sentiment_guidance = (
            "- Treat sentiment as secondary evidence only. Use it to modulate a correction implied by price action; do not create a correction from sentiment alone.\n"
            "- If price action and sentiment conflict, keep the anchor close to the base model forecast."
            if sentiment_secondary
            else "- Use sentiment and price action jointly when setting anchors."
        )

        prompt = f"""Refine the model forecast using the provided rules by proposing small absolute anchor prices at key horizons only.

## Context
Current price: {history[-1]:.2f} {currency}
Recent prices: [{history_str}]
Sentiment history (+1 positive, 0 neutral, -1 negative): [{sent_str}]
{exo_block}

## Base Model Forecast
[{forecast_str}]

## Correction Rules
{rules_text}

## Task
Provide anchor prices for these key horizons only:
{anchor_bullets}

Constraints:
- Each value is an absolute forecast price, not a delta or percentage.
- Keep anchors close to the base model forecast.
- Intermediate days will be interpolated automatically.
- Do not rewrite the full 30-day path.
{sentiment_guidance}

Return ONLY one JSON object:
{{"anchors": {{{horizon_keys}}}}}

Example:
{{"anchors": {{{example_json}}}}}

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
        "CoT-RF-REFLECT-STRICTJSON": CoTRFReflectionStrictJSONTemplate(),
        "CoT-RF-HDELTA-REFLECT-STRICTJSON": CoTRFHorizonDeltaReflectionStrictJSONTemplate(),
        "CoT-RF-HDELTA-REFLECT-STRUCTURED-STRICTJSON": CoTRFHorizonDeltaStructuredReflectionStrictJSONTemplate(),
        "CoT-RF-APPLY": CoTRFApplyTemplate(),
        "CoT-RF-HDELTA-APPLY": CoTRFHorizonDeltaApplyTemplate(),
        "CoT-SENT-RF-REFLECT": CoTSentRFReflectionTemplate(),
        "CoT-SENT-RF-REFLECT-STRICTJSON": CoTSentRFReflectionStrictJSONTemplate(),
        "CoT-SENT-RF-APPLY": CoTSentRFApplyTemplate(),
        "CoT-SENT-RF-APPLY-STRICTJSON": CoTSentRFApplyStrictJSONTemplate(),
        "CoT-SENT-RF-DELTA-APPLY": CoTSentRFDeltaApplyTemplate(),
        "CoT-SENT-RF-HDELTA-APPLY": CoTSentRFHorizonDeltaApplyTemplate(),
        "CoT-SENT-RF-HPRICE-APPLY": CoTSentRFHorizonPriceApplyTemplate(),
    }
    
    if method not in templates:
        raise ValueError(f"Unknown method: {method}. Available: {list(templates.keys())}")
    
    return templates[method]
