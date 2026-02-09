"""
LLM refinement for time series forecasts.

Implements the LLM-based forecast refinement layer that takes
TSM predictions and historical context to produce refined forecasts.
"""

import numpy as np
import pandas as pd
import json
import re
from typing import Dict, List, Optional, Tuple, Union, Sequence
from pathlib import Path
import logging

from .prompts import get_template, PromptTemplate
from .cache import ResponseCache, LogStore

logger = logging.getLogger(__name__)


def parse_json_array(
    response_text: str,
    expected_len: int = 30,
    keys: Sequence[str] = ("yhat",)
) -> Optional[np.ndarray]:
    """
    Parse a numeric array from LLM response JSON.
    
    Args:
        response_text: Raw LLM response
        expected_len: Expected length of array
        keys: JSON keys to search for
        
    Returns:
        Numpy array or None if parsing fails
    """
    # Normalize common formatting artifacts
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", cleaned)
        cleaned = re.sub(r"\n```$", "", cleaned)
    cleaned = cleaned.replace("...", "")
    cleaned = re.sub(r",\s*([\]}])", r"\1", cleaned)

    # Try to extract JSON from response
    # First, try to find JSON object
    key_pattern = "|".join(re.escape(k) for k in keys)
    json_patterns = [
        rf'\{{[^{{}}]*"(?:{key_pattern})"\s*:\s*\[[^\]]+\][^{{}}]*\}}',  # Full JSON object
        rf'"(?:{key_pattern})"\s*:\s*\[([^\]]+)\]',  # Just the array
        r'\[[\d\.,\s]+\]',  # Any array of numbers
    ]
    
    for pattern in json_patterns:
        match = re.search(pattern, cleaned, re.DOTALL)
        if match:
            try:
                text = match.group(0)
                
                # Try to parse as full JSON
                if text.startswith("{"):
                    text = re.sub(r",\s*([\]}])", r"\1", text)
                    data = json.loads(text)
                    for key in keys:
                        if key in data:
                            values = data[key]
                            if len(values) == expected_len:
                                return np.array(values, dtype=float)
                            if len(values) == expected_len - 1:
                                values = list(values) + [values[-1]]
                                return np.array(values, dtype=float)
                
                # Try to parse as array
                if text.startswith("["):
                    text = re.sub(r",\s*([\]}])", r"\1", text)
                    values = json.loads(text)
                    if len(values) == expected_len:
                        return np.array(values, dtype=float)
                    if len(values) == expected_len - 1:
                        values = list(values) + [values[-1]]
                        return np.array(values, dtype=float)
                
                # Try to extract just the numbers
                for key in keys:
                    if key in text:
                        numbers = re.findall(r'[\d.]+', text)
                        if len(numbers) >= expected_len:
                            values = [float(n) for n in numbers[:expected_len]]
                            return np.array(values)
                        if len(numbers) == expected_len - 1:
                            values = [float(n) for n in numbers]
                            values.append(values[-1])
                            return np.array(values)
                        
            except (json.JSONDecodeError, ValueError) as e:
                logger.debug(f"JSON parse attempt failed: {e}")
                continue
    
    # Last resort: find any sequence of numbers
    numbers = re.findall(r'\d+\.?\d*', cleaned)
    if len(numbers) >= expected_len:
        try:
            values = [float(n) for n in numbers[:expected_len]]
            return np.array(values)
        except ValueError:
            pass
    if len(numbers) == expected_len - 1:
        try:
            values = [float(n) for n in numbers]
            values.append(values[-1])
            return np.array(values)
        except ValueError:
            pass
    
    logger.warning(f"Failed to parse forecast from response: {response_text[:200]}...")
    return None


def parse_json_forecast(response_text: str, expected_len: int = 30) -> Optional[np.ndarray]:
    """Parse JSON forecast (yhat) from LLM response."""
    return parse_json_array(response_text, expected_len, keys=("yhat",))


def returns_to_prices(returns: np.ndarray, last_price: float) -> np.ndarray:
    """Convert log returns to price path."""
    last_price = float(last_price)
    return last_price * np.exp(np.cumsum(returns, axis=0))


class LLMRefiner:
    """
    LLM-based forecast refiner.
    
    Uses large language models to refine time series forecasts
    based on historical context and domain knowledge.
    """
    
    def __init__(
        self,
        config: Dict,
        cache_dir: Optional[Path] = None,
        log_dir: Optional[Path] = None
    ):
        """
        Initialize the LLM refiner.
        
        Args:
            config: LLM configuration dict
            cache_dir: Directory for response caching
            log_dir: Directory for logging
        """
        self.config = config
        
        self.provider = config.get("provider", "openai")
        self.model = config.get("model", "gpt-5.2")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 1000)
        self.max_retries = config.get("max_retries", 3)
        self.timeout_seconds = float(config.get("timeout_seconds", 120.0))
        self.client_max_retries = int(config.get("client_max_retries", 2))
        self.blend_config = config.get("blend", {})
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url")
        
        # Initialize cache
        if cache_dir and config.get("cache_enabled", True):
            self.cache = ResponseCache(cache_dir)
        else:
            self.cache = None
        
        # Initialize logger
        if log_dir:
            self.log_store = LogStore(log_dir)
        else:
            self.log_store = None
        
        # API client (lazy initialization)
        self._client = None
    
    def _blend_forecast(self, base: np.ndarray, refined: np.ndarray) -> np.ndarray:
        """Blend refined forecast with base using horizon-weighted weights."""
        mode = self.blend_config.get("mode", "horizon_weighted")
        if mode == "fixed":
            weight = float(self.blend_config.get("weight", 0.5))
            return base * (1 - weight) + refined * weight
        if mode != "horizon_weighted":
            return refined
        
        min_weight = float(self.blend_config.get("min_weight", 0.1))
        max_weight = float(self.blend_config.get("max_weight", 0.6))
        power = float(self.blend_config.get("power", 1.0))
        
        ramp = np.linspace(min_weight, max_weight, base.shape[-1])
        if power != 1.0:
            ramp = np.power(ramp, power)
        return base * (1 - ramp) + refined * ramp
    
    def _get_client(self):
        """Get or create API client."""
        if self._client is not None:
            return self._client
        
        if self.provider == "openai":
            try:
                import os
                from openai import OpenAI
                api_key = self.api_key or os.getenv("OPENAI_API_KEY")
                base_url = self.base_url or os.getenv("OPENAI_BASE_URL")
                kwargs = {}
                if api_key:
                    kwargs["api_key"] = api_key
                if base_url:
                    kwargs["base_url"] = base_url
                kwargs["timeout"] = self.timeout_seconds
                kwargs["max_retries"] = self.client_max_retries
                self._client = OpenAI(**kwargs)
            except ImportError:
                raise ImportError("openai package required for OpenAI provider")
        elif self.provider == "anthropic":
            try:
                import anthropic
                self._client = anthropic.Anthropic()
            except ImportError:
                raise ImportError("anthropic package required for Anthropic provider")
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
        
        return self._client
    
    def _call_llm(
        self,
        prompt: str,
        system_message: str
    ) -> str:
        """
        Call the LLM API.
        
        Args:
            prompt: User prompt
            system_message: System message
            
        Returns:
            Response text
        """
        # Check cache first
        if self.cache:
            import os

            cache_context = {
                "provider": self.provider,
                "base_url": self.base_url or os.getenv("OPENAI_BASE_URL") or "",
                "system_message": system_message,
                "max_tokens": self.max_tokens,
            }
            cached = self.cache.get(
                prompt, self.model, self.temperature, context=cache_context
            )
            if cached:
                return cached.get("content", "")
        
        client = self._get_client()
        
        if self.provider == "openai":
            request = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.temperature
            }
            if self.model.startswith("gpt-5"):
                request["max_completion_tokens"] = self.max_tokens
            else:
                request["max_tokens"] = self.max_tokens
            response = client.chat.completions.create(**request)
            message = response.choices[0].message
            content = message.content
            if not content:
                content = getattr(message, "reasoning_content", None) or ""
        
        elif self.provider == "anthropic":
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_message,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            content = response.content[0].text
        
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
        
        # Cache response
        if self.cache:
            self.cache.set(
                prompt,
                self.model,
                self.temperature,
                {"content": content},
                context=cache_context,
            )
        
        return content

    def _call_llm_messages(
        self,
        messages: List[Dict[str, str]],
        system_message: str = "",
    ) -> str:
        """Call the LLM API with an explicit multi-message chat history."""
        payload = {"messages": messages}
        if system_message:
            payload["system_message"] = system_message
        prompt_key = json.dumps(payload, sort_keys=True, ensure_ascii=False)

        # Cache
        if self.cache:
            import os

            cache_context = {
                "provider": self.provider,
                "base_url": self.base_url or os.getenv("OPENAI_BASE_URL") or "",
                "system_message": system_message,
                "max_tokens": self.max_tokens,
            }
            cached = self.cache.get(
                prompt_key, self.model, self.temperature, context=cache_context
            )
            if cached:
                return cached.get("content", "")

        client = self._get_client()

        if self.provider == "openai":
            request_messages: List[Dict[str, str]] = []
            if system_message:
                request_messages.append({"role": "system", "content": system_message})
            request_messages.extend(messages)

            request = {
                "model": self.model,
                "messages": request_messages,
                "temperature": self.temperature,
            }
            if self.model.startswith("gpt-5"):
                request["max_completion_tokens"] = self.max_tokens
            else:
                request["max_tokens"] = self.max_tokens

            response = client.chat.completions.create(**request)
            message = response.choices[0].message
            content = message.content
            if not content:
                content = getattr(message, "reasoning_content", None) or ""

        elif self.provider == "anthropic":
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_message,
                messages=messages,
            )
            content = response.content[0].text
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        if self.cache:
            self.cache.set(
                prompt_key,
                self.model,
                self.temperature,
                {"content": content},
                context=cache_context,
            )

        return content

    def refine(
        self,
        method: str,
        history: np.ndarray,
        dates: List[str],
        tsm_forecast: Optional[np.ndarray] = None,
        pred_len: int = 30,
        exogenous_summary: Optional[Dict] = None,
        price_base: Optional[float] = None,
        teaching_examples: Optional[List[Dict]] = None,
        sentiment_history: Optional[np.ndarray] = None
    ) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Generate or refine forecast using LLM.
        
        Args:
            method: Prompt method (DP, CoT, CoT-RF, TSM+LLM)
            history: Historical price values
            dates: Historical dates
            tsm_forecast: TSM model forecast (required for TSM+LLM)
            pred_len: Prediction length
            exogenous_summary: Summary of exogenous variables
            
        Returns:
            Tuple of (forecast array, metadata dict)
        """
        if method == "TSM+NEWS-DRIFT":
            if tsm_forecast is None:
                raise ValueError("TSM forecast required for TSM+NEWS-DRIFT method")
            if sentiment_history is None:
                raise ValueError("Sentiment history required for TSM+NEWS-DRIFT method")

            drift_cfg = self.config.get("news_drift", {})
            horizon = int(drift_cfg.get("horizons", 5))
            sentiment_window = int(drift_cfg.get("sentiment_window", 1))
            min_abs = float(drift_cfg.get("min_abs", 0.0))
            scale = float(drift_cfg.get("scale", 1.0))
            pos_mult = float(drift_cfg.get("pos_mult", 1.0))
            neg_mult = float(drift_cfg.get("neg_mult", 1.5))
            vol_window = int(drift_cfg.get("vol_window", 20))
            max_pct = float(drift_cfg.get("max_adjust_pct", 0.03))
            decay_mode = str(drift_cfg.get("decay", "linear")).lower()
            half_life = float(drift_cfg.get("half_life", 3.0))
            calib_cfg = drift_cfg.get("calibration", {})

            if sentiment_window > 0:
                recent_sent = sentiment_history[-sentiment_window:]
            else:
                recent_sent = sentiment_history
            score = float(np.mean(recent_sent)) if len(recent_sent) else 0.0

            if abs(score) < min_abs or horizon <= 0:
                return tsm_forecast, {
                    "method": method,
                    "model": self.model,
                    "temperature": self.temperature,
                    "success": True,
                    "applied": False,
                    "sentiment_score": score,
                }

            if len(history) >= vol_window + 1:
                history_slice = history[-(vol_window + 1):]
            else:
                history_slice = history
            safe_hist = np.clip(history_slice, 1e-8, None)
            returns = np.diff(np.log(safe_hist))
            return_std = float(np.std(returns)) if len(returns) else 0.0
            if return_std < 1e-6:
                return_std = 0.01

            weight = pos_mult if score >= 0 else neg_mult

            if calib_cfg.get("enabled") and "beta" in calib_cfg:
                alpha = float(calib_cfg.get("alpha", 0.0))
                beta = float(calib_cfg.get("beta", 0.0))
                calib_scale = float(calib_cfg.get("scale", 1.0))
                base_adj = (alpha + beta * score) * calib_scale * weight
            else:
                base_adj = score * scale * weight * return_std
            horizon = min(horizon, pred_len, len(tsm_forecast))

            if decay_mode == "exp":
                decay = np.exp(-np.arange(horizon) / max(half_life, 1e-6))
            else:
                decay = np.linspace(1.0, 0.0, horizon, endpoint=True)

            adjusted = np.array(tsm_forecast, copy=True)
            for idx in range(horizon):
                factor = float(np.exp(base_adj * decay[idx]))
                if max_pct > 0.0:
                    factor = max(1.0 - max_pct, min(1.0 + max_pct, factor))
                adjusted[idx] = tsm_forecast[idx] * factor
            adjusted = np.clip(adjusted, 0.01, None)

            return adjusted, {
                "method": method,
                "model": self.model,
                "temperature": self.temperature,
                "success": True,
                "applied": True,
                "sentiment_score": score,
                "return_std": return_std,
                "base_adjustment": base_adj,
                "horizons": horizon,
                "calibrated": bool(calib_cfg.get("enabled") and "beta" in calib_cfg),
            }

        if method == "NEWS-SENTIMENT-ONLY":
            if sentiment_history is None:
                raise ValueError("Sentiment history required for NEWS-SENTIMENT-ONLY method")
            if price_base is None:
                if len(history):
                    price_base = float(history[-1])
                else:
                    raise ValueError("price_base required for NEWS-SENTIMENT-ONLY method")

            signal_cfg = self.config.get("news_signal")
            if signal_cfg is None:
                signal_cfg = self.config.get("news_drift", {})

            horizon = int(signal_cfg.get("horizons", pred_len))
            sentiment_window = int(signal_cfg.get("sentiment_window", 1))
            min_abs = float(signal_cfg.get("min_abs", 0.0))
            scale = float(signal_cfg.get("scale", 0.5))
            pos_mult = float(signal_cfg.get("pos_mult", 1.0))
            neg_mult = float(signal_cfg.get("neg_mult", 1.0))
            vol_window = int(signal_cfg.get("vol_window", 20))
            max_pct = float(signal_cfg.get("max_adjust_pct", 0.03))
            decay_mode = str(signal_cfg.get("decay", "linear")).lower()
            half_life = float(signal_cfg.get("half_life", 3.0))

            if sentiment_window > 0:
                recent_sent = sentiment_history[-sentiment_window:]
            else:
                recent_sent = sentiment_history
            score = float(np.mean(recent_sent)) if len(recent_sent) else 0.0

            horizon = min(horizon, pred_len)
            if abs(score) < min_abs or horizon <= 0:
                forecast = np.full(pred_len, float(price_base))
                return forecast, {
                    "method": method,
                    "model": self.model,
                    "temperature": self.temperature,
                    "success": True,
                    "applied": False,
                    "sentiment_score": score,
                }

            if len(history) >= vol_window + 1:
                history_slice = history[-(vol_window + 1):]
            else:
                history_slice = history
            safe_hist = np.clip(history_slice, 1e-8, None)
            returns = np.diff(np.log(safe_hist))
            return_std = float(np.std(returns)) if len(returns) else 0.0
            if return_std < 1e-6:
                return_std = 0.01

            weight = pos_mult if score >= 0 else neg_mult
            base_adj = score * scale * weight * return_std

            if decay_mode == "exp":
                decay = np.exp(-np.arange(horizon) / max(half_life, 1e-6))
            else:
                decay = np.linspace(1.0, 0.0, horizon, endpoint=True)

            forecast = np.zeros(pred_len)
            current = float(price_base)
            for idx in range(pred_len):
                step = base_adj * decay[idx] if idx < horizon else 0.0
                factor = float(np.exp(step))
                if max_pct > 0.0:
                    factor = max(1.0 - max_pct, min(1.0 + max_pct, factor))
                current *= factor
                forecast[idx] = max(current, 0.01)

            return forecast, {
                "method": method,
                "model": self.model,
                "temperature": self.temperature,
                "success": True,
                "applied": True,
                "sentiment_score": score,
                "return_std": return_std,
                "base_adjustment": base_adj,
                "horizons": horizon,
            }

        template = None
        if method not in ("TSM+LLM-COT-RF", "TSM+LLM-COT-SENT-RF", "TSM+LLM-COT-SENT-RF-DELTA"):
            template = get_template(method)

        # Prepare prompt
        if method == "TSM+LLM":
            if tsm_forecast is None:
                raise ValueError("TSM forecast required for TSM+LLM method")
            prompt = template.format(
                history=history,
                dates=dates,
                tsm_forecast=tsm_forecast,
                pred_len=pred_len,
                exogenous_summary=exogenous_summary
            )
        elif method == "TSM+LLM-NORM-DELTA":
            if tsm_forecast is None:
                raise ValueError("TSM forecast required for TSM+LLM-NORM-DELTA method")
            norm_cfg = self.config.get("norm_delta", {})
            window = int(norm_cfg.get("history_window", len(history)))
            max_delta = float(norm_cfg.get("max_delta", 0.5))
            hist_window = history[-window:] if len(history) >= window else history
            mean = float(np.mean(hist_window)) if len(hist_window) else 0.0
            std = float(np.std(hist_window)) if len(hist_window) else 1.0
            if std < 1e-6:
                std = 1.0
            norm_history = (history - mean) / std
            norm_forecast = (tsm_forecast - mean) / std
            prompt = template.format(
                history=norm_history,
                dates=dates,
                tsm_forecast=norm_forecast,
                pred_len=pred_len,
                currency="EUR",
                history_mean=mean,
                history_std=std,
                max_delta=max_delta
            )
        elif method == "TSM+LLM-COT-SENT":
            if tsm_forecast is None:
                raise ValueError("TSM forecast required for TSM+LLM-COT-SENT method")
            if sentiment_history is None:
                raise ValueError("Sentiment history required for TSM+LLM-COT-SENT method")
            prompt = template.format(
                history=history,
                dates=dates,
                tsm_forecast=tsm_forecast,
                sentiment_history=sentiment_history,
                pred_len=pred_len
            )
        elif method == "TSM+LLM-COT-RF":
            if tsm_forecast is None:
                raise ValueError("TSM forecast required for CoT-RF method")
            if not teaching_examples:
                raise ValueError("Teaching examples required for CoT-RF method")
            reflection_template = get_template("CoT-RF-REFLECT")
            apply_template = get_template("CoT-RF-APPLY")
            prompt = reflection_template.format(
                examples=teaching_examples,
                pred_len=pred_len
            )
        elif method == "TSM+LLM-COT-SENT-RF":
            if tsm_forecast is None:
                raise ValueError("TSM forecast required for CoT-SENT-RF method")
            if not teaching_examples:
                raise ValueError("Teaching examples required for CoT-SENT-RF method")
            if sentiment_history is None:
                raise ValueError("Sentiment history required for CoT-SENT-RF method")
            reflection_template = get_template("CoT-SENT-RF-REFLECT")
            apply_template = get_template("CoT-SENT-RF-APPLY")
            prompt_hist = int(self.config.get("prompt_history_points", 30))
            prompt_sent = int(self.config.get("prompt_sentiment_points", 30))
            prompt = reflection_template.format(
                examples=teaching_examples,
                pred_len=pred_len,
                history_points=prompt_hist,
                sentiment_points=prompt_sent
            )
        elif method == "TSM+LLM-COT-SENT-RF-DELTA":
            if tsm_forecast is None:
                raise ValueError("TSM forecast required for CoT-SENT-RF-DELTA method")
            if not teaching_examples:
                raise ValueError("Teaching examples required for CoT-SENT-RF-DELTA method")
            if sentiment_history is None:
                raise ValueError("Sentiment history required for CoT-SENT-RF-DELTA method")
            reflection_template = get_template("CoT-SENT-RF-REFLECT")
            apply_template = get_template("CoT-SENT-RF-DELTA-APPLY")
            prompt_hist = int(self.config.get("prompt_history_points", 30))
            prompt_sent = int(self.config.get("prompt_sentiment_points", 30))
            prompt = reflection_template.format(
                examples=teaching_examples,
                pred_len=pred_len,
                history_points=prompt_hist,
                sentiment_points=prompt_sent
            )
        elif method == "CoT-RF":
            # First get initial forecast
            initial_result, _ = self.refine(
                "CoT", history, dates, pred_len=pred_len,
                exogenous_summary=exogenous_summary
            )
            if initial_result is None:
                return None, {"error": "Initial forecast failed"}
            
            prompt = template.format(
                history=history,
                dates=dates,
                initial_forecast=initial_result,
                pred_len=pred_len
            )
        else:
            prompt = template.format(
                history=history,
                dates=dates,
                pred_len=pred_len,
                exogenous_summary=exogenous_summary
            )

        # Special handling: two-stage reflect→apply methods
        if method in ("TSM+LLM-COT-RF", "TSM+LLM-COT-SENT-RF", "TSM+LLM-COT-SENT-RF-DELTA"):
            retain_context = bool(
                (self.config.get("cot_rf", {}) or {}).get("retain_context", False)
            )

            # Stage A: reflection → rules_text (plain text)
            reflect_system = reflection_template.get_system_message()
            rules_text = None
            reflect_response = ""
            reflect_error = None
            reflect_attempts = 0
            for reflect_attempt in range(self.max_retries):
                reflect_attempts = reflect_attempt + 1
                try:
                    reflect_response = self._call_llm(prompt, reflect_system)
                    rules_text = (reflect_response or "").strip()
                    if rules_text:
                        break
                except Exception as e:
                    reflect_error = str(e)
                    logger.warning(
                        "LLM reflection call failed (attempt %d): %s",
                        reflect_attempt + 1,
                        e,
                    )

            if self.log_store:
                self.log_store.log_call(
                    prompt=prompt,
                    response={"content": reflect_response},
                    model=self.model,
                    temperature=self.temperature,
                    method=f"{method}:reflect",
                    metadata={
                        "success": bool(rules_text),
                        "attempts": reflect_attempts,
                        "error": reflect_error,
                        "retain_context": retain_context,
                        "teaching_examples": len(teaching_examples or []),
                    },
                )

            if not rules_text:
                metadata = {
                    "method": method,
                    "model": self.model,
                    "temperature": self.temperature,
                    "success": False,
                    "reflect_attempts": reflect_attempts,
                    "apply_attempts": 0,
                    "retain_context": retain_context,
                    "error": reflect_error or "Empty rules_text from reflection stage",
                }
                return None, metadata

            # Stage B: apply rules → refined forecast (JSON)
            prompt_hist = int(self.config.get("prompt_history_points", 30))
            prompt_sent = int(self.config.get("prompt_sentiment_points", 30))

            if method == "TSM+LLM-COT-RF":
                apply_prompt = apply_template.format(
                    history=history,
                    dates=dates,
                    tsm_forecast=tsm_forecast,
                    rules_text=rules_text,
                    pred_len=pred_len,
                )
            else:
                apply_kwargs = {
                    "history": history,
                    "dates": dates,
                    "tsm_forecast": tsm_forecast,
                    "sentiment_history": sentiment_history,
                    "rules_text": rules_text,
                    "pred_len": pred_len,
                    "history_points": prompt_hist,
                    "sentiment_points": prompt_sent,
                }
                if method == "TSM+LLM-COT-SENT-RF-DELTA":
                    delta_cfg = self.config.get("delta", {})
                    max_std = float(delta_cfg.get("max_delta_std", 0.5))
                    max_abs = delta_cfg.get("max_delta_abs")
                    hist_window = history[-20:] if len(history) >= 20 else history
                    hist_std = float(np.std(hist_window)) if len(hist_window) else 0.0
                    max_delta = float(max_abs) if max_abs is not None else max_std * hist_std
                    apply_kwargs["max_delta"] = max_delta

                apply_prompt = apply_template.format(**apply_kwargs)

            apply_system = apply_template.get_system_message()
            forecast = None
            last_error = None
            apply_response = ""
            apply_prompt_for_log = apply_prompt
            apply_attempts = 0

            for apply_attempt in range(self.max_retries):
                apply_attempts = apply_attempt + 1
                try:
                    if retain_context:
                        messages = [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": rules_text},
                            {"role": "user", "content": apply_prompt},
                        ]
                        apply_prompt_for_log = json.dumps(
                            {"system": apply_system, "messages": messages},
                            sort_keys=True,
                            ensure_ascii=False,
                        )
                        apply_response = self._call_llm_messages(messages, apply_system)
                    else:
                        apply_prompt_for_log = apply_prompt
                        apply_response = self._call_llm(apply_prompt, apply_system)

                    if method == "TSM+LLM-COT-SENT-RF-DELTA":
                        delta = parse_json_array(
                            apply_response,
                            expected_len=pred_len,
                            keys=("delta", "delta_price", "adjustment"),
                        )
                        if delta is not None:
                            max_delta = apply_kwargs.get("max_delta", 0.0)
                            if max_delta and max_delta > 0:
                                delta = np.clip(delta, -max_delta, max_delta)
                            refined = tsm_forecast + delta
                            forecast = self._blend_forecast(tsm_forecast, refined)
                    else:
                        forecast = parse_json_forecast(apply_response, expected_len=pred_len)

                    if forecast is not None:
                        break

                    if apply_attempt < self.max_retries - 1:
                        apply_prompt += "\n\nIMPORTANT: Respond ONLY with valid JSON. No other text."

                except Exception as e:
                    last_error = str(e)
                    logger.warning(
                        "LLM apply call failed (attempt %d): %s",
                        apply_attempt + 1,
                        e,
                    )

            metadata = {
                "method": method,
                "model": self.model,
                "temperature": self.temperature,
                "success": forecast is not None,
                "reflect_attempts": reflect_attempts,
                "apply_attempts": apply_attempts,
                "retain_context": retain_context,
                "rules_text": rules_text,
                "teaching_examples": len(teaching_examples or []),
                "teaching_dates": [ex.get("date") for ex in (teaching_examples or [])],
            }
            if last_error:
                metadata["error"] = last_error

            if self.log_store:
                self.log_store.log_call(
                    prompt=apply_prompt_for_log,
                    response={"content": apply_response},
                    model=self.model,
                    temperature=self.temperature,
                    method=f"{method}:apply",
                    metadata=metadata,
                )

            return forecast, metadata

        system_message = template.get_system_message() if template else ""

        # Try with retries (single-call methods)
        forecast = None
        last_error = None
        response = ""

        for attempt in range(self.max_retries):
            try:
                response = self._call_llm(prompt, system_message)
                if method == "TSM+LLM-NORM-DELTA":
                    norm_cfg = self.config.get("norm_delta", {})
                    max_delta = float(norm_cfg.get("max_delta", 0.5))
                    window = int(norm_cfg.get("history_window", len(history)))
                    hist_window = history[-window:] if len(history) >= window else history
                    mean = float(np.mean(hist_window)) if len(hist_window) else 0.0
                    std = float(np.std(hist_window)) if len(hist_window) else 1.0
                    if std < 1e-6:
                        std = 1.0
                    delta = parse_json_array(
                        response,
                        expected_len=pred_len,
                        keys=("delta_z", "delta"),
                    )
                    if delta is not None:
                        delta = np.clip(delta, -max_delta, max_delta)
                        norm_forecast = (tsm_forecast - mean) / std
                        refined_z = norm_forecast + delta
                        refined = refined_z * std + mean
                        forecast = self._blend_forecast(tsm_forecast, refined)
                else:
                    forecast = parse_json_forecast(response, expected_len=pred_len)

                if forecast is not None:
                    break

                if attempt < self.max_retries - 1:
                    prompt += "\n\nIMPORTANT: Respond ONLY with valid JSON. No other text."

            except Exception as e:
                last_error = str(e)
                logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}")

        metadata = {
            "method": method,
            "model": self.model,
            "temperature": self.temperature,
            "success": forecast is not None,
            "attempts": attempt + 1,
        }
        if last_error:
            metadata["error"] = last_error

        if self.log_store:
            self.log_store.log_call(
                prompt=prompt,
                response={"content": response},
                model=self.model,
                temperature=self.temperature,
                method=method,
                metadata=metadata,
            )

        return forecast, metadata
    
    def refine_batch(
        self,
        method: str,
        histories: np.ndarray,
        date_arrays: List[List[str]],
        tsm_forecasts: Optional[np.ndarray] = None,
        pred_len: int = 30,
        exogenous_summaries: Optional[List[Optional[Dict]]] = None,
        price_bases: Optional[np.ndarray] = None,
        teaching_examples: Optional[List[List[Dict]]] = None,
        sentiment_histories: Optional[List[np.ndarray]] = None
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Refine forecasts for a batch of samples.
        
        Args:
            method: Prompt method
            histories: Historical values (batch, seq_len)
            date_arrays: Dates for each sample
            tsm_forecasts: TSM forecasts (batch, pred_len)
            pred_len: Prediction length
            
        Returns:
            Tuple of (forecasts array, metadata list)
        """
        batch_size = len(histories)
        forecasts = np.zeros((batch_size, pred_len))
        metadata_list = []
        
        for i in range(batch_size):
            tsm_forecast = tsm_forecasts[i] if tsm_forecasts is not None else None
            exo_summary = exogenous_summaries[i] if exogenous_summaries is not None else None
            price_base = price_bases[i] if price_bases is not None else None
            examples = teaching_examples[i] if teaching_examples is not None else None
            sentiment_history = (
                sentiment_histories[i] if sentiment_histories is not None else None
            )
            
            forecast, metadata = self.refine(
                method=method,
                history=histories[i],
                dates=date_arrays[i],
                tsm_forecast=tsm_forecast,
                pred_len=pred_len,
                exogenous_summary=exo_summary,
                price_base=price_base,
                teaching_examples=examples,
                sentiment_history=sentiment_history
            )
            
            if forecast is not None:
                forecasts[i] = forecast
            elif tsm_forecasts is not None:
                # Fallback to TSM forecast
                forecasts[i] = tsm_forecasts[i]
                metadata["fallback"] = True
            else:
                # Fallback to persistence
                forecasts[i] = np.full(pred_len, histories[i, -1])
                metadata["fallback"] = True
            
            metadata_list.append(metadata)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{batch_size} samples")
        
        return forecasts, metadata_list


def add_noise_to_forecast(
    forecast: np.ndarray,
    noise_level: float,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Add noise to a forecast for robustness testing.
    
    Args:
        forecast: Original forecast (batch, pred_len) or (pred_len,)
        noise_level: Noise level as fraction of std
        seed: Random seed
        
    Returns:
        Noisy forecast
    """
    if seed is not None:
        np.random.seed(seed)
    
    std = np.std(forecast, axis=-1, keepdims=True)
    noise = np.random.normal(0, noise_level * std, forecast.shape)
    
    return forecast + noise
