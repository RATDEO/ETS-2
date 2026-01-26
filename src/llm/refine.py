"""
LLM refinement for time series forecasts.

Implements the LLM-based forecast refinement layer that takes
TSM predictions and historical context to produce refined forecasts.
"""

import numpy as np
import pandas as pd
import json
import re
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging

from .prompts import get_template, PromptTemplate
from .cache import ResponseCache, LogStore

logger = logging.getLogger(__name__)


def parse_json_forecast(response_text: str, expected_len: int = 30) -> Optional[np.ndarray]:
    """
    Parse JSON forecast from LLM response.
    
    Handles various response formats and extracts the yhat array.
    
    Args:
        response_text: Raw LLM response
        expected_len: Expected length of forecast
        
    Returns:
        Numpy array of predictions or None if parsing fails
    """
    # Try to extract JSON from response
    # First, try to find JSON object
    json_patterns = [
        r'\{[^{}]*"yhat"\s*:\s*\[[^\]]+\][^{}]*\}',  # Full JSON object
        r'"yhat"\s*:\s*\[([^\]]+)\]',  # Just the array
        r'\[[\d\.,\s]+\]',  # Any array of numbers
    ]
    
    for pattern in json_patterns:
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            try:
                text = match.group(0)
                
                # Try to parse as full JSON
                if text.startswith("{"):
                    data = json.loads(text)
                    if "yhat" in data:
                        values = data["yhat"]
                        if len(values) == expected_len:
                            return np.array(values, dtype=float)
                
                # Try to parse as array
                if text.startswith("["):
                    values = json.loads(text)
                    if len(values) == expected_len:
                        return np.array(values, dtype=float)
                
                # Try to extract just the numbers
                if "yhat" in text:
                    numbers = re.findall(r'[\d.]+', text)
                    if len(numbers) >= expected_len:
                        values = [float(n) for n in numbers[:expected_len]]
                        return np.array(values)
                        
            except (json.JSONDecodeError, ValueError) as e:
                logger.debug(f"JSON parse attempt failed: {e}")
                continue
    
    # Last resort: find any sequence of numbers
    numbers = re.findall(r'\d+\.?\d*', response_text)
    if len(numbers) >= expected_len:
        try:
            values = [float(n) for n in numbers[:expected_len]]
            return np.array(values)
        except ValueError:
            pass
    
    logger.warning(f"Failed to parse forecast from response: {response_text[:200]}...")
    return None


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
    
    def _get_client(self):
        """Get or create API client."""
        if self._client is not None:
            return self._client
        
        if self.provider == "openai":
            try:
                from openai import OpenAI
                self._client = OpenAI()
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
            cached = self.cache.get(prompt, self.model, self.temperature)
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
            content = response.choices[0].message.content
        
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
            self.cache.set(prompt, self.model, self.temperature, {"content": content})
        
        return content
    
    def refine(
        self,
        method: str,
        history: np.ndarray,
        dates: List[str],
        tsm_forecast: Optional[np.ndarray] = None,
        pred_len: int = 30,
        exogenous_summary: Optional[Dict] = None
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
        
        system_message = template.get_system_message()
        
        # Try with retries
        forecast = None
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                response = self._call_llm(prompt, system_message)
                forecast = parse_json_forecast(response, expected_len=pred_len)
                
                if forecast is not None:
                    break
                
                # If parsing failed, try with a stricter prompt
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
            "attempts": attempt + 1
        }
        
        if last_error:
            metadata["error"] = last_error
        
        # Log the call
        if self.log_store:
            self.log_store.log_call(
                prompt=prompt,
                response={"content": response if 'response' in dir() else None},
                model=self.model,
                temperature=self.temperature,
                method=method,
                metadata=metadata
            )
        
        return forecast, metadata
    
    def refine_batch(
        self,
        method: str,
        histories: np.ndarray,
        date_arrays: List[List[str]],
        tsm_forecasts: Optional[np.ndarray] = None,
        pred_len: int = 30
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
            
            forecast, metadata = self.refine(
                method=method,
                history=histories[i],
                dates=date_arrays[i],
                tsm_forecast=tsm_forecast,
                pred_len=pred_len
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
