"""
Response caching for LLM calls.

Implements hash-based caching to avoid duplicate API calls
and enable reproducibility.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


class ResponseCache:
    """
    Hash-based cache for LLM responses.
    
    Caches responses to disk based on hash of (prompt, model, temperature).
    """
    
    def __init__(self, cache_dir: Union[str, Path]):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache
        self._memory_cache: Dict[str, Dict] = {}
        
        # Stats
        self.hits = 0
        self.misses = 0
    
    def _compute_hash(
        self,
        prompt: str,
        model: str,
        temperature: float,
        context: Optional[Dict] = None,
    ) -> str:
        """Compute hash key for cache lookup."""
        payload: Dict = {
            "prompt": prompt,
            "model": model,
            "temperature": temperature,
        }
        if context:
            payload["context"] = context

        content = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def get(
        self,
        prompt: str,
        model: str,
        temperature: float,
        context: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """
        Get cached response if available.
        
        Args:
            prompt: The prompt text
            model: Model identifier
            temperature: Temperature setting
            
        Returns:
            Cached response dict or None
        """
        cache_key = self._compute_hash(prompt, model, temperature, context=context)
        
        # Check memory cache first
        if cache_key in self._memory_cache:
            self.hits += 1
            return self._memory_cache[cache_key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    response = json.load(f)
                self._memory_cache[cache_key] = response
                self.hits += 1
                return response
            except Exception as e:
                logger.warning(f"Failed to load cached response: {e}")
        
        self.misses += 1
        return None
    
    def set(
        self,
        prompt: str,
        model: str,
        temperature: float,
        response: Dict,
        context: Optional[Dict] = None,
    ) -> None:
        """
        Cache a response.
        
        Args:
            prompt: The prompt text
            model: Model identifier
            temperature: Temperature setting
            response: Response to cache
        """
        cache_key = self._compute_hash(prompt, model, temperature, context=context)
        
        # Store in memory
        self._memory_cache[cache_key] = response
        
        # Store on disk
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(response, f)
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0,
            "memory_size": len(self._memory_cache),
            "disk_size": len(list(self.cache_dir.glob("*.json")))
        }
    
    def clear(self) -> None:
        """Clear all caches."""
        self._memory_cache.clear()
        for f in self.cache_dir.glob("*.json"):
            f.unlink()
        self.hits = 0
        self.misses = 0


class LogStore:
    """
    Store for LLM interaction logs.
    
    Logs all prompts and responses for analysis.
    """
    
    def __init__(self, log_dir: Union[str, Path]):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.log_dir / "llm_calls.jsonl"
        self.call_count = 0
    
    def log_call(
        self,
        prompt: str,
        response: Dict,
        model: str,
        temperature: float,
        method: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Log an LLM call.
        
        Args:
            prompt: The prompt text
            response: The response dict
            model: Model identifier
            temperature: Temperature setting
            method: Prompt method (DP, CoT, etc.)
            metadata: Additional metadata
        """
        self.call_count += 1
        
        log_entry = {
            "call_id": self.call_count,
            "method": method,
            "model": model,
            "temperature": temperature,
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:16],
            "prompt_length": len(prompt),
            "response": response,
            "metadata": metadata or {}
        }
        
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.warning(f"Failed to log LLM call: {e}")
    
    def get_logs(self) -> list:
        """Load all logged calls."""
        logs = []
        if self.log_file.exists():
            with open(self.log_file, "r") as f:
                for line in f:
                    logs.append(json.loads(line))
        return logs
