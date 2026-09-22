"""Configuration management for EU ETS forecasting project."""

import os
import yaml
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


def get_git_hash() -> Optional[str]:
    """Get current git commit hash for reproducibility."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except Exception:
        pass
    return None


def generate_run_id() -> str:
    """Generate a unique run identifier."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hash_suffix = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:6]
    return f"{timestamp}_{hash_suffix}"


@dataclass
class Config:
    """Main configuration container."""
    
    # Raw config dict
    raw: Dict[str, Any] = field(default_factory=dict)
    
    # Run metadata
    run_id: str = ""
    run_dir: Path = None
    git_hash: Optional[str] = None
    
    def __post_init__(self):
        if not self.run_id:
            self.run_id = generate_run_id()
        if self.raw.get("reproducibility", {}).get("log_git_hash", True):
            self.git_hash = get_git_hash()
    
    # Convenience property accessors
    @property
    def target(self) -> Dict:
        return self.raw.get("target", {})
    
    @property
    def time_series(self) -> Dict:
        return self.raw.get("time_series", {})
    
    @property
    def split(self) -> Dict:
        return self.raw.get("split", {})
    
    @property
    def features(self) -> Dict:
        return self.raw.get("features", {})
    
    @property
    def model(self) -> Dict:
        return self.raw.get("model", {})
    
    @property
    def llm(self) -> Dict:
        return self.raw.get("llm", {})
    
    @property
    def evaluation(self) -> Dict:
        return self.raw.get("evaluation", {})
    
    @property
    def robustness(self) -> Dict:
        return self.raw.get("robustness", {})
    
    @property
    def output(self) -> Dict:
        return self.raw.get("output", {})
    
    @property
    def reproducibility(self) -> Dict:
        return self.raw.get("reproducibility", {})
    
    @property
    def compute(self) -> Dict:
        return self.raw.get("compute", {})
    
    # Key parameters
    @property
    def pred_len(self) -> int:
        return self.time_series.get("pred_len", 30)
    
    @property
    def seq_len(self) -> int:
        return self.time_series.get("seq_len", 120)
    
    @property
    def label_len(self) -> int:
        return self.time_series.get("label_len", 30)
    
    @property
    def horizons(self) -> List[int]:
        return self.time_series.get("horizons", [1, 5, 20, 30])
    
    @property
    def seed(self) -> int:
        return self.reproducibility.get("seed", 42)
    
    @property
    def llm_enabled(self) -> bool:
        """Check if LLM refinement is enabled."""
        return bool(self.llm.get("methods", []))
    
    @property
    def llm_provider(self) -> str:
        return self.llm.get("provider", "openai")
    
    @property
    def llm_model(self) -> str:
        return self.llm.get("model", "gpt-5.2")
    
    def get_device(self) -> str:
        """Determine compute device."""
        device = self.compute.get("device", "auto")
        if device == "auto":
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device
    
    def setup_run_dir(self, base_dir: Optional[Path] = None) -> Path:
        """Create and return the run directory."""
        if base_dir is None:
            base_dir = Path(__file__).parent.parent.parent / self.output.get("runs_dir", "runs")
        
        self.run_dir = base_dir / self.run_id
        
        # Create subdirectories
        subdirs = [
            "data/raw_cache",
            "data/standardized",
            "models",
            "predictions",
            "results/robustness",
            "llm/logs",
            "paper_snapshot"
        ]
        
        for subdir in subdirs:
            (self.run_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        return self.run_dir
    
    def save(self, path: Optional[Path] = None) -> Path:
        """Save the resolved configuration to a file."""
        if path is None:
            if self.run_dir is None:
                self.setup_run_dir()
            path = self.run_dir / "config_resolved.yaml"
        
        # Add metadata
        config_with_meta = {
            "run_id": self.run_id,
            "git_hash": self.git_hash,
            "timestamp": datetime.now().isoformat(),
            **self.raw
        }
        
        with open(path, "w") as f:
            yaml.dump(config_with_meta, f, default_flow_style=False, sort_keys=False)
        
        return path
    
    def save_reproducibility_info(self) -> Path:
        """Save reproducibility information."""
        if self.run_dir is None:
            self.setup_run_dir()
        
        path = self.run_dir / "reproducibility.md"
        
        content = f"""# Reproducibility Information

## Run Metadata
- **Run ID**: {self.run_id}
- **Timestamp**: {datetime.now().isoformat()}
- **Git Hash**: {self.git_hash or "N/A"}

## Configuration
- **Random Seed**: {self.seed}
- **Deterministic Mode**: {self.reproducibility.get("deterministic", True)}

## Environment
- **Python Version**: See requirements.txt
- **Key Dependencies**: torch, pytorch-forecasting, openai

## Data Splits
- **Train End**: {self.split.get("train_end", "N/A")}
- **Val End**: {self.split.get("val_end", "N/A")}
- **Test End**: {self.split.get("test_end", "N/A")}

## Model Configuration
- **TSM Type**: {self.model.get("tsm_type", "autoformer")}
- **Prediction Length**: {self.pred_len}
- **Sequence Length**: {self.seq_len}
"""
        
        with open(path, "w") as f:
            f.write(content)
        
        return path


def load_config(config_path: Optional[str] = None, overrides: Optional[Dict] = None) -> Config:
    """Load configuration from YAML file with optional overrides."""
    
    if config_path is None:
        config_path = Path(__file__).parent / "default.yaml"
    else:
        config_path = Path(config_path)
    
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f) or {}

    # Allow loading from a resolved config (which includes run metadata).
    # If present, use `run_id` to re-open the same run directory (useful for
    # resuming long experiments) and strip metadata keys from the raw config.
    run_id = raw_config.pop("run_id", None)
    raw_config.pop("git_hash", None)
    raw_config.pop("timestamp", None)
    
    # Apply overrides
    if overrides:
        raw_config = _deep_update(raw_config, overrides)

    raw_config = enforce_split_test_end(raw_config)
    
    return Config(raw=raw_config, run_id=str(run_id) if run_id else "")


def _deep_update(base: Dict, updates: Dict) -> Dict:
    """Deep update a nested dictionary."""
    result = base.copy()
    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def enforce_split_test_end(raw_config: Dict[str, Any]) -> Dict[str, Any]:
    """Cap target.max_date at split.test_end so evaluation never runs past the configured test boundary."""
    split_cfg = raw_config.get("split", {}) or {}
    test_end = split_cfg.get("test_end")
    if not test_end:
        return raw_config

    try:
        test_end_date = datetime.fromisoformat(str(test_end)).date()
    except ValueError:
        return raw_config

    result = raw_config.copy()
    target_cfg = dict(result.get("target", {}) or {})
    existing = target_cfg.get("max_date")

    if existing:
        try:
            existing_date = datetime.fromisoformat(str(existing)).date()
        except ValueError:
            existing_date = None
        if existing_date is not None and existing_date <= test_end_date:
            return result

    target_cfg["max_date"] = test_end_date.isoformat()
    result["target"] = target_cfg
    return result
