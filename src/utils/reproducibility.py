"""Reproducibility utilities."""

import random
import numpy as np
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_git_hash() -> Optional[str]:
    """Get the current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def get_environment_info() -> Dict:
    """
    Get information about the computing environment.
    
    Returns:
        Dictionary with environment information
    """
    info = {
        "python_version": sys.version,
        "platform": sys.platform,
        "git_hash": get_git_hash(),
    }
    
    # Package versions
    packages = ["numpy", "pandas", "torch", "sklearn", "openai"]
    for pkg in packages:
        try:
            module = __import__(pkg)
            info[f"{pkg}_version"] = getattr(module, "__version__", "unknown")
        except ImportError:
            info[f"{pkg}_version"] = "not installed"
    
    # PyTorch device info
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
        info["mps_available"] = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except ImportError:
        pass
    
    return info


def save_environment_info(save_path: Path):
    """Save environment information to a file."""
    info = get_environment_info()
    
    with open(save_path, "w") as f:
        f.write("# Environment Information\n\n")
        for key, value in info.items():
            f.write(f"- **{key}**: {value}\n")
