"""Shared utilities for both pipelines."""
import os
import random
import re
import numpy as np
import torch
from pathlib import Path
from typing import Iterable


def set_seed(seed: int):
    """Set deterministic seeds across libraries."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """Return torch device, prefer CUDA if available."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device selected: {device}")
    if torch.cuda.is_available():
        print(f"  CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA device count: {torch.cuda.device_count()}")
    return device


def sanitize_filename(text: str) -> str:
    """Create a safe file name from arbitrary text."""
    text = re.sub(r"[^a-zA-Z0-9_-]", "_", text)
    return text[:80]


def make_dirs(paths: Iterable[str]):
    """Create directories if they do not exist."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def get_gpu_memory_info() -> str:
    """Return a short GPU memory summary."""
    if not torch.cuda.is_available():
        return "No CUDA device"
    allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)
    return f"Allocated: {allocated:.1f} MB | Reserved: {reserved:.1f} MB"
