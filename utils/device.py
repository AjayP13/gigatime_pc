from __future__ import annotations

from typing import Tuple

import torch


def auto_detect_device() -> torch.device:
    """
    Prefer CUDA (GPU) if available, otherwise Apple MPS on macOS,
    otherwise CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda", 0)
    # MPS (Apple Silicon)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def select_dtype_for_device(device: torch.device) -> torch.dtype:
    """
    Choose dtype based on device capabilities:
    - CUDA: bfloat16 if supported, otherwise float16
    - MPS: float16
    - CPU: float32 (safe default for widest op coverage)
    """
    if device.type == "cuda":
        # torch.cuda.is_bf16_supported() covers Ampere+ (or newer) GPUs
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        # As of now, MPS favors float16 for mixed-precision
        return torch.float16
    return torch.float32


def resolve_device(device_arg: str | None) -> Tuple[torch.device, torch.dtype]:
    """
    Resolve device and dtype from user arg; 'auto' or None triggers detection.
    - If device_arg is 'cpu'/'cuda'/'mps', pick that (cuda:0 default)
    - Then select dtype via select_dtype_for_device
    """
    if device_arg in (None, "", "auto"):
        device = auto_detect_device()
    else:
        if device_arg == "cuda":
            device = torch.device("cuda", 0)
        else:
            device = torch.device(device_arg)
    dtype = select_dtype_for_device(device)
    return device, dtype
