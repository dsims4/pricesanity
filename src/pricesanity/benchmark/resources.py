"""Keep benchmark resource limits and hardware evidence explicit."""

from contextlib import contextmanager
import os
import platform
import subprocess
from typing import Any


@contextmanager
def controlled_thread_budget(worker_count: int):
    """Temporarily apply one CPU-thread budget to BLAS, OpenMP, and PyTorch."""

    if worker_count <= 0:
        raise ValueError("CPU worker count must be positive.")
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        threadpool_limits = None
    try:
        import torch
    except ImportError:
        torch = None

    previous_torch_threads = torch.get_num_threads() if torch is not None else None
    limits = threadpool_limits(limits=worker_count) if threadpool_limits else None
    try:
        if limits is not None:
            limits.__enter__()
        if torch is not None:
            torch.set_num_threads(worker_count)
        yield
    finally:
        if torch is not None and previous_torch_threads is not None:
            torch.set_num_threads(previous_torch_threads)
        if limits is not None:
            limits.__exit__(None, None, None)


def hardware_fingerprint(*, device: str, cpu_worker_count: int) -> dict[str, Any]:
    """Describe hardware sufficiently to avoid false cross-machine runtime comparisons."""

    fingerprint: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": _processor_name(),
        "logical_cpu_count": os.cpu_count(),
        "cpu_worker_count": cpu_worker_count,
        "data_loader_worker_count": 0,
        "device": device,
    }
    try:
        import psutil

        fingerprint["physical_cpu_count"] = psutil.cpu_count(logical=False)
        fingerprint["ram_bytes"] = int(psutil.virtual_memory().total)
    except ImportError:
        fingerprint["physical_cpu_count"] = None
    if hasattr(os, "sysconf"):
        try:
            fingerprint["ram_bytes"] = int(os.sysconf("SC_PAGE_SIZE")) * int(
                os.sysconf("SC_PHYS_PAGES")
            )
        except (ValueError, OSError):
            fingerprint["ram_bytes"] = None
    try:
        import torch

        fingerprint["pytorch_version"] = torch.__version__
        fingerprint["pytorch_cuda_version"] = torch.version.cuda
        fingerprint["pytorch_hip_version"] = torch.version.hip
        if device == "mps":
            fingerprint["accelerator_name"] = "Apple Metal Performance Shaders"
        elif device == "cuda" and torch.cuda.is_available():
            fingerprint["accelerator_name"] = torch.cuda.get_device_name()
            fingerprint["backend"] = "ROCm" if torch.version.hip else "CUDA"
        else:
            fingerprint["backend"] = "CPU"
    except ImportError:
        fingerprint["backend"] = "CPU"
    return fingerprint


def _processor_name() -> str | None:
    """Fill platform.processor's common macOS blank without making hardware mandatory."""

    value = platform.processor().strip()
    if value:
        return value
    if platform.system() == "Darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
                timeout=1,
            ).strip() or None
        except (OSError, subprocess.SubprocessError):
            return None
    return platform.uname().processor or None
