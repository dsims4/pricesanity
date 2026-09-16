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
        # This is a process-wide benchmark control, so always restore the caller's PyTorch
        # setting even when fitting raises and the surrounding run is being checkpointed.
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

    if platform.system() == "Linux":
        from pathlib import Path
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
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


def hardware_diagnostic() -> dict[str, Any]:
    """Describe installed build and detected devices without assuming CUDA means NVIDIA."""
    report = hardware_fingerprint(device='cpu', cpu_worker_count=1)
    report.update({'python_version': platform.python_version(),
                   'wsl_kernel': platform.release() if 'microsoft' in platform.release().lower() else None})
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        mps_available = torch.backends.mps.is_available()
        report.update({'cuda_available': cuda_available, 'mps_available': mps_available,
                       'pytorch_build_backend': 'ROCm' if torch.version.hip else 'CUDA' if torch.version.cuda else 'CPU',
                       'pytorch_build': torch.__config__.show(),
                       'torch_cpu_threads': torch.get_num_threads(),
                       'detected_accelerator': torch.cuda.get_device_name() if cuda_available else 'Apple MPS' if mps_available else None})
    except ImportError:
        report['detected_accelerator'] = None
        report['pytorch_available'] = False
    return report


def smoke_devices(*, device: str = 'auto', compare: bool = False) -> dict[str, Any]:
    """Measure tiny complete fits and verify saved inference, using synthetic inputs only."""
    import math
    import tempfile
    from pathlib import Path
    from time import perf_counter
    import numpy as np
    import torch
    from pricesanity.benchmark.registry import build_model, load_model
    from pricesanity.models.sequence import _resolve_device

    detected = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    selected = detected if device == 'auto' else device
    _resolve_device(selected)  # Explicit unavailable acceleration must fail, never fall back.
    devices = list(dict.fromkeys(['cpu', selected])) if compare else [selected]
    rng = np.random.default_rng(42)
    features = rng.normal(size=(32, 16, 4)).astype(np.float32)
    targets = np.arange(32) % 3
    rows = []
    for current_device in devices:
        for family in ('tcn', 'gru', 'transformer'):
            settings = {'epochs': 2, 'batch_size': 8}
            if family == 'tcn': settings.update(channel_width=8,layer_count=2,kernel_size=3)
            if family == 'gru': settings.update(hidden_size=8,layer_count=1)
            if family == 'transformer': settings.update(model_dimension=12,layer_count=4,attention_head_count=4,feedforward_dimension=48,dropout=.1)
            model = build_model(family,random_seed=42,parameters=settings,device=current_device)
            with controlled_thread_budget(1):
                # The unmeasured fit absorbs lazy backend initialization so startup overhead
                # does not masquerade as a family or device training difference.
                warmup = build_model(family,random_seed=42,parameters={**settings, 'epochs': 1},device=current_device)
                warmup.fit(features,targets,(targets+1)%3)
                warmup.synchronize()
                del warmup
                started=perf_counter()
                model.fit(features,targets,(targets+1)%3)
                model.synchronize()
                training_seconds=perf_counter()-started
                model.predict_output(features)
                model.synchronize()
                started=perf_counter()
                output=model.predict_output(features)
                model.synchronize()
                inference_seconds=perf_counter()-started
                with tempfile.TemporaryDirectory(prefix='pricesanity-device-') as temporary:
                    path=Path(temporary)/'model.bin'; model.save(path)
                    restored=load_model(family,str(path),device=current_device)
                    reloaded=restored.predict_output(features)
                    np.testing.assert_array_equal(output.predictions.current,reloaded.predictions.current)
                    np.testing.assert_allclose(output.probabilities.current,reloaded.probabilities.current,rtol=1e-5,atol=1e-6)
                    np.testing.assert_allclose(output.probabilities.anticipated,reloaded.probabilities.anticipated,rtol=1e-5,atol=1e-6)
            rows.append({'model':family,'device':current_device,'training_seconds':training_seconds,
                         'training_steps_per_second':math.ceil(len(features)/8)*2/training_seconds,
                         'training_samples_per_second':len(features)*2/training_seconds,
                         'inference_samples_per_second':len(features)/inference_seconds,
                         'save_reload_verified':True})
    return {'hardware':hardware_diagnostic(),'measurements':rows,
            'accelerator_comparison':'measured' if len(devices)>1 else 'unavailable' if detected=='cpu' else 'not requested',
            'note':'Synthetic 32 × 16 × 4; two epochs; batch 8; one CPU thread. One warm-up fit precedes timing; measured fit includes model setup. Timings are machine-specific.'}
