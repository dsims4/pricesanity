# WSL and AMD training

Price Sanity supports PyTorch acceleration on a compatible AMD Radeon/ROCm installation under
Linux or WSL. PyTorch exposes ROCm through its `torch.cuda` API, so project commands select the
device with `--device cuda`; the spelling does not imply that an AMD run uses NVIDIA hardware.

## Environment layout

Keep the checkout in WSL's Linux filesystem, for example `~/code/pricesanity`, rather than
`/mnt/c`. Training and artifact publication perform many small filesystem operations, which are
slower and less predictable across the Windows mount boundary.

Use a project-local virtual environment with a Python version supported by the exact ROCm/PyTorch
release being installed:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,gui,benchmark]'
```

Install the accelerator-enabled PyTorch wheel using the current AMD and PyTorch instructions, then
install or verify the project without replacing that selected build. Compatibility is a matrix of
GPU, Windows/WSL driver, ROCm release, Linux distribution, Python, and PyTorch versions; a command
copied from another release is not a stable project dependency.

Authoritative setup references:

- [AMD Radeon WSL compatibility](https://rocmdocs.amd.com/projects/radeon/en/latest/docs/compatibility/wsl/wsl_compatibility.html)
- [AMD Radeon WSL installation](https://rocmdocs.amd.com/projects/radeon/en/latest/docs/install/wsl/install-radeon.html)
- [PyTorch installation selector](https://pytorch.org/get-started/locally/)

Python 3.14.7 is the current tested Price Sanity interpreter. Accelerator installation may require
a different supported interpreter until the relevant vendor wheel supports that version. Do not
change `.python-version` without checking both project and accelerator compatibility.

## Verify the backend

An importable `torch` module is not proof of GPU execution. Run:

```bash
pricesanity-benchmark hardware
pricesanity-benchmark device-smoke --device cpu
pricesanity-benchmark device-smoke --device cuda --compare
```

A verified AMD result has all of these properties:

- `torch.cuda.is_available()` is true;
- `torch.version.hip` identifies a ROCm build;
- `torch.cuda.get_device_name()` identifies the intended Radeon device;
- TCN, GRU, and Transformer fit, predict, save, and reload successfully;
- reloaded predictions match the saved model's predictions;
- the diagnostic reports the actual device and synchronized timings.

The Price Sanity workflow has been verified with an AMD RX 7800 XT through this ROCm/`cuda`
interface. That result establishes the code path, not universal machine compatibility. Each study
must retain its own hardware and dependency evidence.

## Runtime selection

Classical scikit-learn estimators remain CPU models even when a mixed-family study assigns an
accelerator to PyTorch families. Use:

- `--device cpu` for a CPU-only benchmark;
- `--device cuda` for TCN, GRU, and Transformer on a verified ROCm or NVIDIA CUDA environment;
- `--device mps` on supported Apple Silicon systems.

`device-smoke --compare` uses small synthetic `32 × 16 × 4` inputs and a warm-up before timing fit
and synchronized inference. Small networks can be faster on CPU because transfer and launch costs
dominate. Treat it as a correctness and local planning check, not as a benchmark result.

## Reproducibility and failures

Runtime metadata records Python, PyTorch, backend, actual accelerator name, platform, CPU thread
budget, and DataLoader worker count. Do not pool runtime measurements from unlike hardware into one
controlled efficiency result.

An unavailable-device error means the requested backend cannot execute in the active environment.
Check the wheel's HIP value, driver/ROCm compatibility, device identity, WSL support, and Python
version. A partial-resume incompatibility means source, dependencies, hardware, device, platform,
or thread settings differ from the unfinished artifact. Restore its recorded environment or begin
a distinct run; do not bypass the identity check.

CPU execution requests deterministic PyTorch algorithms where supported. ROCm kernels may still
have release-specific numerical or nondeterministic behavior. Reproducibility therefore means the
same declared source, data, configuration, seed, dependencies, and compatible backend—not bitwise
identity across unrelated accelerators.
