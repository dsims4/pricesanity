# Accelerator setup

## Device abstraction

Price Sanity depends on PyTorch tensor, neural-network, serialization, and device APIs rather
than one GPU vendor. The standalone Transformer and the TCN, GRU, and Transformer benchmark
adapters support `cpu`, `mps`, and `cuda`. Classical scikit-learn estimators remain CPU models.
Explicitly requesting an unavailable accelerator fails rather than silently using CPU.

The standalone trainer and synthetic smoke helper also accept `auto`: CUDA/ROCm is preferred,
then MPS, then CPU. Benchmark study commands require a concrete device and default to CPU.
The same `cuda` selector covers NVIDIA CUDA and compatible AMD ROCm builds. PyTorch documents
this shared API in its [HIP semantics](https://docs.pytorch.org/docs/stable/notes/hip.html).

## Project requirements and platform-specific builds

The package requires Python **>=3.11**. Its optional `training` extra declares **torch>=2.4**;
this is the project API requirement, not a claim that every later wheel supports every platform.
It deliberately contains no CUDA- or ROCm-specific wheel pin. Choose a binary compatible with
the operating system, interpreter, driver, and hardware using the
[PyTorch installation selector](https://pytorch.org/get-started/locally/).

Create a project-local environment and install the selected PyTorch build before the other
extras when following vendor-specific installation instructions:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the platform-appropriate PyTorch build using its official instructions.
python -m pip install -e '.[dev,gui,benchmark]'
python -m pip check
```

Verify that installing extras preserves the intended build. A generic dependency resolution
does not establish accelerator support. One workstation's wheel must not become the dependency
for every platform. Commands here use a POSIX shell. The managed downloader and benchmark writer
use POSIX advisory locks (`fcntl`), so use Linux, macOS, or WSL for those workflows; native Windows
is not an equivalent supported execution path.

## CPU

Use `--device cpu` for a portable sequence-model execution path and for timing classical models.
CPU thread budgets are explicit so BLAS, OpenMP, and PyTorch do not silently give different
families different resources. Small neural workloads can be faster on CPU than an accelerator.

## Apple Silicon / MPS

Use a macOS PyTorch build with an available Metal Performance Shaders backend and select
`--device mps`. Both build support and runtime availability matter; a CPU-only installation
cannot execute MPS. Follow [PyTorch's MPS guidance](https://docs.pytorch.org/docs/stable/notes/mps.html)
and verify the complete adapter smoke before selecting MPS for a study.

## NVIDIA / CUDA

Install a PyTorch CUDA build compatible with the NVIDIA driver and choose `--device cuda`.
The diagnostic must report runtime availability and the intended device, not merely a CUDA
version compiled into the wheel. The project does not require NVIDIA libraries on CPU, MPS,
or AMD environments.

## AMD / ROCm

### Linux

Choose a supported GPU, distribution, driver, Python, and ROCm/PyTorch combination from AMD's
compatibility documentation. PyTorch's public `torch.cuda` interface exposes ROCm devices;
`torch.version.hip` distinguishes a HIP build from a NVIDIA CUDA build. Select `--device cuda`.
A separate `rocm` device string is not part of the project's CLI contract.

### WSL

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

Downloaded ROCm installer packages, including `.deb` files, are environment artifacts and do
not belong in project source. Choose an interpreter supported by the selected vendor wheel;
a project's Python minimum does not imply that every wheel supports every newer interpreter.
Check project and accelerator compatibility before changing a local `.python-version`.

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

For Apple Silicon, substitute `--device mps --compare`. The smoke uses synthetic data and
temporary artifacts; it does not open a benchmark snapshot or its final holdout.

## Tested environments versus support

Project validation has exercised CPU execution, macOS ARM64/MPS, and Linux/WSL ROCm with an
AMD RX 7800 XT. These are examples of tested paths, not required hardware. Python 3.14.7 is
one tested interpreter, not the package minimum. A test result applies only to its recorded
source, Python/PyTorch versions, driver, device, and workload. Keep `hardware` and smoke output,
dependency versions, and study metadata with private experiment records; do not infer success
on another machine from this list. A skipped device test is not accelerator validation.

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
