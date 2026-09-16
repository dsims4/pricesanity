# WSL and AMD training

Keep the checkout in WSL's Linux filesystem, for example `~/code/pricesanity`, and use a
pyenv-managed Python with a project-local `.venv`. Avoid training from `/mnt/c` because repeated
small artifact and database operations then cross the Windows filesystem boundary.

## Environment actually tested

On September 15, 2026 this checkout ran on WSL2 kernel
`6.18.33.2-microsoft-standard-WSL2`, Python **3.14.7**, and PyTorch **2.14.0+cu130**.
The CPU was an AMD Ryzen 7 7800X3D (8 physical / 16 logical cores). WSL exposed approximately
15.2 GiB RAM; the host's 32 GB is not the memory budget available to this process.
The installed PyTorch build has CUDA 13.0 support and **no HIP/ROCm build**.
Neither CUDA nor MPS was available. No accelerator name was reported by PyTorch.
TCN, GRU, and Transformer CPU fit/predict/save/reload smoke checks passed.
**RX 7800 XT acceleration has not been verified.**

This CPU stack works on the installed Python 3.14.7; this does not establish compatibility with
an AMD ROCm wheel. Choose the Python version supported by the exact AMD WSL/PyTorch release you
install. A supported Python 3.12 or 3.13 environment is preferable to forcing an unsupported
newer interpreter. Do not change `.python-version` solely because a newer Python exists.

## Set up and inspect

Select a supported installed interpreter with `pyenv local VERSION`, then:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,gui,benchmark]'
# Install the matching PyTorch build using the authoritative instructions below.
pricesanity-benchmark hardware
pricesanity-benchmark device-smoke --device cpu
pricesanity-benchmark device-smoke --device cuda --compare
```

Installing the generic `training` extra can resolve to a CUDA wheel, as this environment shows.
For AMD, use the vendor's compatible ROCm wheel/driver combination. Preserve the selected build
when installing the project's other extras. Record `python -m pip freeze` alongside the study.
Do not transplant an installation command from another ROCm version or another operating system.
Consult [AMD's WSL compatibility matrix](https://rocmdocs.amd.com/projects/radeon/en/latest/docs/compatibility/wsl/wsl_compatibility.html),
[AMD's WSL driver installation guide](https://rocmdocs.amd.com/projects/radeon/en/latest/docs/install/wsl/install-radeon.html),
and [PyTorch's installation selector](https://pytorch.org/get-started/locally/).

ROCm PyTorch uses the `torch.cuda` device interface. `torch.version.hip`, availability, and the
actual name from `torch.cuda.get_device_name()` distinguish a working Radeon backend from a
CUDA build. A successful installation must identify the RX 7800 XT and pass the tiny model
smokes; an imported `torch` module alone does not prove accelerator support.

## Runtime selection and failure messages

Use `--device cpu` for classical estimators. They remain CPU estimators even when a mixed-family
command requests an accelerator for neural models. Use `--device cuda` for TCN, GRU, and
Transformer on a verified ROCm installation; `--device mps` remains available on Apple Silicon.
Explicit unavailable devices fail clearly. The synthetic helper's `--device auto` chooses and
reports an available backend; ordinary study commands require the selected device explicitly or
default to CPU.

An unavailable-device error means the requested backend cannot run in this environment. Check
the build's HIP field, matching driver, WSL support, GPU identity, and interpreter compatibility.
A partial-resume incompatibility means code, dependencies, hardware, device, or thread budget
changed: restore the original environment or start a distinct experiment. Never bypass the
identity check to reuse incompatible partial artifacts.

`device-smoke --compare` uses synthetic 32 × 16 × 4 inputs and a warm-up fit, then measures a
small fit and synchronized inference for each neural family. It verifies save/reload predictions.
Tiny networks may be faster on CPU. The helper reports an unavailable comparison honestly when
no accelerator is detected. Runtime metadata includes the actual model device, processor,
accelerator name when available, CPU thread budget, and DataLoader worker count. Do not pool
unlike hardware timings into one controlled efficiency score.
