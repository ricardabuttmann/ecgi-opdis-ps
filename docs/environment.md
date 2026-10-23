# Reproducible environment

## Installation

Use Python 3.12. The release was tested with Python 3.12.13 and the exact top-level package
versions in `requirements-tested.txt`. Create an isolated environment from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-tested.txt
python -m pip install -e . --no-deps --no-build-isolation
python -m pytest
```

The first install pins the tested NumPy, SciPy, PyTorch, and pytest versions together with the
build tools. The editable install then adds this repository without resolving dependencies again
(`--no-deps`) and builds with the already installed setuptools (`--no-build-isolation`), so it needs
no second, network-dependent build environment.

Run the classical CPU methods first as a quick check:

```bash
python scripts/run_example.py --method tikhonov --no-open
python scripts/run_example.py --method tv --no-open
python scripts/run_example.py --method st-tikhonov --no-open
```

## Devices

Tikhonov, ST-Tikhonov, and TV run on CPU. The Transformer selects CUDA, Apple MPS, or CPU
automatically. The diffusion methods (Palette, DPS, DDRM, and the OpDis-PS variants) use CUDA when
available and otherwise CPU; Apple MPS is not supported for them in this release. Their released
random-number path uses case-local seeded generators on CPU and CUDA; numerical results can vary
across devices, PyTorch versions, and CUDA versions.

A complete run with the frozen 16-draw diffusion settings requires a CUDA GPU to finish in
practical time. `run_all_methods.py` prints the name of each method before its potentially long
computation starts. On a remote system, keep `--no-open` and download the selected case's
`outputs/<case>/results.html` after the run.

## GPU memory

The official A100 benchmark of the GPU-based methods recorded peak allocated CUDA tensor memory of
0.052 GiB for Transformer, 0.341 GiB for Palette, 1.156 GiB for DDRM and OpDis-PS, and 1.972 GiB
for DPS. These measurements exclude CUDA context, framework, and other process overhead. A CUDA
GPU with at least 4 GiB VRAM is therefore the practical minimum for the released examples; more
memory provides a safer margin across software versions.
