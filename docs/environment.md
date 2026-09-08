# Reviewer environments

The inverse-method example and the full openCARP simulation have deliberately separate runtime
requirements. A cluster is not required for the reconstruction example.

## Reconstruction and metric table

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

Run the classical CPU methods first as a quick check:

```bash
python scripts/run_example.py --method tikhonov --no-open
python scripts/run_example.py --method tv --no-open
python scripts/run_example.py --method st-tikhonov --no-open
```

The Transformer selects CUDA, Apple MPS, or CPU automatically. The stochastic diffusion methods in
this release support CUDA and CPU, but not Apple MPS. They use case-local seeded generators; exact
numerical identity across devices, PyTorch versions, or CUDA versions is not claimed. A cluster is
not required, but a complete run with the frozen 16-draw diffusion settings requires a CUDA GPU to
finish in practical time. On a remote system, keep `--no-open` and download the selected case's
`outputs/<case>/results.html` after the run.

The official A100 benchmark of the GPU-based methods recorded peak allocated CUDA tensor memory of
0.052 GiB for Transformer, 0.341 GiB for Palette, 1.156 GiB for DDRM and OpDis-PS, and 1.972 GiB
for DPS. These measurements exclude CUDA context, framework, and other process overhead. A CUDA
GPU with at least 4 GiB VRAM is therefore the practical minimum for these released examples; more
memory provides a safer margin across software versions.

## Prepared-case openCARP simulation

Bundle validation and output-table code run in the Python environment above and do not require
openCARP. The full simulation additionally requires:

- openCARP 11.0;
- its matching carputils v11 installation;
- an MPI runtime configured for openCARP.

Confirm the installation before running the public script:

```bash
openCARP -buildinfo
python -c "import carputils; print(carputils)"
```

The archived case contains 1,315,579 nodes and 8,001,130 tetrahedra. A cluster is not a formal
requirement, but a multicore Linux system with at least 16 GiB RAM is the practical target. The
reference run used 72 MPI ranks, peaked at approximately 11.7 GiB, and completed in 5 min 11 s.
Fewer ranks should work on a sufficiently large-memory workstation but will take longer.

The simulation example was validated with openCARP 11.0 and the matching carputils v11
installation. Other openCARP versions have not been tested.
