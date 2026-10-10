# Anonymous ECGI paper artifact

This repository provides reproducible 40 dB ECGI reconstructions for the methods reported in the
paper. Three anonymized examples cover the two operator discrepancies studied in the paper:
`lung-conductivity`, `subcutaneous-fat`, and the pathological
`lbbb-lung-conductivity` case.

The numerical models, samplers, likelihood functions, and metrics under `src/ecgi/` are the
implementations used for the paper experiments. The code under `src/opdis_ps/` and `simulations/`
provides the release-specific interfaces for loading the public artifacts and running the
reconstruction and HSP simulation examples.

During anonymous peer review, this repository is provided under the
[review-only terms](LICENSE). Third-party materials remain subject to their respective terms.

## Installation

Use Python 3.12. The release was tested with Python 3.12.13.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-tested.txt
python -m pip install -e . --no-deps --no-build-isolation
python -m pytest
```

Exact tested package versions and an isolated-environment recipe are documented in
[`docs/environment.md`](docs/environment.md).

## Run an example

Run one fast CPU method on the lung-conductivity example:

```bash
python scripts/run_example.py --method tikhonov --case lung-conductivity
```

Run the same method on the subcutaneous-fat example:

```bash
python scripts/run_example.py --method tikhonov --case subcutaneous-fat
```

Run the pathological complete-LBBB example shown in the paper's qualitative comparison:

```bash
python scripts/run_example.py --method tikhonov --case lbbb-lung-conductivity
```

Run every method through the same interface:

```bash
python scripts/run_all_methods.py --case lung-conductivity
```

Each run writes its reconstruction and metrics under `outputs/<case>/<method>/`. The combined
`results.csv` and `results.html` are updated automatically. The CSV is the machine-readable result;
the HTML file is a static local view of the same values, displayed to four decimal places for easier
inspection in a browser. It does not start a server or upload results. The HTML view opens by default
for an interactive local run; use `--no-open` on a remote machine, cluster, sandbox, or CI runner.

The result table contains the paper metrics in this order: RE, ATE, LE, sCC, and tCC. ATE is
reported in milliseconds and LE in millimetres. The example verifies the released inference path;
it does not replace the paper's cohort-level aggregation over cases and training replicates.
Expected Tikhonov and seed-0 Transformer values for all three examples are provided in
[`docs/reference_results.csv`](docs/reference_results.csv). They give reviewers a quick numerical
comparison target in addition to the automated structural tests. The values were generated through
this release runner and checked against the corresponding production reports. Minor backend
rounding is acceptable; use absolute tolerances of 0.001 for RE/sCC/tCC and 0.01 ms/mm for ATE/LE.

## Run an indexed benchmark case

The accepted-paper dataset is distributed separately as `paper-evaluation-v1` and
`paper-training-v1`. After extracting the evaluation package, verify every downloaded byte:

```bash
python scripts/validate_dataset.py artifacts/paper-evaluation-v1
```

Run one case through the same method adapter used by the bundled examples:

```bash
python scripts/run_benchmark_case.py \
  --dataset artifacts/paper-evaluation-v1 \
  --case-id healthy-test-p0001-r00 \
  --forward-spec conductivity \
  --snr-db 40 \
  --method tikhonov
```

The loader joins `truth`, `measurement_clean`, and the nominal reconstruction operator by their
public `case_id`, `geometry_id`, and `operator_id`. It reads only the necessary row from each NumPy
shard, loads the shared surface and regularizer, and recreates the frozen average-referenced
`awgn_iid` noise direction from the anonymous per-case seed in `noise.jsonl`. Internal simulation
paths and source subject identifiers are neither required nor exposed.

Run a complete public cohort with the same loader and method settings:

```bash
python scripts/run_cohort.py \
  --dataset artifacts/paper-evaluation-v1 \
  --cohort healthy \
  --forward-spec conductivity \
  --snr-db 40 \
  --method tikhonov
```

`--cohort` accepts `all`, `healthy`, `pathology`, or an exact public `block_id` from
`cases.jsonl`, such as `test_lbbb_mild`. Use `--limit 1` for a quick end-to-end check. Results are
written after every case: `results.csv` contains one row per case and `summary.csv` contains the
mean and sample standard deviation for each population/pathology block. The command reuses a case
only when its package manifest, operator, method config, checkpoint, likelihood artifact, forward
specification, and SNR identity match, so rerunning the same command safely continues an interrupted
job. Pass `--force` to recompute completed cases.

## Methods and frozen settings

### Variational regularization

| Method | Paper setting |
|---|---|
| `tikhonov` | spatial order 1, GCV |
| `st-tikhonov` | spatial order 1, temporal order 2, GCV, rho=1 |
| `tv` | alpha=0.01, epsilon=1e-6, tolerance=1e-5, at most 100,000 iterations |

TV can take roughly a minute for each bundled case on a recent laptop CPU; the exact runtime is
hardware-dependent.

### Amortized inference

| Method | Paper setting |
|---|---|
| `transformer` | direct temporal Transformer, seed-0 checkpoint |
| `palette` | BSP-conditioned diffusion, cold start, 50 steps, eta=0, 16-draw mean |

### Posterior sampling

| Method | Paper setting |
|---|---|
| `dps` | zeta=40, 50 steps, eta=0, 16-draw mean |
| `ddrm` | measurement-only DDRM, eta=1, eta_b=1, 20 steps, 16-draw mean |
| `opdis-ps` | conditional-B correction and residual covariance; same DDRM sampler settings |

The two OpDis-PS ablations use the same prior and sampler: `opdis-ps-marginal` is reported as
**OpDis-PS (marginal)** and `opdis-ps-marginal-isotropic` as **OpDis-PS (marginal, matched
isotropic)**.

Tikhonov, ST-Tikhonov, and TV run on CPU. The Transformer automatically selects CUDA, Apple MPS,
or CPU. The stochastic diffusion methods in this release support CUDA and CPU, but not Apple MPS;
they use CUDA when available and otherwise CPU. Their released random-number path uses case-local
seeded generators on CPU and CUDA; exact numerical identity across devices, PyTorch versions, or
CUDA versions is not claimed. A complete `run_all_methods.py` run uses the frozen 16-draw paper
settings and requires a CUDA GPU to finish in practical time. On CPU, run the classical methods and
Transformer as smoke tests; the script prints the name of each method before its potentially long
computation starts.
The official A100 benchmark measured a maximum of 1.972 GiB of PyTorch-allocated CUDA tensor
memory (DPS); DDRM and OpDis-PS each used 1.156 GiB. This excludes framework and process overhead,
so a CUDA GPU with at least 4 GiB VRAM is the practical minimum for the released examples.

## Examples, checkpoints, and likelihood artifacts

The two physiological files in `examples/` share the same anonymous ground truth, geometry, and
nominal reconstruction operator. Their measurements differ according to the selected operator
discrepancy. The third file is the complete-LBBB lung-conductivity case used in the paper's
qualitative comparison. All examples use the paper's deterministic 40 dB noise protocol.

The compact checkpoints are:

- `checkpoints/prior-s0.pt` for DPS, DDRM, the ablations, and OpDis-PS;
- `checkpoints/palette-s0.pt` for Palette;
- `checkpoints/transformer-s0.pt` for the Transformer.

Each example automatically selects the matching files under
`artifacts/<operator-discrepancy>/`: a structured marginal covariance, its trace-matched isotropic
control, and the conditional-B model. The LBBB example therefore reuses
`artifacts/lung-conductivity/`. No command-line choice between internal discrepancy variants is
required.

The public artifact names follow their role in the paper: `opdis-ps.npz` for the main method,
`opdis-ps-marginal.npz` for the marginal ablation, and
`opdis-ps-marginal-isotropic.npz` for its trace-matched isotropic control.

Verify all published binary files with:

```bash
sha256sum -c CHECKSUMS.sha256
python verify_checkpoints.py
```

On standard macOS installations without GNU coreutils, use
`shasum -a 256 -c CHECKSUMS.sha256` instead. The same substitution works for
`SIMULATION_BUNDLE.sha256` below.

## Prepared-case openCARP simulation

The repository also contains the tested openCARP 11 runner for one prepared electrophysiology
simulation. It starts from a complete mesh, runs the fixed simulation, and extracts the epicardial
HSP sequence. In the paper workflow, BSP observations are generated separately from HSPs with the
recording forward operator. The example does not rebuild anatomy, meshes, fibres, APD tags, or
stimulus sites.

Download the 111,073,600-byte `simulation-example-v1.tar.zst` bundle from the
[anonymous OSF project](https://osf.io/26yt7/overview?view_only=3bad4b98b9a94e178377e144739626cf):

```bash
curl -L -o simulation-example-v1.tar.zst \
  'https://osf.io/download/en8c9/?view_only=3bad4b98b9a94e178377e144739626cf'
sha256sum -c SIMULATION_BUNDLE.sha256
mkdir -p artifacts
tar --zstd -xf simulation-example-v1.tar.zst -C artifacts
python scripts/run_simulation_example.py \
  --case-dir artifacts/simulation-example-v1 --seed 1 --validate-only
```

With openCARP/carputils 11 and MPI installed, omit `--validate-only` and add the desired rank count,
for example `--np 8`. The prepared inputs occupy about 519 MiB and include files above GitHub's
normal file-size limit, so the bundle is distributed separately under the review-only terms in its
included `LICENSE.txt`.

See [`docs/simulation_protocol.md`](docs/simulation_protocol.md) for the case contract and
[`THIRD_PARTY_DATA.md`](THIRD_PARTY_DATA.md) for source attribution.

## Reproducibility coverage

This artifact focuses on the paper's inference stage: it provides the reported methods and metrics,
frozen settings and checkpoints, three self-contained 40 dB examples, reference results, and one
prepared openCARP simulation. The compact release boundary keeps every reviewer-facing workflow
explicit and path-independent while the large cohort-generation and training infrastructure remains
outside the artifact.
