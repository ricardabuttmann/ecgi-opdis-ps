# Paper benchmark

This guide runs the released methods on the indexed `paper-evaluation-v1` package, from a single
case up to the complete Test cohorts, and combines the five training replicates of a learned method
into the paper values. It assumes the installation from the [README](../README.md#installation) and
extracted packages under `artifacts/`.

## Validate the packages

Verify every byte of the extracted evaluation package:

```bash
python scripts/validate_dataset.py artifacts/paper-evaluation-v1
```

`paper-training-v1` does not repeat the shared chart geometry; validate it against the evaluation
package:

```bash
python scripts/validate_dataset.py artifacts/paper-training-v1 \
  --geometry-root artifacts/paper-evaluation-v1
```

## Contents of the evaluation package

`cases.jsonl` lists 970 Test cases:

| Cohort | `block_id` values | Cases | Available `--forward-spec` |
|---|---|---|---|
| `healthy` | none | 250 | `nominal`, `conductivity`, `subcut_fat_mild` |
| `pathology` | `test_lbbb_mild`, `test_lbbb_severe_cv1p0`, `test_lbbb_severe_cv0p8`, `test_lbbb_severe_complete` | 4 x 60 | `conductivity`, `subcut_fat_mild` |
| `pathology` | `test_rbbb_mild`, `test_rbbb_severe_cv1p0`, `test_rbbb_severe_cv0p8`, `test_rbbb_severe_complete` | 4 x 60 | `conductivity`, `subcut_fat_mild` |
| `pathology` | `test_scar_subendo`, `test_scar_partial`, `test_scar_transmural`, `test_scar_transmural_large` | 4 x 60 | `conductivity`, `subcut_fat_mild` |

`conductivity` is the lung-conductivity discrepancy and `subcut_fat_mild` the subcutaneous-fat
discrepancy; `nominal` uses the reconstruction operator for the recording as well (see
[Operator discrepancies](../README.md#operator-discrepancies)).

## Run one case

```bash
python scripts/run_benchmark_case.py \
  --dataset artifacts/paper-evaluation-v1 \
  --case-id healthy-test-p0001-r00 \
  --forward-spec conductivity \
  --snr-db 40 \
  --method transformer \
  --models artifacts/paper-pretrained-models-v1 \
  --replicate 0
```

The loader joins `truth`, `measurement_clean`, and the nominal reconstruction operator by their
public `case_id`, `geometry_id`, and `operator_id`. It reads only the necessary row from each NumPy
shard, loads the shared surface and regularizer, and recreates the frozen average-referenced
`awgn_iid` noise realization from the anonymous per-case seed in `noise.jsonl`. Pass `--clean`
instead of `--snr-db` to reconstruct from the noise-free measurement. Results are written under
`outputs/benchmark/<case-id>/<method>/` unless `--output` replaces `outputs/benchmark`. Classical
methods do not need `--models`. The OpDis-PS variants require a discrepancy measurement and
therefore reject `--forward-spec nominal`.

## Run a cohort

```bash
python scripts/run_cohort.py \
  --dataset artifacts/paper-evaluation-v1 \
  --cohort healthy \
  --forward-spec conductivity \
  --snr-db 40 \
  --method opdis-ps \
  --models artifacts/paper-pretrained-models-v1 \
  --replicate 0
```

- **Selection.** `--cohort` accepts `all` (default), `healthy`, `pathology`, or one exact
  `block_id` from the table above. `--case-id` selects individual cases and can be repeated;
  `--limit N` keeps the first `N` selected cases.
- **Output.** The default output directory is
  `outputs/cohort/<cohort>/<forward-spec>/snr-<snr>db/<method>/`, with `clean` instead of
  `snr-<snr>db` for `--clean`. Checkpoint-based methods add `replicate-s0` through `replicate-s4`.
  Runs using `--limit` or `--case-id` add a `selection-<count>-<id>/` subdirectory and therefore
  never replace a complete cohort summary.
- **Tables.** `results.csv` contains one row per case and is written after every case.
  `summary.csv` contains the mean and sample standard deviation for each population/pathology
  block. Both tables record the resolved `model_id`, so runs from different replicates cannot be
  mixed up.
- **Resuming.** A completed case is reused, and reported as `skipping`, only when its package
  manifest, operator, public implementation, method config, checkpoint, likelihood artifact,
  forward specification, and SNR all match. Rerunning the same command therefore continues an
  interrupted job. If an existing result differs in any of these identities, the command stops and
  asks for another `--output` or an explicit `--force`, which recomputes the cases.
- **Runtime.** `--limit` does not change the released method configuration: OpDis-PS still draws
  16 samples with 20 diffusion steps and can take several minutes per case on a CPU. Use a CUDA GPU
  for anything beyond a one-case check.

## Aggregate the training replicates

The paper reports, for each learned method, the mean and sample standard deviation across five
independent trainings. After the cohort has completed for all five replicates, run:

```bash
python scripts/aggregate_training_replicates.py \
  outputs/cohort/healthy/conductivity/snr-40db/opdis-ps
```

The command requires `replicate-s0` through `replicate-s4`, verifies identical ordered case and
measurement identities, checks every per-replicate `summary.csv` against its `results.csv`, and
rejects incomplete runs and reused or incorrectly named checkpoints. It writes one immutable
`training-replicates/aggregation-<id>/` directory (or the directory given with `--output`)
containing:

| File | Content |
|---|---|
| `summary.csv` | the paper values; `*_training_sd` is the sample SD across the five training runs |
| `replicates.csv` | the five run-level cohort means; `*_case_sd` is the SD among cases within one run |
| `manifest.json` | the checked dataset, code, configuration, artifact, case-list, and checkpoint identities |

The `*_case_sd` columns are not the paper uncertainty; for a one-case selection they are `nan`.
Stochastic methods keep the released fixed inference `sample_seed` across all five checkpoints, so
the training SD measures training-run variation rather than sampler-seed variation.

To aggregate five matching `--limit 1` smoke tests, pass the selection directory printed by
`run_cohort.py`:

```bash
python scripts/aggregate_training_replicates.py \
  outputs/cohort/healthy/conductivity/snr-40db/transformer \
  --selection selection-1-68391d11b50f
```
