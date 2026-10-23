# Public training entry points

The three paper model families can be trained from an **extracted** `paper-training-v1` package. It
contains 5,100 Train and 100 Validation HSP fields paired with clean nominal FEM BSP. The entry
points match the `hsp` and `measurements/nominal` shards by public `case_id`.

Install the repository as described in the [README](../README.md#installation) and validate the
package against the evaluation package, which holds the shared chart geometry:

```bash
python scripts/validate_dataset.py artifacts/paper-training-v1 \
  --geometry-root artifacts/paper-evaluation-v1
```

Then run one of:

```bash
python -m ecgi.pipeline.train_diffusion --dataset-dir artifacts/paper-training-v1 --config configs/training/prior.toml --output-dir runs/prior-s0
python -m ecgi.pipeline.train_transformer --dataset-dir artifacts/paper-training-v1 --config configs/training/transformer.toml --output-dir runs/transformer-s0
python -m ecgi.pipeline.train_conditional_diffusion --dataset-dir artifacts/paper-training-v1 --config configs/training/palette.toml --output-dir runs/palette-s0
```

Each command requires a new output directory and writes `config.toml`, `provenance.json`,
`history.jsonl`, `latest.pt`, and the lowest Validation-loss `best.pt`. Checkpoints contain the
selected EMA weights for the prior and Palette, and raw weights for the Transformer, in the public
inference checkpoint format. `--device auto|cpu|cuda|mps` chooses hardware (default `auto`);
`--max-steps 1` checks one update and one validation pass. The TOML files freeze the released model
architecture, normalization scales, chart, loss, augmentation, and run settings.

The prior and Palette regress the v diffusion target. Transformer training uses per-sample
average-referenced iid noise with a uniform 10–40 dB SNR band and 20% clean probability. Palette
training uses clean nominal BSP conditions; at inference, it conditions on the observed BSP.

The paper's full numerical reproduction additionally needs the complete five-seed training and
selection protocol, 15 selected checkpoints, and substantial accelerator time. These entry points
expose the training algorithms and a Validation-loss selection rule. The published checkpoints and
frozen Test evaluation remain the reference for exact paper numbers.
