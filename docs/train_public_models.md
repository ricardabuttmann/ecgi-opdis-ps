# Public training entry points

The three paper model families can be trained from an **extracted** `paper-training-v1` package. The package contains 5,100 Train and 100 Validation HSP fields and their clean nominal FEM BSP pairs. The entry points join the `hsp` and `measurements/nominal` shards by public `case_id`. No private source catalog, scheduler, mesh simulation, or data materialization is used.

Install the repository with `pip install -e .`, then run one of:

```bash
python -m ecgi.pipeline.train_diffusion --dataset-dir /data/paper-training-v1 --config configs/training/prior.toml --output-dir /results/prior-s0
python -m ecgi.pipeline.train_transformer --dataset-dir /data/paper-training-v1 --config configs/training/transformer.toml --output-dir /results/transformer-s0
python -m ecgi.pipeline.train_conditional_diffusion --dataset-dir /data/paper-training-v1 --config configs/training/palette.toml --output-dir /results/palette-s0
```

Each command requires a new output directory and writes `config.toml`, `provenance.json`, `history.jsonl`, `latest.pt`, and the lowest Validation-loss `best.pt`. Checkpoints contain the selected EMA weights for the prior and Palette, and raw weights for the Transformer, in the public inference checkpoint format. `--device cpu|cuda|mps` chooses hardware; `--max-steps 1` checks one update and one validation pass. The TOML files freeze the released model architecture, normalization scales, chart, loss, augmentation and run settings. The prior and Palette regress the v diffusion target; Transformer uses per-sample average-referenced iid noise with a uniform 10–40 dB SNR band and 20% clean probability; Palette uses clean BSP conditions.

The paper's full numerical reproduction additionally needs the complete validated `paper-training-v1` package, the shared chart/geometry identity in `paper-evaluation-v1`, the complete five-seed training/selection protocol and 15 selected checkpoints, and substantial accelerator time. These entry points expose the training algorithms and a Validation-loss selection rule; they do not recreate the original campaign's exact five runs, private cache attestations, or any raw-cohort generation. The selected published checkpoints and frozen Test evaluation remain the reference for exact paper numbers.
