# Artifacts

## Likelihood artifacts

The `lung-conductivity/` and `subcutaneous-fat/` directories contain the OpDis-PS inference
artifacts, estimated once from the frozen training split:

| File | Used by | Content |
|---|---|---|
| `opdis-ps.npz` | `opdis-ps` | correction model, signal and discrepancy means, residual covariance |
| `opdis-ps-marginal.npz` | `opdis-ps-marginal` | marginal structured covariance and mean |
| `opdis-ps-marginal-isotropic.npz` | `opdis-ps-marginal-isotropic` | trace-matched isotropic control covariance and mean |

All three files also store `reduced_basis`, the Helmert basis used for average-referenced electrode
measurements and covariances. For the bundled examples, the runner selects the directory from the
example's `operator_discrepancy`; the LBBB example therefore reuses `lung-conductivity/`. Their checksums are
listed in [`CHECKSUMS.sha256`](../CHECKSUMS.sha256).

## Extracted release packages

The commands in the README and in `docs/` assume that downloaded packages are extracted here:

```text
artifacts/paper-evaluation-v1/
artifacts/paper-training-v1/
artifacts/paper-pretrained-models-v1/
artifacts/native-ecgi-v1/
```

These directories are ignored by Git. Every command also accepts another location.
