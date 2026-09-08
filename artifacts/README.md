# Likelihood artifacts

The `lung-conductivity/` and `subcutaneous-fat/` directories contain the
inference artifacts derived from frozen training-split estimates:

- `opdis-ps.npz`: OpDis-PS correction model, signal/discrepancy means, residual covariance, and
  Helmert basis used by OpDis-PS;
- `opdis-ps-marginal.npz`: marginal structured covariance, mean, and Helmert basis;
- `opdis-ps-marginal-isotropic.npz`: trace-matched isotropic control covariance, mean, and Helmert
  basis.

Each file contains only the arrays consumed during inference.
