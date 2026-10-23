# Public example bundles

The three `.npz` files are non-pickled NumPy archives (schema version 2) shared by every method.
Each contains one Test case on the paper's 30 x 60 chart with 176 electrodes and 91 time steps at
5 ms, observed at the paper's deterministic 40 dB noise level.

| File | Activation | Operator discrepancy |
|---|---|---|
| `lung-conductivity.npz` | healthy | lung conductivity |
| `subcutaneous-fat.npz` | healthy | subcutaneous fat |
| `lbbb-lung-conductivity.npz` | complete LBBB, shown at 110 ms in the paper's qualitative comparison | lung conductivity |

The two healthy archives represent the same simulated heart case: their HSP ground truth is
identical, and they share the nominal reconstruction operator, template coordinates, geometry
coordinates, surface Laplacian, and time step. Only the observation differs. The identical HSP is
propagated through the respective perturbed recording operator, producing different clean BSP for
the two discrepancies; each clean BSP then receives the paper's deterministic 40 dB `awgn_iid`
realization. The LBBB archive contains its own pathological ground truth and geometry and reuses
the lung-conductivity likelihood artifacts.

## Schema

`state` is the 1,800-point chart (30 x 60); `time` has 91 samples.

| Array | Shape | Meaning |
|---|---|---|
| `measurement` | `[electrodes, time]` | noisy average-referenced BSP used as input |
| `measurement_clean` | `[electrodes, time]` | clean BSP before the 40 dB noise realization |
| `forward` | `[electrodes, state]` | nominal average-referenced reconstruction operator |
| `truth` | `[state, time]` | simulated HSP ground truth |
| `points_mm` | `[state, 3]` | coordinates used by the localization metric |
| `template_z`, `template_phi` | `[state]` | fixed chart coordinates |
| `template_shape` | `[2]` | tensor-grid dimensions, 30 x 60 |
| `laplacian_data`, `laplacian_indices`, `laplacian_indptr`, `laplacian_shape` | CSR arrays | surface Laplacian for the classical methods |
| `lambda_min` | scalar | smallest positive Laplacian eigenvalue, scales the Tikhonov null-mode ridge |
| `dt_ms` | scalar | time step in milliseconds |
| `case_id` | scalar string | anonymous example identifier |
| `operator_discrepancy` | scalar string | matching likelihood-artifact directory under `artifacts/` |
| `noise_kind`, `noise_seed` | scalar | noise model (`awgn_iid`) and the seed of its realization |
| `target_snr_db`, `realized_snr_db` | scalar | requested and realized SNR |
| `clean_norm`, `noise_norm` | scalar | Frobenius norms of the clean BSP and of the realized perturbation |
| `source_hsp_sha256`, `source_bsp_sha256`, `source_forward_sha256`, `source_electrodes_sha256` | scalar string | SHA-256 digests of the source arrays the example was exported from |
| `schema_version` | scalar | example schema version, currently 2 |
| `activation_pattern`, `paper_figure_time_ms` | scalar | LBBB archive only: `lbbb-complete` and the figure time, 110 ms |
