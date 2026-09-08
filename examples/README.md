# Public example bundles

The three `.npz` files are non-pickled NumPy archives shared by every method. The
`lbbb-lung-conductivity` archive contains the complete-LBBB case used for the paper's qualitative
comparison at 110 ms.

The two physiological archives represent the same simulated heart case: their HSP ground-truth
sequences are identical and come from the same electrophysiology simulation. They also share the
nominal reconstruction operator, template coordinates, geometry coordinates, surface Laplacian,
and time step. Only the observation generation differs. The identical HSP is propagated through
the respective perturbed recording operator, producing different clean BSP sequences for the
lung-conductivity and subcutaneous-fat cases. Each clean BSP then receives the paper's deterministic
40 dB `awgn_iid` realization.

The LBBB archive contains its own pathological ground truth and geometry and reuses the released
lung-conductivity likelihood artifacts. Public case names replace all internal subject, mesh, and
seed identifiers.

The schema contains:

| Array | Shape | Meaning |
|---|---|---|
| `measurement` | `[electrodes, time]` | noisy average-referenced BSP used as input |
| `measurement_clean` | `[electrodes, time]` | clean BSP before the 40 dB noise realization |
| `forward` | `[electrodes, state]` | nominal average-referenced reconstruction operator |
| `truth` | `[state, time]` | simulated HSP ground truth |
| `points_mm` | `[state, 3]` | coordinates used by the localization metric |
| `template_z`, `template_phi` | `[state]` | fixed chart coordinates |
| `template_shape` | `[2]` | tensor-grid dimensions, 30 x 60 |
| `laplacian_*` | CSR arrays | surface Laplacian for the classical methods |
| `case_id` | scalar string | anonymous example identifier |
| `operator_discrepancy` | scalar string | matching public likelihood-artifact directory |
| `target_snr_db`, `realized_snr_db` | scalar | requested and realized SNR |
| `noise_norm` | scalar | Frobenius norm of the realized perturbation |

Source attestations are SHA-256 digests only; no private paths or original case identifiers are
stored.
