# Prepared openCARP simulation example

## Scope

This workflow demonstrates the electrophysiology simulation step used to produce heart-surface
potentials (HSPs). It begins after all expensive anatomy, mesh, fiber, tag, and stimulus preparation
is complete:

```text
verified prepared case -> one openCARP run -> phie.igb -> epicardial HSP extraction
```

In the paper pipeline, the simulated HSPs are subsequently mapped to the common heart template and
the BSP observations are generated separately with the recording forward operator. This example
therefore does not extract BSP from the openCARP output. It also does not reproduce the complete
cohort-generation pipeline from CT images.

## Required case bundle

The separately archived directory contains `case_manifest.json` and the following declared roles:

| Role | Typical path | Purpose |
|---|---|---|
| `mesh_points` | `combined_volume.pts` | combined heart/torso node coordinates |
| `mesh_elements` | `combined_volume.elem` | tagged tetrahedra, including APD regions |
| `mesh_fibers` | `combined_volume.lon` | fiber and sheet directions |
| `parameter_file` | `combined_parameter.par` | ionic and conductivity settings |
| `ground_vertices` | `ground.vtx` | extracellular ground constraint |
| `torso_vertices` | `torso_outer.surf.vtx` | torso index set retained with the prepared bundle; not used for HSP extraction |
| `epicardial_vertices` | `epi.vtx` | HSP extraction indices |
| stimulus roles | `stim_*.vtx` | already prepared stimulus node sets |

Every file record contains a relative path, exact byte size, and SHA-256 digest. The manifest also
fixes an anonymous case identifier, the physiology seed, stimulus start times, expected mesh and
HSP dimensions, and the number of output frames. Absolute paths and parent-directory traversal are
rejected.

## Commands

The distributed archive is `simulation-example-v1.tar.zst`. Download it through the anonymous OSF
view-only link, check the byte stream, and extract it below `artifacts/`:

```bash
curl -L -o simulation-example-v1.tar.zst \
  'https://osf.io/download/en8c9/?view_only=3bad4b98b9a94e178377e144739626cf'
sha256sum -c SIMULATION_BUNDLE.sha256
mkdir -p artifacts
tar --zstd -xf simulation-example-v1.tar.zst -C artifacts
```

The expected archive digest is
`00c4475760b179ceb2733d9eb41e73c255a79f9f4b8418e588998d9d4acb5b8d`.

On standard macOS installations without GNU coreutils, use
`shasum -a 256 -c SIMULATION_BUNDLE.sha256` for the same check.

Bundle validation does not import openCARP and can run in the standard Python environment:

```bash
python scripts/run_simulation_example.py \
  --case-dir artifacts/simulation-example-v1 \
  --config configs/paper/simulation-example.toml \
  --output outputs/simulation-example \
  --np 8 \
  --seed 1 \
  --validate-only
```

Remove `--validate-only` on a configured openCARP system. The paper configuration selects the
canonical carputils v11 adapter, 450 ms duration, 5 ms output spacing, pseudo-bidomain mode 2,
200 uA/cm2 stimulus strength, and 2 ms stimulus duration.

The public seed identifies the already prepared APD tags and stimulus plan. It is an anonymized
bundle-local identifier rather than the internal data-generation seed. Supplying a different value
is rejected instead of silently changing only part of the physiology.

## Outputs

The command writes:

- `raw/phie.igb`, the openCARP extracellular-potential output;
- `raw/run_info.txt`, the resolved anonymous simulation settings;
- `hsp.npy`, epicardial potentials average-referenced independently at every time frame;
- `simulation_manifest.json`, output shapes, hashes, referencing conventions, and IGB timing.

Existing output directories are never overwritten.

## Artifact status

The code path and finalized bundle are ready. Its uncompressed inputs occupy approximately 519 MiB
and are therefore distributed separately rather than committed to Git. During anonymous peer
review the derived bundle is provided only under its included review-only notice. Third-party
terms continue to apply; see `THIRD_PARTY_DATA.md` for attribution and modification details. The
anonymous OSF file record reports the same byte size and SHA-256 digest as
`SIMULATION_BUNDLE.sha256`.

The finalized bundle was validated and run end to end with openCARP 11.0. A 72-rank run completed
the 450 ms simulation and produced HSP shape `(46488, 91)`. Compared with the retained production
reference, the maximum absolute HSP difference was below `1e-6` and the correlation was greater
than `0.99999999999999`.
