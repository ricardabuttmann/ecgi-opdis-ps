# Native-resolution dataset

`native-ecgi-v1` keeps every HSP signal on its own epicardial surface instead of the paper's
30 x 60 chart, together with the matching heart placement, torso surface, electrodes, nominal
fine forward operator, and optional lung surfaces. Users can analyze the native signals directly or map them to a template mesh
or another representation of their choice. The released neural models use the 30 x 60 chart;
applying them to a native surface requires a user-defined mapping.

## Contents and components

The package contains 6,170 signal cases (5,100 Train, 100 Validation, 970 Test; 5,450 healthy and
720 pathological) on 2,050 heart-torso geometries built from 1,000 hearts and 139 torsos. Each
geometry has one nominal fine operator.

The package is split into components that can be downloaded independently. Extract all chosen
TAR files into the same destination so that their common `native-ecgi-v1/` root is merged.

| Component | Zenodo files | Paths | Size | Requires |
|---|---|---|---|---|
| `metadata` | `native-ecgi-v1-metadata.tar` | `*.jsonl`, `manifest.json`, `SHA256SUMS`, ... | < 0.01 GiB | — |
| `heart-surfaces` | `native-ecgi-v1-geometry.tar` | `geometry/hearts/` | 0.9 GiB | `metadata` |
| `torso-geometry` | `native-ecgi-v1-geometry.tar` | `geometry/torsos/`, `geometry/placements/` | 0.1 GiB | `metadata` |
| `lung-surfaces` | `native-ecgi-v1-lungs.tar` | `geometry/lungs/` | 0.09 GiB | `metadata` |
| `native-hsp` | `native-ecgi-v1-hsp-part-001-of-014.tar` to `-014-of-014.tar` | `signals/hsp/` | 97 GiB | `metadata`, `heart-surfaces` |
| `operators-native` | `native-ecgi-v1-operators-part-001-of-010.tar` to `-010-of-010.tar` | `operators/native/` | 62 GiB | `metadata`, `heart-surfaces`, `torso-geometry` |

The geometry archive contains both the `heart-surfaces` and `torso-geometry` components. Together
with `metadata`, it forms the minimum usable geometry layer. Add the native HSP for signal
analysis, the operators only when a workflow needs `A_native`, and the small `lung-surfaces`
archive when building a volume conductor of your own.

## Validate

Inspect the installed components and validate all metadata joins:

```bash
python scripts/validate_native_dataset.py artifacts/native-ecgi-v1
```

Add `--checksums` for a complete byte-level verification. That option reads every installed file,
roughly 161 GiB for the complete package.

## Load a case

Load one native HSP and its matching heart surface:

```python
from opdis_ps.native_dataset import NativeDataset

dataset = NativeDataset("artifacts/native-ecgi-v1")
case = dataset.load_case("healthy-test-p0001-r00")

print(case.hsp.shape)
print(case.heart_points_mm.shape)
print(case.heart_faces.shape)
```

The loader reads only the required slice from each memory-mapped NumPy shard and returns HSP on its
native vertices.

With the `operators-native` component installed, the matching electrode measurement can be
generated without a FEM rebuild:

```python
case = dataset.load_case("healthy-test-p0001-r00", include_operator=True)
bsp = case.synthesize_bsp()  # exactly A_native @ hsp
```

The optional torso layer is loaded explicitly:

```python
case = dataset.load_case(
    "healthy-test-p0001-r00",
    include_operator=True,
    include_torso_geometry=True,
)
print(case.torso.points_mm.shape)
print(case.placement.electrode_xyz_mm.shape)
print(case.placement.heart_to_torso)
```

The equivalent command-line inspection is:

```bash
python scripts/inspect_native_case.py \
  --dataset artifacts/native-ecgi-v1 \
  --case-id healthy-test-p0001-r00
```

Add `--include-torso-geometry` to load the torso layer, or `--bsp-output bsp.npy` to save
`A_native @ hsp`.

## Anatomical sources and coordinates

The 1,000 heart surfaces come from the synthetic statistical-shape-model cohort of
[Rodero et al.](https://doi.org/10.5281/zenodo.4506930). CT-derived heart anatomy was used to place
these synthetic hearts in the torsos. Each of the 139 torso surfaces is linked to its public source
CT in TotalSegmentator v2.0.1 through `torso-ct-sources.jsonl`:

```python
source = dataset.torso_source(case.torso_id)
print(source.source_subject_id)       # for example: s0574
print(source.ct_relative_path)        # s0574/ct.nii.gz
print(source.source_dataset_doi)      # 10.5281/zenodo.10047292
```

The released torso points and heart placements use the source CT's physical LPS coordinates in
millimetres, and so do the released lung surfaces. For organs beyond the lungs, users can retrieve
the corresponding public CT and TotalSegmentator masks, preserve their NIfTI physical transform,
and convert the masks into aligned surfaces. Together with the released heart placements, lung
surfaces, and electrode positions, these surfaces can support alternative
volume-conductor models and forward operators after meshing and assigning conductivities. The
released outer torso surface is the cropped and processed boundary used for the ECGI conductor.

## Lung surfaces and adjustments

The lung surfaces exist at two levels:

- **Per torso** (`load_lungs(torso_id)`): the left and right lung surface of each of the 139
  torsos, as segmented from the source CT. "Cleaned" means only that duplicate points within
  1 µm were merged, the same first step the FEM mesher applied.
- **Per geometry** (`load_adjusted_lungs(geometry_id)`): the lungs as they entered one
  heart-torso geometry's FEM mesh. The inserted synthetic heart often intersected the source
  lungs, so affected lungs were pushed radially away from the placed heart; 1,602 of the 2,050
  geometries have a non-zero displacement on at least one side. The same torso can therefore have differently displaced lungs in different geometries.

```python
base_lungs = dataset.load_lungs(case.torso_id)
fem_lungs = dataset.load_adjusted_lungs(case.geometry_id)
print(base_lungs.coordinate_system)
print(fem_lungs.lung_left.points_mm.shape, fem_lungs.lung_left.faces.shape)

adjustment = dataset.lung_adjustment(case.geometry_id)
print(adjustment.radial_origin_mm)
print(adjustment.lung_left.radial_push_mm)
print(adjustment.lung_right.radial_push_mm)
```

For every `geometry_id`, `lung-adjustments.jsonl` records the origin `c` at the centre of the
placed epicardial surface's bounding box and separate left and right displacements `d` in
millimetres, including zero values. Each lung point `p` was moved to
`p_adjusted = p + d * (p - c) / ||p - c||`. `load_adjusted_lungs` applies exactly this formula to
the per-torso surface and leaves a side with `d = 0` unchanged; the triangulation is the same at
both levels. Surfaces constructed again from the source masks should still be checked for
intersections with the placed heart, because their triangulation may differ.

The upstream datasets and their licenses are listed in
[`THIRD_PARTY_DATA.md`](../THIRD_PARTY_DATA.md). Their original terms remain separate from the
license of this repository and the ECGI data packages.
