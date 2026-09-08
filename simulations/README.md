# Prepared HSP simulation

The thin entry point preserves the pseudo-bidomain semantics used for the paper with its tested
carputils environment:

- `run_prepared_hsp_v11.py` is the adapter for openCARP/carputils 11;
- `prepared_hsp_common.py` contains the shared scientific command construction.

Users normally invoke `scripts/run_simulation_example.py`, which validates every bundle file,
selects the adapter from `configs/paper/simulation-example.toml`, launches openCARP, and extracts
the average-referenced epicardial HSP from `phie.igb`.

No script in this directory creates meshes, samples cohorts, submits jobs, or discovers private
storage paths.
