# Task recipes

Canonical patterns for recurring rfx tasks. Each recipe answers:
- **When** to use this pattern
- **Which** primitives to use (decision tree vs alternatives)
- **Code skeleton** (copy-paste-able)
- **Why not** the obvious alternative (pitfalls avoided)
- **Canonical example** in the repo

## Discipline

Add a new recipe whenever you:
1. Complete a non-trivial crossval / measurement / analysis pattern, AND
2. That pattern can be reused by future work

Keep each recipe short (~30-60 lines). The goal is to surface the
*right primitive* quickly, not to teach FDTD theory.

## Current recipes

| File | Task | Canonical example |
|------|------|-------------------|
| `rt_measurement.md` | Reflection/transmission R(f), T(f) of a slab / scatterer (dispersive or not) | `examples/crossval/08_material_dispersion.py` |
| `resonance_extraction.md` | Resonance frequencies + Q via Harminv ringdown | `examples/crossval/02_ring_resonator.py` |
| `source_matching_meep.md` | Matching rfx source params to Meep for crossval (GaussianSource, line, plane-wave) | `examples/crossval/05,06,08*.py` |
| `farfield_radiation.md` | Far-field antenna patterns via NTFF, directivity, radiation pattern | `examples/crossval/ (deleted — was broken)` |
| `field_snapshot_comparison.md` | Side-by-side Ez field snapshot panels (rfx vs Meep) — mandatory visual check | `examples/crossval/01,02,05,06,08*.py` |
| `waveguide_sparams.md` | N-port S-matrix for bounded rectangular waveguide structures (TE10 mode decomposition) | `examples/crossval/09_mode_decomposition.py` |
| `patch_antenna_resonance.md` | TM010 resonance of a rectangular patch on a finite ground plane (explicit PEC Box, not `pec_faces`) | `examples/crossval/05_patch_antenna.py` |

See also:
- `../index.md` for durable project knowledge
- `../../../CLAUDE.md` "Feature discovery" section for the grep map
