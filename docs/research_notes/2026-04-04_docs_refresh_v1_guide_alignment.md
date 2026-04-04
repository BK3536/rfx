# 2026-04-04 Docs Refresh — v1.0 Guide Alignment

## Context

`rfx` had accumulated substantial functionality since the original guide set was
written: non-uniform z meshing, lumped RLC elements, via/curved geometry
helpers, updated source/port semantics, stronger public docs, and contributor
workflow changes.

Several repo-local guides still reflected older API details, especially:

- old `pulse=` examples instead of `waveform=`
- outdated material-name examples
- no repo-local guide for non-uniform mesh
- no repo-local guide describing validation scope
- stale migration / contributor notes around subgridding-only thinking

## Changes made

Updated:

- `README.md`
- `docs/guide/index.md`
- `docs/guide/installation.md`
- `docs/guide/quickstart.md`
- `docs/guide/simulation_api.md`
- `docs/guide/advanced.md`
- `docs/guide/migration.md`
- `docs/guide/contributing.md`

Added:

- `docs/guide/sources_ports.md`
- `docs/guide/nonuniform_mesh.md`
- `docs/guide/validation.md`

## Intent

The refresh makes the repo-local docs better match the current v1.0 surface and
the public `remilab.ai/rfx/` direction, while keeping the repo docs more
technical and contributor-oriented.

## Verification

Executed:

```bash
pytest -q tests/test_auto_config.py tests/test_nonuniform_api.py tests/test_lumped_rlc.py tests/test_animation.py tests/test_via.py
```

Result:

- `67 passed`

Also checked:

- `git diff --check`
- no stale `pulse=` examples remain in `README.md` / `docs/guide/*.md`
