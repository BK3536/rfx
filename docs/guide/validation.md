# Validation

This page summarizes how to read `rfx` validation claims.

## Strongly benchmarked areas

The strongest current benchmark areas are:

- PEC cavity resonances
- dielectric-loaded cavity resonances
- rectangular waveguide cutoff / modal workflows
- simple lumped-port reference problems

These are the right places to make strong numerical-agreement claims.

## Antenna / feed caution

Patch-antenna resonance benchmarks are useful and important, but they should be
read more carefully than cavity/waveguide benchmarks.

Best current interpretation:

- **current-source + Harminv** antenna resonance workflow: solid and useful
- **fully calibrated public patch-feed / port validation**: still evolving

That means:

- resonance agreement can be discussed confidently,
- feed/port interpretation should be described more cautiously.

## Recommended wording discipline

When documenting results:

- say **benchmark** or **comparison** when appropriate
- reserve stronger wording for cases with clear reproducible evidence
- distinguish:
  - analytical agreement
  - agreement with Meep/OpenEMS
  - diagnostic results
  - evolving/public-doc examples

## Where to look

- `README.md` for top-level project summary
- `docs/guide/quickstart.md` for current example entry points
- `docs/guide/nonuniform_mesh.md` for thin-substrate workflows
- `docs/research_notes/` for chronological validation history
