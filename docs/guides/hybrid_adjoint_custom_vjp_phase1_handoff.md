# Hybrid adjoint custom_vjp — Phase 1 handoff

Status: execution handoff

## Phase 1A

Goal:
- extract and document the canonical `simulation.py` step/replay seam

Required outputs:
- seam map
- carry inventory
- replay contract

## Phase 1B

Goal:
- implement Strategy A POC on the extracted seam

Required fixture:
- uniform
- lossless
- PEC-only
- no CPML
- no Debye/Lorentz
- no NTFF
- simple `time_series` objective

Required acceptance:
- pure AD vs hybrid relative gradient error `<= 1e-4`
- deterministic replay
- explicit fallback to pure AD for unsupported paths

## Stop conditions

Stop and revise the plan if:
- the canonical seam cannot be identified cleanly
- the POC requires CPML/dispersion to make progress
- the replay contract is too entangled to keep Phase 1 narrow

## 2026-04-16 implementation status

Phase 1 now has an explicit experimental entrypoint:

- `Simulation.forward_hybrid_phase1(...)`

Implemented surface:

- uniform grid
- lossless materials
- PEC-only boundary
- point-source / point-probe time-series objectives
- extracted fast-path seam using PEC-baked coefficients + `update_he_fast(...)`

Explicit fallback / rejection:

- non-uniform grids
- CPML / non-PEC boundaries
- periodic / Floquet-style paths
- Debye / Lorentz dispersion
- lumped or wire-port source paths
- NTFF / waveguide-port accumulation
- PEC mask / occupancy replay
- lossy materials / port-loaded conductivity

Verification snapshot:

- `pytest -q tests/test_hybrid_adjoint_phase1.py`
- `pytest -q tests/test_optimize.py -k 'forward_returns_minimal_result_contract or gradient_check_matches'`
- `pytest -q tests/test_differentiable_material_fit.py -k 'test_gradient_through_fdtd'`

The Phase 1 gradient-agreement fixtures now cover both:

- an interior design cell
- a source-coupled design cell

Phase 2 hardening has also started with a public inspection surface:

- `Simulation.inspect_hybrid_phase1(...)` reports support/fallback reasons
- the same inspection path exposes replay inventory without reaching into private prep helpers
- `Simulation.build_hybrid_phase1_context(...)` now builds the stable replay context for supported Phase 1 runs
- `Simulation.forward_hybrid_phase1_from_context(...)` now executes that built context through a stable API surface
- `Simulation.prepare_hybrid_phase1(...)` now returns the public inspection + context bundle for supported/unsupported Phase 1 requests
- `Simulation.forward_hybrid_phase1_from_prepared(...)` now executes that public bundle directly without forcing callers back onto separate context plumbing
- downstream callers can now target one stable prepared Phase 1 contract instead of stitching together separate inspect/build calls
- the prepared bundle itself now exposes the canonical support/inventory/require_context surface, reducing caller-side branching further
- the prepared bundle now also exposes canonical `reason_text` / `require_supported()` behavior, so unsupported-path handling no longer needs ad hoc string joining in higher layers
- the prepared bundle now also exposes core seam metadata (`source_count`, `probe_count`, `boundary`, `periodic`) so higher layers can stay on the bundle surface without reopening the report
- the prepared bundle can now execute the replay seam directly at the time-series layer, further reducing caller-side plumbing
- the replay context itself is now self-executing at the time-series layer, and the higher-level bundle/API helpers delegate to that canonical path
- the replay context and prepared bundle can now both emit a ForwardResult directly, further collapsing result-shaping duplication across the public APIs
- result shaping is now also exposed through a seam-owned `phase1_forward_result(...)` helper, which the context delegates through
- prepared-bundle construction itself is now owned by `rfx/hybrid_adjoint.py`, so the seam module—not `Simulation`—owns the canonical report/context assembly too
- `Simulation.build_hybrid_phase1_inputs(...)` now exposes the seam-owned input spec directly for uniform Phase 1 preparation paths
- that seam-owned input type now also owns construction from the full inspected runner state (supported or unsupported), further reducing API-layer branching around input creation
- that seam-owned input spec now also exposes the canonical reason/support/run surfaces directly, further reducing caller dependence on higher wrappers
- seam-owned forward helper aliases now also cover the prepared-runner and inspected-runner execution paths, completing the main execution helper family across the major Phase 1 seam states
- the seam-owned replay inventory type is now also exported at the package root, so callers can stay on the public contract surface even when typing against inspection inventory
- the seam-owned field-carry type is now also exported at the package root, so callers can stay on the public contract surface when typing against built replay contexts
- the raw seam execution primitive and custom_vjp forward factory are now also exported at the package root, so low-level Phase 1 execution can stay on the public contract surface too
- the canonical support-reason helper is now also exported at the package root, so unsupported-path gating can stay on the same public contract surface too
- public Simulation execution wrappers now also cover the prepared-runner and inspected-runner seam states, so callers can stay on the high-level API even when driving lower-level seam snapshots
- public Simulation inspect/prepare/context wrappers now also cover those runner-state seam surfaces, rounding out the high-level API family for lower-level seam snapshots
- public Simulation input-builder wrappers now also cover those runner-state seam surfaces, so all major lower-level seam snapshots can still enter the canonical input object through the high-level API
- the seam-owned input spec now also caches its canonical report and prepared bundle views, further concentrating repeat access on the seam object itself
- the built context now carries the supported baseline `eps_r`, so context execution reproduces lossless dielectric setups without reconstructing prep state
- the public `Simulation` family is now symmetric across top-level and runner-state entry surfaces for input/inspect/prepare/context/forward, reducing surface-specific drift risk before later subsystem expansion
- the public `Simulation` wrapper family and the seam-owned `hybrid_adjoint` helper family now both carry explicit parameter/return annotations, making the current Phase 1 contract easier to read and safer to extend
- the regression suite now also leans on narrower shared fixture/helpers for supported and unsupported setup paths, reducing repeated test scaffolding while preserving behavior coverage
- contract regressions now guard root exports, wrapper/default symmetry, omitted-`n_steps` behavior on supported and unsupported lanes, and unsupported reason-text parity across inspect/prepare/context/forward surfaces
- the current regression guard now spans top-level, runner-state, input-surface, prepare-bundle, and forward/context raise paths across both supported and unsupported entry modes, making surface-drift harder to reintroduce before later subsystem expansion
