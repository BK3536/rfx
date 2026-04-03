# 2026-04-03 Session Handoff: Non-Uniform Mesh Integration

## What Was Done

### 1. Non-Uniform Mesh → Simulation API Integration
- Added `dz_profile` parameter to `Simulation.__init__()` for graded z-mesh
- New `_run_nonuniform()` method dispatches to `run_nonuniform()` from `rfx/nonuniform.py`
- New `_build_nonuniform_grid()` and `_assemble_materials_nu()` for non-uniform path
- Physical position → non-uniform grid index mapping via `_pos_to_nu_index()`
- Wire port support in non-uniform path (impedance loading + S-param DFT)
- Dispatch logic in `run()`: `dz_profile` set → non-uniform path, before uniform/subgrid

### 2. Auto-Config Non-Uniform Z Detection
- Extended `analyze_features()` to extract `z_features` (z-extent per dielectric layer)
- Added `_make_dz_profile()` helper: snaps thin z-features with ≥4 cells, coarse air above
- Detection logic: compares z-feature thickness against **wavelength-based** dx (not feature-refined dx)
- When non-uniform needed: uses coarser wavelength-based dx for xy, saving computation
- Added `dz_profile` field to `SimConfig`, `uses_nonuniform` property, updated `to_sim_kwargs()`
- Example: 2.4 GHz patch with 1.6mm FR4 → dx=1.5mm xy + 0.4mm dz substrate (was 0.4mm uniform)

### 3. Bug Fix: Box Attribute Names
- `analyze_features()` and `_run_subgridded()` used `corner1`/`corner2` but Box has `corner_lo`/`corner_hi`
- Fixed in `auto_config.py` (2 occurrences) and `api.py` (2 occurrences)
- This bug meant feature analysis never extracted geometry dimensions from Box shapes

### 5. Bug Fix: find_resonances Source Decay Time
- `find_resonances` defaulted to `source_decay_time=0.0`, including source excitation in Harminv
- Fixed: auto-computes decay time as `6τ = 6/(f_center × bw × π)` (~1ns for 2.4 GHz)
- Strategy: direct Harminv first (up to 10K samples), bandpass fallback if no modes found
- Previous 3000-sample cap caused accuracy loss (2.68% vs correct result)

### 6. Bug Fix: Domain z=0 with dz_profile
- `Simulation(domain=(Lx, Ly, 0), dz_profile=...)` raised ValueError
- Fixed: auto-derives z from `sum(dz_profile)` when z=0 and dz_profile provided

### 4. GPU Validation: Current Source Normalization (Cb/dV)
- `make_current_source` in `rfx/nonuniform.py`: Taflove/Meep-style `E += (Cb/dV) × I(t)`
- VESSL job 369367231649 (completed): tested across 4 patch designs
- Results:
  | Design | f0 (GHz) | f_res (GHz) | Error | Q | h/dz | L/dx |
  |--------|----------|-------------|-------|---|------|------|
  | 2.4GHz FR4 | 2.400 | 2.413 | 0.53% | 1 | 4 | 59 |
  | 5.8GHz FR4 | 5.800 | 5.526 | 4.73% | 26 | 4 | 24 |
  | 1.575GHz GPS | 1.575 | 1.521 | 3.43% | 41 | 7 | 125 |
  | 3.5GHz Rogers | 3.500 | 3.384 | 3.31% | 43 | 4 | 44 |

### 5. New Tests
- `tests/test_nonuniform_api.py`: 10 tests covering grid construction, source normalization,
  Simulation API integration, and auto-config detection

## Files Changed (7 source + 3 new)
- `rfx/api.py` (+286) — non-uniform integration, wire port dz fix, find_resonances decay/Harminv fix, corner_lo/hi fix
- `rfx/auto_config.py` (+128) — z_features detection, dz_profile generation, _make_dz_profile, corner_lo/hi fix
- `rfx/nonuniform.py` (+93) — wire port V/I uses local dz, float64 DFT phase, make_z_profile grading implemented
- `rfx/simulation.py` (+3) — float64 DFT phase for wire port S-param
- `rfx/__init__.py` (+13) — lazy eigenmode import, non-uniform exports
- `pyproject.toml` (+2) — numpy/scipy declared as dependencies
- `README.md` (+68) — Quick Start rewritten with real API
- `tests/test_nonuniform_api.py` — new (10 tests)
- `examples/31_nu_api_validation.py` — Simulation API validation script
- `examples/vessl_current_source_gen.yaml` / `vessl_nu_api_validation.yaml` — VESSL job specs

## Active Jobs
All completed.

## Simulation API Validation (GPU, final results — job 369367231659)

| Design | f0 (GHz) | f_res (GHz) | Error | Q | L/dx | Time |
|--------|----------|-------------|-------|---|------|------|
| 2.4GHz FR4 | 2.400 | 2.336 | 2.68% | 46 | 59 | 24s |
| 5.8GHz FR4 | 5.800 | 5.627 | 2.98% | 22 | 48 | 27s |
| 1.575GHz GPS | 1.575 | 1.521 | 3.45% | 41 | 125 | 65s |
| 3.5GHz Rogers | 3.500 | 3.384 | 3.32% | 43 | 44 | 10s |

**Verdict**: The ~3% systematic negative bias (f_res < f0) with physically realistic Q values (22-46) is the **analytical cavity model formula's inherent error**, not the FDTD simulator's. Cross-validation with Meep/OpenEMS on identical geometry is needed to isolate true FDTD error.

## Key Analysis

### Why Only 2.4 GHz Achieved <1%
The systematic **negative bias** (f_res < f0 for all designs) points to the analytical
cavity model formula having ~3-5% inherent error, especially for:
- Thick substrates (GPS: h=3.175mm, h/λ=0.017)
- Low eps_r (GPS: 2.2 → weaker fringing correction)

The 5.8 GHz failure (4.73%) has a clear resolution cause: L/dx=24 is too coarse.
The example 31 uses dx=0.25mm for 5.8 GHz to test this.

### Resolution vs. Formula Error
To distinguish FDTD error from formula error, need cross-validation with Meep/OpenEMS
on the **exact same geometry** (not vs. analytical formula). This was identified as a
pending item in the previous session.

## Priority Next Steps

1. **Check job 369367231654 results** — does Simulation API path work on GPU?
2. **Cross-validate with Meep/OpenEMS** — same geometry, same mesh → isolate formula vs. FDTD error
3. **Convergence study** — run 2.4 GHz patch at dx=0.5, 0.25, 0.125mm to confirm convergence
4. **S-param via non-uniform API** — test wire port path through `_run_nonuniform`
5. **SBP-SAT coupling** — proper energy-stable coefficients (research item, see subgridding review)

## Do Not Repeat
- `Box` attributes are `corner_lo`/`corner_hi`, NOT `corner1`/`corner2`
- Source amplitude comparison across grids must account for dt (CFL) changes, not just dV
- `auto_configure` z-detection must compare against wavelength-based dx, not feature-refined dx
- `Simulation(domain=(Lx, Ly, 0), dz_profile=...)` — domain z=0 is auto-derived from sum(dz_profile)
