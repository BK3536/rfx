# rfx Program Spec v6: Post-Waveguide-Overhaul Roadmap

## Current State (2026-03-31)

**152 tests passing, 1 xfail (oblique TFSF leakage).**

### Completed Stages
| Stage | Scope | Tests |
|-------|-------|-------|
| 1 | Core 3D Yee solver, CPML, PEC | 20 |
| 2 | Compiled `jax.lax.scan` runner, lumped ports, S-params | 18 |
| 3 | Debye/Lorentz dispersion, lossy conductors, DFT probes | 22 |
| 4 | NTFF far-field, optimizer, visualization, Touchstone I/O | 16 |
| 5 | 2D TE/TM, snapshots, HDF5 checkpoint | 5 |
| 5+ (Codex) | TFSF enhancements, DFT planes, waveguide ports, per-axis boundaries | 30 |
| 5++ (Codex) | Waveguide modal S-params, de-embedding, N-port assembly, multi-axis ports | 21 |
| Verification | TFSF gradient, DFT gradient, oblique Fresnel (xfail) | 3 |

### Architecture Summary
```
rfx/
├── api.py              # High-level Simulation builder (1400+ lines)
├── simulation.py       # Low-level compiled scan runner
├── grid.py             # Yee grid, Courant, per-axis CPML config
├── core/yee.py         # Yee update kernels (E, H)
├── sources/
│   ├── sources.py      # GaussianPulse, LumpedPort
│   ├── tfsf.py         # TFSF plane-wave (normal + oblique stub)
│   └── waveguide_port.py  # Modal waveguide ports, S-matrix assembly
├── boundaries/
│   ├── cpml.py         # CPML absorber (per-axis)
│   └── pec.py          # PEC walls
├── probes/probes.py    # DFT point/plane, S-param probes, windowing
├── materials/          # Debye, Lorentz/Drude, thin conductor
├── geometry/csg.py     # Box, Sphere, Cylinder masks
├── farfield.py         # NTFF, radiation pattern, directivity
├── optimize.py         # Design region, Adam optimizer
├── io.py               # Touchstone read/write
├── visualize.py        # matplotlib helpers
└── checkpoint.py       # HDF5 save/load
```

---

## Phase 6: Simulator Quality & Research Readiness

### Objective
Make rfx trustworthy for three target applications:
1. **Waveguide filter/coupler optimization** (differentiable inverse design)
2. **Antenna characterization** (TFSF + NTFF + RCS)
3. **Publication-ready validation** (cross-validation, convergence studies)

### 6A. Code Hygiene (1 commit, immediate)

1. **Deduplicate `_dft_window_weight`**
   - Identical function exists in both `probes/probes.py` and `sources/waveguide_port.py`
   - Move to a shared utility (e.g., `rfx/core/dft_utils.py`) and import from both
   - Zero behavior change

2. **Commit + push Codex waveguide overhaul**
   - 152 tests pass, review complete
   - Single commit for the +1,793 line changeset

### 6B. End-to-End Optimization Convergence Test (critical quality gap)

**Problem:** `rfx/optimize.py` has `optimize()` with Adam gradient descent, but no test verifies it actually converges on a real problem.

**Deliverable:** `tests/test_optimize_convergence.py` with:
- **Test 1: Impedance matching** — optimize a dielectric slab thickness to minimize |S11| at a target frequency. Verify objective decreases monotonically and final |S11| < -10 dB.
- **Test 2: Waveguide filter** — optimize eps_r in a design region between two waveguide ports to maximize |S21| in a passband. Verify convergence within 20 iterations.

**Why this matters:** Without this test, we cannot claim the differentiable pipeline works end-to-end. This is the single most important gap for a "differentiable EM simulator" identity.

### 6C. Custom Waveform Support (high-value functionality)

**Problem:** Only `GaussianPulse` exists. No CW source, no chirp, no user-defined waveform.

**Deliverable:**
- Add `CWSource(f0, amplitude, ramp_steps)` — sinusoidal with smooth onset
- Add `CustomWaveform(callable)` — user-defined `f(t) -> float`
- Both should work with `make_source()` and the compiled runner
- Test: CW source reaches steady-state, DFT at f0 matches expected amplitude

### 6D. Magnetic Material Validation (correctness gap)

**Problem:** `mu_r` parameter exists in `MaterialArrays` and the Yee H-update, but no test validates correctness for mu_r ≠ 1.

**Deliverable:** `tests/test_magnetic.py` with:
- **Test 1: Impedance** — Verify plane-wave impedance η = √(μ/ε) * η₀ in a μ_r=4 slab via reflection coefficient
- **Test 2: Phase velocity** — Verify v = c/√(μ_r·ε_r) via travel-time measurement

### 6E. Oblique TFSF via Dispersion-Matched 1D Auxiliary Grid

**Problem:** Current analytic oblique TFSF has 27% vacuum leakage (xfail).

**Approach:** Instead of a full 2D aux grid, use Schneider's "1D FDTD with modified dx" trick:
- For angle θ, use `dx_1d = dx * cos(θ)` and `dt_1d = dt`
- The 1D aux grid then has the same numerical dispersion as the 3D grid along the oblique direction
- Add transverse phase correction `exp(j·k_y·j·dy)` at each TFSF boundary cell

**Deliverable:**
- Modified `update_tfsf_1d_h/e` for oblique with dispersion-matched dx
- Updated `apply_tfsf_h/e` with per-cell transverse phase
- Passing `test_oblique_tfsf_fresnel` (remove xfail)

### 6F. RCS Pipeline (antenna/scattering research enabler)

**Problem:** No monostatic/bistatic RCS computation. NTFF box exists but needs TFSF integration.

**Deliverable:**
- `rfx.rcs.compute_rcs(sim, theta_range, phi_range)` using TFSF + NTFF
- Monostatic: single TFSF angle, NTFF in backscatter direction
- Bistatic: full angular sweep from NTFF
- Validation: PEC sphere RCS vs Mie series (analytical)

---

## Phase 7: Public Release Quality (after Phase 6)

### 7A. README & Examples
- README with installation, quick start, feature matrix
- 3 example notebooks: cavity resonator, waveguide filter optimization, antenna RCS
- Each notebook runs in < 5 min on CPU

### 7B. API Polish
- Optimization objective library (`target_s11`, `target_bandwidth`, `maximize_directivity`)
- Boolean CSG operations (union, difference, intersection)
- `Simulation.suggest_dx()` mesh convergence helper

### 7C. Documentation
- Docstring completeness audit
- Theory reference for each physics module
- Contributor guide

---

## Priority & Dependency Graph

```
6A (hygiene) ──────────────────────────────────────> commit
6B (optimizer test) ───────────────────────────────> commit
6C (CW source) ────────────────────────────────────> commit
6D (mu_r validation) ──────────────────────────────> commit
6E (oblique TFSF) ─────> 6F (RCS pipeline) ───────> commit
                                                      │
                                                      v
                                               Phase 7 (release)
```

6A–6D are independent and can be parallelized.
6E blocks 6F (RCS needs clean oblique TFSF).
Phase 7 requires all of Phase 6.

## Acceptance Criteria for Phase 6

- [ ] 6A: `_dft_window_weight` deduplicated, all 152+ tests pass
- [ ] 6B: `test_optimize_convergence.py` with 2 convergence tests passing
- [ ] 6C: CW + custom waveform sources with tests
- [ ] 6D: `test_magnetic.py` with 2 validation tests passing
- [ ] 6E: `test_oblique_tfsf_fresnel` passes (xfail removed)
- [ ] 6F: `test_rcs.py` with PEC sphere Mie validation
- [ ] Total test count: 165+
- [ ] Zero regressions in existing suite
