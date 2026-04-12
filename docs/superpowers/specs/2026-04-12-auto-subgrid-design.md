# Auto-Subgrid Design Spec

**Date**: 2026-04-12
**Project**: rfx FDTD simulator
**Author**: Byungkwan Kim + AI Agent
**Status**: Approved (rev.2 — post-review fixes)
**Reviewers**: Architect agent, Codex agent, Physics critic agent

## Motivation

rfx needs automatic mesh refinement so that:
1. Users don't manually guess where fine mesh is needed
2. LLM agents can programmatically inspect refinement plans and decide whether to apply them
3. Simulation accuracy is validated, not assumed

Two design pillars of rfx drive this: **autograd** (differentiable simulation) and **RF design with LLM agents**. The auto-subgrid system must be both human-inspectable (visualizations) and machine-readable (structured JSON specs).

## Design Principles

1. **No hardcoding** — no axis-specific paths (z-only), no full-domain assumptions (fine spans all x,y). All operations parametric on arbitrary 3D box regions.
2. **No simplified assumptions** — if the physics requires sub-stepping, implement sub-stepping. Don't substitute a "global timestep" shortcut that breaks the stability proof.
3. **Build on proven foundations** — validate each layer before building the next.
4. **Differentiable by default** — `run_refined()` must be JAX-differentiable. Non-differentiable side-effects (visualization, JSON export) live outside the grad path.

## Roadmap: C → A1 → A2 → B

### Phase C: 3D SBP-SAT Stabilization + Validation

**Goal**: Make 3D subgridding trustworthy before building automation on top.

#### C.1 Fix JIT runner H-field coupling (CRITICAL)

The standalone `step_subgrid_3d()` calls both E-field and H-field SAT coupling. The production JIT runner (`jit_runner.py`) only imports and calls `_shared_node_coupling_3d` (E-field). **H-field coupling is entirely absent from the JIT path.** This violates energy conservation.

- File: `rfx/subgridding/jit_runner.py`
- Fix: Import `_shared_node_coupling_h_3d` from `sbp_sat_3d.py`. Call it in the scan body after H-update, before E-update, mirroring the order in `step_subgrid_3d()`.
- Validation: energy test must pass with JIT runner, not just standalone stepper.

#### C.2 Resolve timestep scheme (CRITICAL)

Current inconsistency:
- **2D** (`sbp_sat_2d.py`): temporal sub-stepping — fine grid runs `ratio` sub-steps per coarse step. Matches Cheng et al. literature.
- **3D** (`sbp_sat_3d.py`): global timestep — both grids use same `dt`. **Different scheme from the paper.**

Cheng et al. 2025 derives energy-stable penalty coefficients assuming temporal sub-stepping. These coefficients cannot be transplanted into a global-timestep scheme.

**Decision required before implementation:**

| Option | Description | Pro | Con |
|--------|-------------|-----|-----|
| **Sub-stepping (match paper)** | 3D fine grid runs `ratio` sub-steps, coarse H frozen during fine steps | Mathematically proven stable, matches 2D scheme | Higher implementation complexity, operator-splitting error |
| **Global timestep (current)** | Both grids same `dt` from fine CFL | Simple, no splitting error | No stability proof, coarse grid wastes compute (could use larger dt), paper coefficients invalid |

**Recommendation**: Sub-stepping. It has a stability proof. The operator-splitting error is bounded (O(dt_c), documented in 1D tests). Align 3D with the proven 2D scheme rather than inventing an unproven shortcut.

If sub-stepping is chosen:
- Rewrite `sbp_sat_3d.py` to match `sbp_sat_2d.py` sub-stepping pattern
- Implement Cheng et al. penalty coefficients (which are derived for this scheme)
- Update JIT runner to handle sub-stepping loop

If global timestep is chosen:
- Derive a new stability proof (research-grade task, not engineering)
- Document that this is NOT the Cheng et al. scheme
- Accept that "implement Cheng et al." is misleading

#### C.3 Clarify what changes in penalty coefficients

Current code (`sbp_sat_3d.py:232-246`):
```python
alpha_f = tau * ratio / (ratio + 1)
alpha_c = tau * 1 / (ratio + 1)
```

This may already be the Cheng-derived formula with `tau` as a scaling factor. Phase C must determine:
- Is the current formula correct for the chosen timestep scheme?
- Is the default `tau=0.5` optimal, or should it be `1.0` (paper default)?
- Does H-field coupling use the same or different alpha?

**Deliverable**: Document exactly which equations from Cheng et al. are implemented, with equation numbers.

#### C.4 Stability validation

- Test 1: PEC cavity, Gaussian pulse, 100k steps — energy non-increasing (monotone) to float32 tolerance
- Test 2: CPML domain, 50k steps — energy decays, no late-time growth
- Both tests must pass with **JIT runner** (not standalone stepper)

#### C.5 Accuracy validation (crossval-grade)

Two test cases required:

**Test A (1D-like)**: crossval 04 (multilayer Fresnel slab)
- Uniform fine vs subgridded, R(f)/T(f) mean error < 5%
- Tests z-axis coupling accuracy

**Test B (3D stress test)**: metallic cavity with off-axis source
- Subgridded region at oblique orientation to wave propagation
- Validates 3D corner/edge coupling on all six faces
- Field snapshot comparison vs uniform fine, max error < 10%

Fresnel slab alone is insufficient — it's a 1D problem that doesn't stress 3D coupling.

#### C.6 Test updates

- Remove energy growth warning from `test_sbp_sat_3d.py`
- Add quantitative energy conservation test (non-increasing, JIT runner)
- Add accuracy crossval tests (Fresnel + 3D cavity)
- Fix dead code in `runners/subgridded.py:44` (expression statement not assigned)

**Exit criteria**: JIT runner with H-coupling passes energy conservation AND <5% accuracy on Fresnel AND <10% on 3D cavity.

---

### Phase A1: Indicators + RefinePlan (single-region, existing API)

**Goal**: Build the indicator system and RefinePlan object on top of the existing single-region z-slab API. No runner changes.

#### A1.1 Indicator System

**Architecture**: Plugin-style indicators, each independent. Two strategies for combining.

```python
class RefinementIndicator:
    name: str
    def evaluate(self, sim, result) -> IndicatorResult:
        """Return (error_map: ndarray, regions: List[RefineRegion])"""
        ...
```

**Built-in indicators**:

| Indicator | Source | What it detects |
|-----------|--------|-----------------|
| `GradientIndicator` | `result.state` | High field gradients (under-resolved features) |
| `MaterialBoundaryIndicator` | `sim._geometry` | Dielectric interfaces (subpixel accuracy needed) |
| `PMLProximityIndicator` | `sim._cpml_layers` | Structures near PML (absorption artifacts) |
| `SourceVicinityIndicator` | `sim._sources, sim._ports` | Near-field around sources/ports |

**False-positive suppression (IMPORTANT)**:
- `GradientIndicator` must suppress regions near known material boundaries. High gradient at a dielectric interface is physical, not numerical under-resolution.
- Implementation: `GradientIndicator.evaluate()` accepts optional `exclude_regions: List[Box]`. `MaterialBoundaryIndicator` can provide its detected boundaries as exclusion zones.
- Alternatively: GradientIndicator operates on error residuals (field - expected), not raw field gradients. This requires a reference solution, which may not be available — so exclusion zones are the practical approach.

Each indicator returns regions with:
```python
class RefineRegion:
    box: Box                    # physical coordinates
    ratio: int                  # suggested refinement ratio (per-region)
    reasons: List[str]          # indicator names that contributed
    max_error: float            # peak error in this region
    confidence: float           # 0-1
```

**Combining strategies**:

- `strategy="rule_based"` (default): Each indicator runs independently. Results unioned. Overlapping regions merged. Reason field lists all contributing indicators.
- `strategy="composite"`: Normalized score maps weighted sum. Weights configurable.

**Merge logic**:
- Overlapping boxes: union bounding box, max ratio, concatenated reasons
- Regions < `min_cells` (default 64): dropped
- Total refined cells > `max_fraction` (default 0.3): return `{"recommendation": "use_uniform_fine"}`

#### A1.2 RefinePlan Object

```python
class RefinePlan:
    regions: List[RefineRegion]
    error_maps: Dict[str, ndarray]   # per-indicator
    grid: Grid
    strategy: str
    
    def report(self, path=None):
        """Human-readable: PNG with eps_r + error overlay + region boxes."""
    
    def to_spec(self) -> dict:
        """Agent-readable: JSON-serializable structured output."""
        return {
            "status": "refinement_suggested" | "uniform_fine_recommended" | "no_refinement_needed",
            "regions": [...],
            "metrics": {
                "max_error": float,
                "mean_error": float,
                "cells_to_refine": int,
                "total_cells": int,
                "memory_increase_estimate": float,
                "indicators_triggered": List[str]
            },
            "action": "sim.run_refined(plan)"
        }
```

#### A1.3 API (Phase A1 — on existing single-region API)

```python
result = sim.run(n_steps=1000)
plan = sim.suggest_refinement(result, indicators="all", strategy="rule_based", threshold=0.3)
plan.report(path="refinement_plan.png")   # human
spec = plan.to_spec()                      # agent

# Execute: uses existing add_refinement() internally
# Takes the highest-priority region from plan, maps to z-range
result_refined = sim.run_refined(plan)
```

`suggest_refinement()` always operates on **coarse-grid result** (pre-refinement). If called on a subgridded result, raise ValueError.

#### A1.4 Visualization (`plan.report()`)

2x2 figure:
- Top-left: eps_r geometry with refinement region boxes overlaid
- Top-right: composite error map (or dominant indicator)
- Bottom-left: per-indicator breakdown
- Bottom-right: text summary (region count, cells, memory, recommendations)

---

### Phase A2: 3D Box API + Runner Generalization

**Goal**: Remove z-slab hardcoding. Arbitrary 3D refinement boxes.

#### A2.1 API Extension

```python
sim.add_refinement(
    region=Box(corner_lo, corner_hi),   # arbitrary 3D box
    ratio=3,
    tau=0.5,
)
```

- `sim._refinements: List[RefinementSpec]` (was single `_refinement: dict`)
- Phase A2: still single-region (raise ValueError if called twice). Multi-region deferred to Phase B.
- Backward compat: `z_range=` keyword still accepted, internally converted to Box

#### A2.2 Runner Generalization

Current `runners/subgridded.py` hardcodes:
```python
fi_lo = cpml_layers        # always starts at CPML boundary
fi_hi = nx - cpml_layers   # always ends at CPML boundary
fj_lo = cpml_layers
fj_hi = ny - cpml_layers
# Only k_lo, k_hi come from user's z_range
```

This must be replaced with **parametric index mapping**:
- Convert `Box(corner_lo, corner_hi)` physical coordinates to coarse-grid indices `(fi_lo, fi_hi, fj_lo, fj_hi, fk_lo, fk_hi)`
- Enforce: refined region must be inside CPML boundary (margin >= cpml_layers)
- No axis-specific code paths — all three axes treated identically

#### A2.3 Per-Region Ratio and CFL

With per-region ratio, global `dt` must use the finest `dx_f` across all regions:
```python
dx_f_min = min(region.dx_c / region.ratio for region in regions)
dt = courant * dx_f_min / (c0 * sqrt(3))
```

Each region gets its own interpolation operators (ratio-dependent). Document this explicitly.

#### A2.4 `run_refined(plan)` differentiability

- `run_refined(plan)` MUST be JAX-differentiable (supports `jax.grad`)
- `plan.report()` and `plan.to_spec()` are non-differentiable side-effects — must NOT be called inside a `jax.grad` scope
- The subgridded runner's `lax.scan` is inherently differentiable
- Indicator evaluation and plan construction are outside the grad path

---

### Phase B: Multi-Region + Auto-Convergence (Future)

**Goal**: Multiple refinement regions + iterative convergence.

**Deferred until Phase A2 is validated.** Key challenges:

1. **Multi-region JIT**: `lax.scan` carry shape must be static. Options:
   - Pad all fine-grid states to max size, mask unused cells
   - Sequential coupling passes (one per region)
   - Compile separate scan for each region count (recompile on change)
   
2. **Multi-region energy accounting**: Each region boundary has SAT coupling. Energy tracked per-region + global sum. Not tested — single-region energy must pass first.

3. **Auto-convergence loop**:
```python
result = sim.run(n_steps=1000, auto_refine=True, convergence_target=0.05, max_levels=3)
```
   Internal: coarse → plan → refined → compare → repeat until converged.

4. **Nested subgrids**: subgrid inside subgrid. Requires recursive config.

---

## File Changes Summary

### Phase C
| File | Change |
|------|--------|
| `rfx/subgridding/sbp_sat_3d.py` | Sub-stepping (match 2D), Cheng et al. coefficients, document equations |
| `rfx/subgridding/jit_runner.py` | Add H-field SAT coupling import + call |
| `rfx/runners/subgridded.py` | Fix dead code (line 44) |
| `tests/test_sbp_sat_3d.py` | Energy conservation (JIT), remove warning |
| `tests/test_subgrid_crossval.py` | New: Fresnel + 3D cavity accuracy tests |

### Phase A1
| File | Change |
|------|--------|
| `rfx/indicators.py` | New: base class + 4 indicators + false-positive suppression |
| `rfx/refine_plan.py` | New: RefinePlan, RefineRegion, merge logic, report(), to_spec() |
| `rfx/api.py` | Add `suggest_refinement()`, `run_refined()` |
| `rfx/amr.py` | Deprecate (alias to indicators.py) |
| `tests/test_indicators.py` | New: unit tests per indicator |
| `tests/test_refine_plan.py` | New: merge, report, to_spec |

### Phase A2
| File | Change |
|------|--------|
| `rfx/api.py` | `add_refinement(region=Box(...))`, `_refinements: List` |
| `rfx/runners/subgridded.py` | Parametric 3D box index mapping (no axis-specific code) |
| `rfx/subgridding/jit_runner.py` | Per-region interpolation operators |
| `tests/test_auto_refine_e2e.py` | New: E2E with 3D box |

## Dependencies

- Phase C: Cheng et al. 2025 paper (DOI: 10836194) — must read and identify exact equations
- Phase A1 depends on Phase C completion
- Phase A2 depends on Phase A1 validation
- Phase B depends on Phase A2 validation
- scipy: optional (connected-component labeling)

## Success Criteria

### Phase C
- [ ] JIT runner calls both E-field and H-field SAT coupling
- [ ] 3D uses sub-stepping (aligned with 2D and paper)
- [ ] Penalty coefficients documented with paper equation numbers
- [ ] Energy non-increasing over 100k steps (PEC cavity, JIT runner)
- [ ] Energy decays in CPML domain (50k steps, JIT runner)
- [ ] Fresnel slab: subgridded R/T within 5% of uniform-fine
- [ ] 3D cavity: field error < 10% vs uniform-fine

### Phase A1
- [ ] `suggest_refinement()` returns valid plan for patch antenna
- [ ] Gradient indicator suppresses false positives near material boundaries
- [ ] `plan.report()` produces informative PNG
- [ ] `plan.to_spec()` returns valid JSON for LLM agent
- [ ] `run_refined(plan)` improves accuracy over coarse-only

### Phase A2
- [ ] `add_refinement(region=Box(...))` works for arbitrary 3D box
- [ ] Runner uses parametric index mapping (no hardcoded axes)
- [ ] Per-region ratio works
- [ ] `run_refined()` is JAX-differentiable
- [ ] E2E: coarse → suggest → refine produces correct results

### Phase B (future)
- [ ] Multi-region JIT scan works (2+ non-overlapping regions)
- [ ] Iterative refinement converges
- [ ] Memory budget respected
