# 2026-04-03 Read-Only Review: rfx Subgridding and Adaptive Meshing

## Scope

User request on **2026-04-03**: review the `rfx` project and research
subgridding / adaptive meshing options, with emphasis on what makes sense for
a **JAX-based** solver. This note is **read-only analysis**; no simulator code
was changed.

## Executive Summary

`rfx` already has a meaningful **SBP-SAT subgridding** path. The repository is
not starting from zero:

- `rfx/api.py` exposes `Simulation.add_refinement(...)`
- `rfx/api.py` dispatches to `_run_subgridded(...)`
- `rfx/subgridding/sbp_sat_{1d,2d,3d}.py` implements staged prototypes
- `rfx/subgridding/jit_runner.py` uses `jax.lax.scan`
- `tests/test_sbp_sat_{1d,2d,3d}.py` already checks bounded-energy / stability
  style behavior

The current architecture is best described as:

- **static block refinement**
- **single refinement region**
- **single refinement ratio**
- effectively **fixed-shape arrays**

That is a strong fit for JAX. In contrast, true runtime AMR with changing mesh
topology would be much less compatible with XLA compilation and gradient flow.

## What Exists in the Current Codebase

### Public / high-level surface

- `rfx/api.py:351` — `Simulation.add_refinement(...)`
- `rfx/api.py:1324` — `_run_subgridded(...)`
- `rfx/api.py:1593` — run-path dispatch into the subgridded flow

### Subgridding implementations

- `rfx/subgridding/sbp_sat_1d.py`
  - 1D prototype with SBP norm / diff operators
  - energy-oriented shared-node coupling
- `rfx/subgridding/sbp_sat_2d.py`
  - 2D TM extension
  - rectangular fine region
- `rfx/subgridding/sbp_sat_3d.py`
  - 3D coarse+fine formulation
  - SAT-style coupling on 6 faces
- `rfx/subgridding/jit_runner.py`
  - JIT-compiled coarse/fine update loop via `jax.lax.scan`

### Tests / examples

- `tests/test_sbp_sat_1d.py`
- `tests/test_sbp_sat_2d.py`
- `tests/test_sbp_sat_3d.py`
- `examples/05_patch_antenna_subgrid.py`
- `examples/21_subgrid_jit_patch.py`

### Internal notes already present

- `docs/research_notes/2026-04-01_subgridding_survey.md`
- `docs/research_notes/2026-04-01_sbp_sat_implementation_plan.md`
- `docs/research_notes/2026-04-01_session_handoff_final.md`
- `docs/research_notes/2026-04-02_session_handoff.md`

## Current Architectural Reading

The present implementation is promising, but it is still closer to a
**prototype integration** than a finished general local-refinement system.

### Strengths

1. **JAX-friendly structure**
   - fixed array shapes
   - explicit coarse/fine state
   - `jax.lax.scan` time loop
   - interface logic isolated to a small part of the code

2. **Correct strategic direction**
   - stable subgridding matters more than “fancy AMR”
   - SBP-SAT is a better long-time-stability bet than older interpolation-first
     approaches

3. **Useful differentiable-programming potential**
   - the interface corrections are algebraic array ops
   - this is a much more natural autodiff story than remeshing or topology
     mutation

### Important limitations observed

1. **Refinement is not yet general adaptive meshing**
   - current API is `add_refinement(z_range=..., ratio=...)`
   - this is a constrained refinement model, not arbitrary local 3D AMR

2. **Refinement region construction is still simple**
   - `_run_subgridded(...)` currently sets the fine region to full coarse
     interior in `x/y`, with refinement mainly controlled through `z`
   - this is more like a refined slab than fully local block refinement

3. **`xy_margin` appears underused**
   - the API exposes it, but the current region construction does not yet
     realize a truly geometry-tight x/y block

4. **3D path uses a global timestep limited by fine dx**
   - this preserves stability and simplicity
   - but it mainly saves **memory**, not necessarily maximum step size

5. **Subgrid path is not yet feature-complete relative to the main path**
   - returned result currently omits some richer outputs
   - e.g. subgrid run path returns `s_params=None`
   - downstream analyses are not yet equally integrated

6. **No explicit subgrid-gradient verification yet**
   - project notes say this should be verified
   - I did not find a dedicated test proving `jax.grad` through the subgrid
     interface

7. **Material/geometry rasterization for the fine region is still basic**
   - implementation is currently most comfortable with box-style geometry
   - future support for richer geometry + conformal/subpixel interactions will
     need more thought

## Research Review: What Makes Sense for a JAX-Based Solver

There are indeed many subgridding / adaptive meshing options in the literature,
but JAX changes the ranking substantially.

### Best fit for `rfx`

#### 1. SBP-SAT block subgridding

This remains the strongest option for `rfx`.

Why:

- best stability story
- natural fit for static block interfaces
- compatible with dense field arrays + small interface operators
- consistent with the code already in the repo

For `rfx`, this is the most realistic path to “accelerated local refinement”
without breaking differentiability or compiler friendliness.

#### 2. Static compile-time adaptive meshing / multi-block refinement

This is the next-best extension.

Meaning:

- keep refinement **static during a run**
- allow multiple rectangular refinement blocks
- choose blocks from geometry preprocessing
- compile one fixed coarse/fine/multi-block layout

This preserves the biggest JAX advantage: fixed shapes and predictable memory.

### Worth considering later

#### 3. Temporal subgridding / local time stepping

This could recover performance because the current 3D formulation uses the fine
grid CFL globally.

Potential upside:

- better wall-clock efficiency than global fine-dt

But:

- more interface complexity
- harder stability analysis
- harder implementation / debugging burden

Recommendation: only after the current static subgrid path is hardened.

### Lower priority / riskier for JAX

#### 4. Dynamic AMR

This is the least attractive near-term direction for `rfx`.

Why:

- shape changes are hostile to XLA compilation
- irregular remeshing complicates autodiff
- runtime mesh mutation adds major engineering complexity
- the EM payoff may not justify the implementation cost yet

If ever attempted, it should be **block-structured AMR with fixed block
templates**, not arbitrary cell-wise remeshing.

#### 5. Huygens-style subgridding

Still historically important, but less attractive here than SBP-SAT.

Main issue:

- weaker stability posture
- less aligned with the current repo direction

## Recommended Ranking for `rfx`

| Option | Stability | JAX fit | Effort | Recommendation |
|---|---:|---:|---:|---|
| SBP-SAT static block subgridding | High | High | Medium | **Primary path** |
| Static multi-block adaptive refinement | High | High | Medium-high | **Second path** |
| Local time stepping / temporal subgridding | Medium-high | Medium | High | Later |
| High-order overlapping / virtual-node schemes | Medium-high | Medium-low | High | Research backup |
| Huygens / switched Huygens | Medium | Medium | Medium | Lower priority |
| Dynamic AMR | Variable | Low | Very high | Avoid for now |

## Recommended Next Steps

### Priority 1: Harden what already exists

1. Verify the current subgrid path against uniform fine-grid references
   for a few canonical cases:
   - cavity
   - waveguide
   - patch / layered substrate problem
2. Measure interface reflection explicitly.
3. Add a dedicated test for **`jax.grad` through subgrid coupling**.

### Priority 2: Generalize from refined slab to static local block refinement

1. Move from `z_range`-only thinking toward explicit rectangular refinement
   blocks.
2. Make `xy_margin` meaningful or replace it with a clearer block API.
3. Support multiple compile-time refinement blocks if the single-block version
   proves stable and maintainable.

### Priority 3: Integrate subgridding with the rest of the solver stack

1. S-parameters
2. DFT probes / monitoring
3. NTFF / far-field where relevant
4. geometry handling beyond simple boxes
5. interactions with subpixel / conformal paths

### Priority 4: Revisit temporal refinement only after the above

If runtime, not memory, becomes the dominant bottleneck, then local time
stepping becomes more compelling.

## Practical Conclusion

For `rfx`, the best strategy is:

> **static SBP-SAT block refinement first, richer compile-time adaptive
> meshing second, dynamic AMR last.**

JAX acceleration does not automatically make every meshing strategy attractive.
It strongly rewards:

- fixed-shape arrays
- static block layouts
- algebraic interface operators
- scan-based time marching

The current `rfx` subgridding direction is therefore strategically correct.
The main work left is not choosing a brand-new method, but hardening and
generalizing the one already underway.

## External References

- Cheng et al., SBP-SAT FDTD subgridding:
  https://arxiv.org/abs/2110.09054
- JCP article page for stable SBP-SAT FDTD subgridding:
  https://www.sciencedirect.com/science/article/abs/pii/S0021999123006058
- Iteration-based temporal subgridding (2024):
  https://www.mdpi.com/2227-7390/12/2/302
- Stable local time stepping on adaptive mesh:
  https://www.sciencedirect.com/science/article/abs/pii/S0021999119303870
- FDTDX JOSS paper:
  https://www.tnt.uni-hannover.de/papers/data/1813/joss.pdf
- JAX documentation:
  https://docs.jax.dev/en/latest/
- `jax.checkpoint`:
  https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint.html

## Repo Files Reviewed

- `README.md`
- `docs/guides/simulation_methodology.md`
- `docs/research_notes/2026-04-01_subgridding_survey.md`
- `docs/research_notes/2026-04-01_differentiable_solver_roadmap.md`
- `docs/research_notes/2026-04-01_session_handoff_final.md`
- `docs/research_notes/2026-04-02_next_steps.md`
- `docs/research_notes/2026-04-02_session_handoff.md`
- `rfx/grid.py`
- `rfx/simulation.py`
- `rfx/api.py`
- `rfx/subgridding/runner.py`
- `rfx/subgridding/jit_runner.py`
- `rfx/subgridding/sbp_sat_1d.py`
- `rfx/subgridding/sbp_sat_2d.py`
- `rfx/subgridding/sbp_sat_3d.py`
- `tests/test_sbp_sat_1d.py`
- `tests/test_sbp_sat_2d.py`
- `tests/test_sbp_sat_3d.py`
- `examples/05_patch_antenna_subgrid.py`
- `examples/21_subgrid_jit_patch.py`
