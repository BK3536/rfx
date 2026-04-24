# SBP-SAT boundary coexistence RFC

## Status

This is the **Milestone 6** pre-implementation contract for future
BoundarySpec coexistence with SBP-SAT subgridding.

It began as a pure pre-implementation contract. The current runtime now includes
a selected **reflector/periodic subset** (PMC reflector faces and periodic axes
under the implemented box-shape restrictions) and a bounded **CPML absorbing
subset** for interior boxes outside the active absorber pad plus one coarse-cell
guard. UPML, per-face CPML thickness overrides, mixed CPML+reflector, mixed
CPML+periodic, and broader mixed boundary classes remain blocked until the
remaining gates in this RFC are satisfied.

## Purpose

Define the rules that must exist before SBP-SAT subgridding can coexist with:

- PMC reflector faces
- periodic axes
- CPML outer boundaries
- UPML outer boundaries
- per-face absorber thickness overrides via `Boundary.lo_thickness` /
  `Boundary.hi_thickness`

This RFC also distinguishes future **open-boundary** benchmark requirements
from the current PEC-cavity proxy benchmarks.

## Current baseline

The current codebase already establishes four important facts:

1. `BoundarySpec` is the canonical boundary model and already carries
   `pec`, `pmc`, `periodic`, `cpml`, and `upml` tokens together with optional
   per-face thickness overrides.
2. `BoundarySpec` already forbids mixing CPML and UPML in the same simulation.
3. `Simulation._validate_subgrid_boundary_mode()` now accepts the currently
   implemented reflector/periodic subset plus bounded CPML boxes that satisfy
   the absorber separation rule. It still rejects UPML, per-face CPML thickness
   overrides, mixed CPML+reflector, mixed CPML+periodic, mixed PMC+periodic
   configurations, and one-side-touch periodic axes.
4. `Simulation._resolve_face_layers()` and related preflight code already define
   the non-subgridded meaning of per-face absorber layer counts.

Milestone 6 starts from that baseline rather than inventing a second boundary
system.

## Non-goals

Milestone 6 does not:

- prove full SBP-SAT stability for every PMC / periodic / CPML / UPML case
- enable physical R/T, S-parameter, or calibrated open-boundary claims
- enable UPML, per-face CPML thickness overrides, or mixed absorber+reflector/periodic classes
- promote the public support matrix beyond experimental proxy evidence

## Boundary coexistence classes

| Class | Outer boundary configuration | Intended future role | Current Milestone 6 status |
|---|---|---|---|
| A | all-PEC | current supported SBP-SAT baseline | already shipped for z-slab only |
| B | reflector-only with PMC faces (all-PMC or mixed PEC/PMC) | reflector coexistence | implemented in the current experimental subset |
| C | periodic axes with reflector faces on remaining axes | unit-cell / translational-symmetry coexistence | implemented only when the box is interior to that axis or spans it end-to-end |
| D | all-absorbing CPML outer boundary | bounded open-boundary coexistence | implemented only when the box is interior to every CPML face by pad + one-cell guard |
| E | all-absorbing UPML outer boundary | future open-boundary coexistence | blocked by RFC gate |
| F | mixed absorber + reflector faces under one `BoundarySpec` | future asymmetric/open-structure coexistence | blocked by RFC gate |
| G | per-face absorber thickness overrides | future asymmetric absorber contract | blocked by RFC gate |
| H | mixed absorber families (`cpml` and `upml`) | never supported in one simulation without a new derivation | invalid at `BoundarySpec` construction today |

## Coexistence invariants

Any future non-PEC coexistence support must preserve these invariants:

1. **Single canonical boundary source of truth**
   - subgrid code reads the canonical `BoundarySpec`, not a parallel
     scalar/flags shadow model.
2. **Single absorber family per simulation**
   - if any absorbing face is active, every absorbing face in the simulation
     uses the same absorber family (`cpml` or `upml`).
3. **Reflector / periodic faces consume zero absorber layers**
   - `pec`, `pmc`, and `periodic` faces own zero active absorber padding on
     that face.
4. **Refinement interface is distinct from outer boundary policy**
   - the fine/coarse interface is not itself an outer `BoundarySpec` face and
     must not silently borrow absorber semantics from the domain boundary.
5. **Benchmark before promotion**
   - no coexistence class is advertised or made public until its explicit
     boundary tests and benchmark family pass.

## Per-face CPML / UPML layer and padding contract

### Resolved layer counts

For any future subgridded coexistence path, define the resolved absorber layer
count on an outer face `f` as:

- `L_f = Boundary.resolved_lo_thickness(cpml_layers)` for `*_lo`
- `L_f = Boundary.resolved_hi_thickness(cpml_layers)` for `*_hi`

If face `f` is `pec`, `pmc`, or `periodic`, then `L_f = 0`.

### Physical pad thickness

For the current uniform-grid coarse lane, the physical absorber thickness on a
face `f` is:

- `pad_f = L_f * dx_c`

For future nonuniform coarse axes, physical pad thickness must be defined from a
**face-local** mapping of the relevant cell sizes.  Until such a mapping is
specified, nonuniform/per-face-absorber/subgrid coexistence remains blocked.

### Separation rule for a future subgrid box

For a future axis-aligned subgrid box with coarse-space bounds
`[fi_lo, fi_hi)`, `[fj_lo, fj_hi)`, `[fk_lo, fk_hi)`, define one coarse-cell
safety guard `g = 1`.

A subgrid box may coexist with an absorbing outer face only if, on each axis:

- `box_lo_a >= pad_lo_a + g * dx_c`
- `box_hi_a <= domain_a - pad_hi_a - g * dx_c`

where `pad_lo_a` and `pad_hi_a` are the resolved per-face absorber pad
thicknesses on that axis.

Equivalently: the coarse cells touched by the subgrid box may neither overlap
nor touch the active absorber pad cells; the guard band starts immediately after
that absorber pad.

### Asymmetric face-layer rule

If `L_lo != L_hi` on one axis, the coexistence implementation must treat the
low and high sides separately for both preflight and benchmark accounting.
Using `max(L_lo, L_hi)` as a single symmetric shortcut is not acceptable for the
subgrid coexistence path.

## Periodic-axis coexistence contract

Periodic coexistence is defined only for full-axis periodicity on an axis:

- `Boundary(lo='periodic', hi='periodic')`

No asymmetric periodic face is allowed.  The currently implemented periodic
subset requires:

1. the periodic axis contributes zero absorber padding;
2. the subgrid interface operators preserve phase consistency across the wrapped
   domain direction;
3. unit-cell benchmarks include at least one periodic axis and at least one
   non-periodic axis.

Any periodic axis touched on only one side by the refinement box remains a
hard-fail condition.

## PMC coexistence contract

PMC coexistence is defined as a reflector coexistence problem, not an absorber
problem.

The currently implemented PMC subset requires:

1. a face-orientation contract for tangential `H` vs tangential `E` treatment at
   the outer PMC face;
2. explicit tests showing that the subgrid interface and the outer PMC face do
   not apply contradictory updates in the same half step;
3. reflector benchmarks separate from the open-boundary benchmark family.

Broader PMC coexistence beyond this reflector subset remains blocked until the
later benchmark and support-promotion gates are satisfied.

## Open-boundary benchmark definitions

These benchmarks are required for future calibrated CPML/UPML claims and are
distinct from the current PEC-cavity and CPML-decay proxy benchmarks.

### OB-1: normal-incidence slab benchmark

- outer boundary: CPML or UPML on the propagation axis
- fixture: vacuum -> dielectric slab -> vacuum
- subgrid placement: refined box fully inside the domain and outside absorber
  pads + guard band
- observables: incident, reflected, and transmitted measurements with a
  documented separation method
- reference: analytic transfer matrix plus uniform-fine numerical reference
- pass metrics: `R(f)`, `T(f)`, phase, and energy residual

### OB-2: oblique face-orientation benchmark

- outer boundary: open on the propagation axis
- fixture: source and probe geometry chosen so the dominant wavefront crosses a
  non-z subgrid face before reaching the observation region
- requirement: separate x-face and y-face cases, not only z-face regression
- reference: uniform-fine comparison at the same effective fine resolution

### OB-3: periodic unit-cell benchmark

- boundary: periodic on one or two transverse axes, absorbing on the remaining
  open axis or axes
- fixture: one translationally periodic structure with a documented source and
  observable contract
- requirement: prove that periodicity and absorber coexistence do not silently
  reintroduce the PEC-cavity proxy assumptions

These benchmarks are implementation gates, not public claims at this stage.

## Unsupported-combination hard-fail matrix

| Combination | Current phase of rejection | Current behavior | Future gate to enable |
|---|---|---|---|
| scalar `boundary='cpml'` + interior guarded subgrid | implemented path | accepted in the current experimental subset | proxy CPML decay evidence now exists; true OB-1 still deferred |
| scalar `boundary='cpml'` + box inside absorber guard | `add_refinement(...)` | hard-fail | keep rejected until a new absorber-interface contract exists |
| scalar `boundary='upml'` + subgrid | `add_refinement(...)` or preflight | hard-fail | OB-1 + coexistence implementation |
| all-CPML `BoundarySpec` with guarded interior box | implemented path | accepted in the current experimental subset | proxy CPML decay evidence now exists; true OB-1 still deferred |
| any `BoundarySpec` face token `upml` | `add_refinement(...)` | hard-fail | absorber RFC + OB-1/OB-2 |
| selected PMC reflector faces in `BoundarySpec` | implemented path | accepted in the current experimental subset | already benchmarked via reflector proxy tests |
| periodic axis in `BoundarySpec` with interior/full-axis box | implemented path | accepted in the current experimental subset | already benchmarked via periodic proxy tests |
| periodic axis touched on one side only | `run(...)` | hard-fail | keep rejected until a one-side-touch contract exists |
| per-face CPML thickness override on any absorbing face | `add_refinement(...)` | hard-fail | per-face padding contract + OB-1 |
| `set_periodic_axes(...)` after refinement with an interior/full-axis box | implemented path | accepted in the current experimental subset | already covered by periodic subset tests |
| `set_periodic_axes(...)` after refinement with one-side-touch periodic geometry | `run(...)` | hard-fail | keep rejected until a one-side-touch contract exists |
| mixed PMC + periodic faces with subgrid | `add_refinement(...)` | hard-fail | explicit combined coexistence spec and tests |
| mixed reflector + absorber faces with subgrid | `add_refinement(...)` | hard-fail | asymmetric coexistence spec + OB-1/OB-3 |
| mixed periodic + CPML faces with subgrid | `add_refinement(...)` | hard-fail | OB-3 periodic absorber benchmark |
| mixed absorber families (`cpml` + `upml`) | `BoundarySpec` construction | invalid configuration | new derivation and new API contract required |

## Implementation plan

### Phase 6A — coexistence metadata contract

- define the internal metadata structure that maps each outer face to
  `{token, resolved_layers, physical_pad, benchmark_family}`
- keep runtime behavior unchanged while tests and docs lock the contract

### Phase 6B — reflector coexistence

- retain PEC/PMC interaction ordering against the subgrid interface
- extend reflector coverage only after new PMC benchmark evidence exists

### Phase 6C — absorber coexistence

- add absorber separation preflight using the resolved layer contract
- support the first CPML-only subset for boxes outside active absorber pads plus a one-cell guard
- keep UPML, per-face CPML thickness overrides, one-side-touch periodic geometry, and CPML+periodic/reflector mixtures out of scope in this phase

### Phase 6D — periodic coexistence

- retain periodic-axis phase continuity and benchmark it separately
- keep reflector/absorber ordering explicit rather than implicit

### Phase 6E — combined coexistence audit

- enumerate and test every combined boundary class that remains unsupported
- update the support matrix only after the explicit tests and open-boundary
  benchmarks pass

## Implementation gate

Milestone 6 completes when this RFC exists and is regression-locked.
It now includes the bounded CPML subset described above, but it does **not**
mean UPML, per-face absorber overrides, calibrated open-boundary R/T, or broad
mixed boundary coexistence is implemented.

Non-PEC coexistence remains blocked until all of the following are true:

- explicit `BoundarySpec` coexistence tests exist for the enabled class
- the relevant open-boundary or reflector benchmark family is implemented and
  passing
- per-face absorber padding is computed by the coexistence contract, not a
  legacy symmetric shortcut
- the support matrix is updated only for enabled subsets after those tests and
  benchmarks pass
