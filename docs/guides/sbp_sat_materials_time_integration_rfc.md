# SBP-SAT materials, dispersion, and time-integration RFC

## Status

This is the **Milestone 8** pre-implementation contract for materials,
dispersion, and time integration in SBP-SAT subgridding.

It does **not** widen shipped runtime support.  The current runtime still uses
one shared timestep, material-unaware SAT penalties, and a narrow proxy-evidenced
material surface.

## Purpose

Define the rules that must exist before SBP-SAT subgridding can widen beyond
its current material/time envelope.  This RFC covers:

- material-scaled SAT penalty policy
- accepted / blocked material classes
- CFL and sub-stepping decision record
- discrete energy estimate scope
- a benchmark ladder that separates bulk-material error from interface error

## Current baseline

The current codebase already establishes six important facts:

1. The coarse and fine bulk field updates already accept `MaterialArrays` with
   `eps_r`, `sigma`, and `mu_r`.
2. The SAT interface update does **not** use material-dependent scaling; the
   current coefficients depend only on `ratio` and `tau`.
3. The current subgrid config validator enforces one canonical global timestep:
   `dt = phase1_3d_dt(dx_f)`.
4. The shipped SBP-SAT runtime has no sub-stepping machinery.
5. The shipped JIT/subgrid path has no subgridded Debye/Lorentz/Drude,
   anisotropic, or nonlinear state evolution.
6. `compute_energy_3d(...)` is a useful overlap-counted **proxy metric**, but it
   is not yet a general material energy functional because it weights fields by
   vacuum `EPS_0` / `MU_0` rather than the full accepted material model.

Milestone 8 starts from that baseline rather than overstating what current
plumbing implies.

## Non-goals

Milestone 8 does not:

- implement material-scaled SAT coupling
- add sub-stepping
- enable dispersive, anisotropic, magnetic, or nonlinear public support
- reinterpret the current proxy energy metric as a proof-backed conserved
  energy for general materials

## Material classes

| Class | Material/time case | Current code plumbing | Current support status |
|---|---|---|---|
| A | vacuum / isotropic linear dielectric | bulk updates + proxy benchmarks | current internal baseline |
| B | isotropic conductive material (`sigma > 0`) | bulk updates exist | blocked for support claims until benchmarked |
| C | isotropic magnetic material (`mu_r != 1`) | bulk `update_h` supports `mu_r` | blocked for support claims until benchmarked |
| D | Debye dispersion | uniform/nonuniform reference lane has machinery; subgrid runtime does not | blocked |
| E | Lorentz / Drude-style dispersion | reference lane has machinery; subgrid runtime does not | blocked |
| F | anisotropic permittivity | separate reference-lane helpers exist; subgrid runtime does not use them | blocked |
| G | nonlinear / Kerr material | reference-lane machinery exists; subgrid runtime does not use it | blocked |
| H | reduced-dt or sub-stepped coarse/fine integration | no subgrid machinery exists | blocked |

## Bulk-update vs interface-update contract

Future widening must keep these concepts separate:

1. **Bulk update**
   - the FDTD update inside the coarse region and inside the fine region
   - may depend on `eps_r`, `sigma`, `mu_r`, dispersion state, anisotropy, or
     nonlinear state
2. **Interface update**
   - the SAT coupling across the coarse/fine interface
   - currently depends only on `ratio` and `tau`

A material class is not eligible for support merely because the bulk update path
can numerically step it.  The interface policy for that class must be defined
explicitly.

## SAT penalty policy

### Current shipped rule

The current penalty rule is:

- `alpha_c = tau / (ratio + 1)`
- `alpha_f = tau * ratio / (ratio + 1)`

This rule is **material-unaware**:

- no dependence on `eps_r`
- no dependence on `sigma`
- no dependence on `mu_r`
- no dependence on dispersion pole state
- no dependence on anisotropic tensor orientation

That rule remains acceptable only for the current narrow internal baseline and
must not be generalized by implication.

### Future material-scaled penalty requirements

Any future material-bearing interface support must specify, for the accepted
class:

- whether penalty scaling uses electric impedance, magnetic impedance, wave
  speed, energy norm, or another derived quantity
- whether the coarse and fine sides use symmetric or asymmetric scaling
- whether electric and magnetic SAT penalties share the same scaling
- whether lossy or dispersive state enters the penalty directly or only through
  an effective frozen coefficient
- whether the scaling is evaluated cellwise, facewise averaged, or benchmark-fit

Without that explicit rule, the class remains blocked even if the bulk update
already runs.

## CFL and sub-stepping decision record

### Current shipped decision

The shipped SBP-SAT lane uses **one shared timestep**:

- `dt = phase1_3d_dt(dx_f)`
- coarse and fine updates both use that same `dt`
- the fine grid sets the stability limit

### Why this remains the default

The current lane values:

- one canonical scan path
- one canonical config validator
- one direct mapping between source samples and fine-grid time indices
- one simple interface ordering for H-half then E-half updates

This simplicity is part of the current reliability envelope and should not be
traded away casually.

### Sub-stepping gate

Any future reduced-dt or coarse/fine sub-stepping design must specify:

- which grid advances multiple times per outer step
- where SAT coupling is applied in the multi-rate schedule
- how source waveforms and observables are time-aligned across the schedule
- how stability and phase error are benchmarked relative to the shared-dt lane

Until that contract and benchmark family exist, sub-stepping remains blocked.

## Discrete energy estimate contract

### Current proxy metric

`compute_energy_3d(...)` currently provides an overlap-counted proxy metric:

- coarse overlap region is excluded so the coarse/fine overlap is counted once
- electric and magnetic field squares are weighted by `EPS_0` and `MU_0`
- no material-dependent electric/magnetic energy density is included
- no dispersive stored-energy term is included
- no nonlinear stored-energy term is included

This metric is sufficient for current smoke stability checks, but it is not a
proof-backed general discrete energy for accepted future material classes.

### Future energy estimate requirements

Any future material/time widening must state whether the accepted energy metric
includes:

- `eps_r`-scaled electric energy
- `mu_r`-scaled magnetic energy
- conductive dissipation accounting
- Debye / Lorentz / Drude auxiliary stored energy
- anisotropic tensor energy form
- nonlinear stored-energy correction
- sub-step/interface exchange work terms if multi-rate updates are used

Without that explicit statement, the metric remains a proxy diagnostic only.

## Benchmark ladder

These benchmarks are implementation gates, not public claims at this stage.

### MT-1: vacuum / isotropic dielectric interface baseline

- fixture: current proxy geometry with at least one isotropic dielectric slab
- goal: maintain the existing baseline and ensure future material work does not
  regress the current z-slab proxy lane

### MT-2: conductive-material bulk-vs-interface isolation

- fixture: one isotropic lossy slab whose bulk attenuation is measurable
- reference: uniform-fine run with the same material
- goal: separate bulk conductive damping error from interface coupling error

### MT-3: magnetic-material interface isolation

- fixture: one isotropic magnetic slab or block with `mu_r != 1`
- reference: uniform-fine run with the same material
- goal: determine whether the material-unaware SAT penalty is still acceptable
  or clearly fails for magnetic mismatch

### MT-4: dispersive-material gate

- fixture: one Debye or Lorentz material with a documented frequency band
- reference: uniform-fine run using the same auxiliary-state model
- goal: measure whether auxiliary-state error or interface error dominates

### MT-5: anisotropic / nonlinear gate

- fixture: one anisotropic or nonlinear material case with a reference result
- goal: confirm that bulk machinery, interface coupling, and observables all use
  a consistent tensor/state interpretation before any support widening

### MT-6: time-integration gate

- fixture: shared-dt baseline vs any future multi-rate/sub-stepped candidate
- goal: measure phase error, stability margin, and source/observable alignment
  relative to the current shared-dt lane

## Implementation plan

### Phase 8A — current material envelope lock

- document the current proxy-evidenced baseline: vacuum + isotropic dielectric
  fixtures only
- keep support claims for conductivity, magnetic media, and all advanced
  material classes blocked

### Phase 8B — material-scaled SAT derivation surface

- choose the candidate scaling variables for electric and magnetic SAT terms
- benchmark them first on conductive and magnetic isolation fixtures

### Phase 8C — energy metric extension

- separate proxy diagnostic energy from any proof-backed accepted energy form
- add material-aware energy terms only together with a documented accepted scope

### Phase 8D — dispersion / anisotropy / nonlinear gates

- add one class at a time with explicit auxiliary-state and benchmark policy
- do not batch Debye/Lorentz/anisotropic/nonlinear widening into one rollout

### Phase 8E — time-integration gate

- keep shared-dt as the default until a multi-rate schedule has explicit
  stability and phase benchmarks

## Implementation gate

Milestone 8 completes when this RFC exists and is regression-locked.
It does **not** mean material-scaled SAT or sub-stepping is implemented.

Any future material or time-integration widening remains blocked until all of
the following are true:

- the accepted material/time class has an explicit interface policy
- the energy metric for that accepted class is stated explicitly
- the relevant benchmark family above is implemented and passing
- the support matrix and public docs are updated only after those gates pass
