# SBP-SAT ports and observables RFC

## Status

This is the **Milestone 7** pre-implementation contract for ports and
observables inside SBP-SAT refined regions.

It does **not** widen shipped runtime support.  The current runtime remains
limited to **soft point sources** and **point probes** whose positions map to
valid fine-grid cells inside the refined box.

## Purpose

Define what must be true before SBP-SAT subgridding can support additional
ports or observables inside refined regions, including:

- impedance point ports
- wire / extent ports
- coaxial ports
- waveguide ports
- Floquet ports
- DFT planes
- flux monitors
- NTFF / Huygens-box observables

This RFC also fixes the source-normalization vocabulary so future port work does
not mix incompatible amplitude conventions across coarse/fine paths.

## Current baseline

The current codebase already establishes five important facts:

1. `Simulation.add_source(...)` reuses `_PortEntry` with `impedance=0.0` to
   represent a soft point source.
2. The subgridded runner currently maps every source and probe through one
   fine-grid indexer and raises if the mapped position lies outside the refined
   z-slab fine grid.
3. The subgridded result surface currently returns `grid=fine_grid`, so the
   observable contract is fine-grid-centric rather than mixed coarse/fine.
4. The subgrid runtime hard-fails impedance point ports, wire/extent ports,
   coaxial ports, waveguide ports, Floquet ports, DFT planes, flux monitors,
   NTFF, TFSF, and lumped RLC.
5. The uniform reference lane already contains candidate normalization helpers:
   `make_j_source(...)`, `make_port_source(...)`, and
   `make_wire_port_sources(...)`.

Milestone 7 starts from that baseline rather than widening support implicitly.

## Non-goals

Milestone 7 does not:

- implement any new port or observable in the subgridded runtime
- relax the current all-PEC / z-slab / fine-grid-only support boundary
- declare DFT / flux / NTFF / modal ports usable in the shipped SBP-SAT lane
- promote any new public support-surface claim

## Port and observable classes

| Class | Item | Intended future role | Current Milestone 7 status |
|---|---|---|---|
| A | soft point source | current baseline excitation | already shipped |
| B | point probe | current baseline observable | already shipped |
| C | impedance point port | future lumped source/load inside fine region | blocked by RFC gate |
| D | wire / extent port | future distributed fine-region port | blocked by RFC gate |
| E | coaxial port | future feed model inside refined region | blocked by RFC gate |
| F | waveguide port | future modal boundary / aperture port | blocked by RFC gate |
| G | Floquet port | future periodic / phased-array excitation | blocked by RFC gate |
| H | DFT plane probe | future planar spectral observable | blocked by RFC gate |
| I | flux monitor | future planar Poynting observable | blocked by RFC gate |
| J | NTFF / Huygens box | future far-field observable | blocked by RFC gate |

## Current shipped contract

### Supported items

The shipped SBP-SAT runtime supports only:

- soft point sources created by `Simulation.add_source(...)`
- point probes created by `Simulation.add_probe(...)` or `add_vector_probe(...)`

### Placement rule

A currently supported source or probe is accepted only if its physical position
maps to a valid **fine-grid cell index** inside the refined box.  If the mapped
index falls outside the refined box, runtime must hard-fail with a placement
error rather than silently sampling or exciting the coarse grid.

### Result-surface rule

The current `Result` object returned by the subgridded path is fine-grid-centric:

- `state = result.state_f`
- `grid = fine_grid`

Therefore any future coarse-side or mixed coarse/fine observable must first
extend the result contract explicitly rather than piggybacking on the current
fine-grid-only result shape.

## Source normalization contract

### Current soft-point-source rule

The current subgridded soft point source uses a **raw field add**:

- find one fine-grid cell index `(i, j, k)`
- evaluate `waveform(t)`
- inject that waveform directly into the chosen field component at that fine
  cell after the subgrid step

This current rule is suitable for resonance probing and proxy benchmarks, but it
is **not** an absolute-power or calibrated-port normalization.

### Current reference normalization helpers

The uniform reference lane already defines two stronger normalization families:

1. **Current-like soft source normalization** via `make_j_source(...)`
   - `Cb = dt / (eps * (1 + sigma*dt/(2*eps)))`
   - injected waveform is `Cb * waveform(t)`
2. **Port-source normalization** via `make_port_source(...)` and
   `make_wire_port_sources(...)`
   - `waveform_port = (Cb / d_parallel) * excitation(t)`
   - multi-cell wire sources divide the excitation by `N_cells`

These helpers are the reference vocabulary for future subgridded port work.

### Future normalization requirements

Any future subgridded port or distributed source must state explicitly:

- whether the injected amplitude is a raw field add, a current-like source, a
  voltage-normalized source, or a power-normalized source
- whether local material values `eps` and `sigma` enter through `Cb`
- whether local cell size or parallel edge length enters through `d_parallel`
- how excitation is distributed across multiple fine cells
- which invariant is preserved across refinement changes:
  - raw field amplitude
  - total current
  - total voltage drop
  - delivered power

Without that explicit declaration, the port/observable is not eligible for
runtime support.

### Modal-source rule

Waveguide and Floquet ports are **not** just point or wire sources with a new
shape.  They require:

- a mode or plane-wave normalization convention
- a direction / incoming-vs-outgoing convention
- a benchmark that verifies the reported S-parameter or modal amplitude against
  a reference result

Therefore waveguide/Floquet support remains blocked until both Milestone 6
boundary coexistence gates and the port benchmark family below are satisfied.

## Placement contract for future support

Each future source/observable must declare one of these placement classes:

| Placement class | Meaning | Current status |
|---|---|---|
| `fine_cell` | maps to one or more fine-grid cells entirely inside the refined box | soft point source / point probe only |
| `fine_face` | lies on a refined-box face but not on an edge/corner | blocked |
| `fine_edge_or_corner` | touches a refined edge or corner ownership region | blocked |
| `coarse_exterior` | lies outside the refined box and samples/excites the coarse grid | blocked until result/source contract expands |
| `cross_interface_surface` | spans both coarse and fine regions or crosses the coarse/fine interface | blocked |
| `outer_boundary_attached` | attaches to an outer domain boundary (waveguide/Floquet/NTFF-style) | blocked until Milestone 6 coexistence gates pass |

### Required placement rule per supported item

Every item that becomes supported later must provide:

1. **one positive placement test** for each supported placement class;
2. **at least one unsupported-placement failure test** proving a nearby invalid
   placement hard-fails with an explicit message.

This is the exact Milestone 7 exit criterion translated into a test contract.

## DFT / flux / NTFF contract

Future DFT planes, flux monitors, and NTFF boxes need more than a yes/no flag.
They must specify:

- whether the surface is allowed to lie entirely in the fine region,
  entirely in the coarse exterior, or both
- whether interface-crossing surfaces are allowed
- if interface-crossing surfaces are allowed, which grid owns each sample and
  how duplicate coverage is prevented
- whether accumulation uses coarse fields, fine fields, or an explicit merged
  sampling operator

Until such a contract exists, DFT planes, flux monitors, and NTFF remain
hard-fail observables for the SBP-SAT lane.

## Unsupported-combination hard-fail matrix

| Combination | Current phase of rejection | Current behavior | Future gate to enable |
|---|---|---|---|
| soft point source outside refined box | `run(...)` | hard-fail placement error | explicit coarse/mixed source contract |
| point probe outside refined box | `run(...)` | hard-fail placement error | explicit coarse/mixed result contract |
| impedance point port in refined box | `run(...)` | hard-fail | impedance-port normalization benchmark |
| wire / extent port in refined box | `run(...)` | hard-fail | distributed port normalization benchmark |
| coaxial port with subgrid | `run(...)` | hard-fail | coaxial feed benchmark + placement contract |
| waveguide port with subgrid | `run(...)` | hard-fail | Milestone 6 boundary coexistence + modal benchmark |
| Floquet port with subgrid | `run(...)` | hard-fail | Milestone 6 boundary coexistence + periodic benchmark |
| DFT plane with subgrid | `run(...)` | hard-fail | planar spectral observable contract + placement benchmark |
| flux monitor with subgrid | `run(...)` | hard-fail | planar flux contract + placement benchmark |
| NTFF with subgrid | `run(...)` | hard-fail | Huygens-box ownership contract + far-field benchmark |
| any interface-crossing port/observable | `run(...)` or earlier | hard-fail by policy | explicit coarse/fine merged ownership model |

## Benchmark families

These benchmarks are implementation gates, not public claims at this stage.

### PO-1: impedance point-port benchmark

- fixture: one localized impedance point port inside the refined region
- reference: uniform-fine run with the same local material and excitation
- metrics: time-series agreement, extracted resonance or transfer response,
  and normalization invariance under refinement

### PO-2: wire / extent port benchmark

- fixture: one multi-cell distributed wire port inside the refined region
- reference: uniform-fine run with the same physical extent
- metrics: distributed-source normalization, total delivered excitation, and
  convergence under finer refinement

### PO-3: coaxial feed benchmark

- fixture: one coaxial feed into a canonical cavity or patch-style geometry
- reference: uniform-fine run and, when available, an external solver or
  analytic check on the same structure
- metrics: resonance / input response agreement plus placement sensitivity

### PO-4: waveguide modal benchmark

- fixture: one waveguide-port excitation with a refined region in the guided
  structure
- gate: only after Milestone 6 boundary coexistence for the required outer
  boundaries is implemented
- metrics: mode amplitude, calibrated S-parameters, and direction convention

### PO-5: Floquet / periodic benchmark

- fixture: one periodic unit-cell or phased-array-style excitation
- gate: only after Milestone 6 periodic coexistence is implemented
- metrics: modal amplitude, scan-angle consistency, and benchmarked periodic
  response vs a reference

### PO-6: planar / far-field observable benchmark

- fixture: DFT plane, flux monitor, or NTFF box placed entirely on one grid
  ownership side first, then (optionally) across explicit merged ownership
  operators later
- metrics: spectral agreement, energy accounting, and ownership invariance

## Implementation plan

### Phase 7A — current supported contract lock

- keep soft point source + point probe as the only shipped support surface
- lock fine-grid placement and fine-grid-centric result semantics

### Phase 7B — impedance / wire / coaxial source normalization

- define and benchmark the normalization family for point, wire, and coaxial
  sources before enabling them in the subgridded runtime

### Phase 7C — result-surface and placement expansion

- define how coarse-side and mixed-placement probes appear in `Result`
- add explicit placement-class tests before any coarse/fine mixed observable is
  accepted

### Phase 7D — planar and far-field observables

- define DFT / flux / NTFF ownership and accumulation rules
- benchmark each class before runtime enablement

### Phase 7E — modal / periodic ports

- defer waveguide and Floquet support until Milestone 6 coexistence gates pass
- benchmark their normalization and direction conventions explicitly

## Implementation gate

Milestone 7 completes when this RFC exists and is regression-locked.
It does **not** mean additional ports or observables are implemented.

Any future port or observable support remains blocked until all of the following
are true:

- the item has an explicit normalization contract
- every supported placement class has a positive test
- at least one nearby unsupported placement has a negative test
- the relevant benchmark family above is implemented and passing
- the support matrix and public docs are updated only after those gates pass
