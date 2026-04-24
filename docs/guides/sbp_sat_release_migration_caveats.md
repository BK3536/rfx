# SBP-SAT release notes and migration caveats

## Release note summary

The current branch includes a documented SBP-SAT subgridding lane, but it is
still intentionally narrow:

- experimental only
- all-PEC plus a selected reflector/periodic boundary subset only
- one axis-aligned refinement box only
- soft point source + point probe only
- proxy numerical-equivalence benchmark only

This is a documentation and contract maturity improvement, not a broad runtime
promotion.

## User-facing caveats

### Boundary caveat

Subgridding currently accepts only:

- `boundary="pec"`; or
- an all-PEC `BoundarySpec`; or
- a selected reflector/periodic subset (`PEC/PMC` reflector faces, or periodic
  axes when the box is interior to that axis or spans it end-to-end, but not a
  mixed PMC+periodic combination)

Any CPML, UPML, absorbing `BoundarySpec`, or one-side-touch periodic axis with
subgridding must be expected to hard-fail.

### Geometry caveat

`Simulation.add_refinement(...)` currently means:

- one axis-aligned all-PEC refinement box only
- explicit `x_range` / `y_range` / `z_range` box bounds are allowed
- geometry-driven `xy_margin` auto-box refinement is still unsupported

### Source / port caveat

Shipped support is limited to `Simulation.add_source(...)` as a soft point
source. The following remain unsupported with subgridding:

- impedance point ports
- wire / extent ports
- coaxial ports
- waveguide ports
- Floquet ports

### Observable caveat

Shipped support is limited to point probes. The following remain unsupported
with subgridding:

- DFT planes
- flux monitors
- NTFF / Huygens-box far field

### Benchmark caveat

`tests/test_subgrid_crossval.py` is a **proxy benchmark** only. It is not a
claims-bearing physical reflection / transmission benchmark, S-parameter
benchmark, or open-boundary validation result.

### Material / time caveat

The current SBP-SAT lane does not yet claim support for:

- material-scaled SAT policies at the coarse/fine interface
- conductive, magnetic, dispersive, anisotropic, or nonlinear material
  widening beyond the current proxy-evidenced baseline
- sub-stepped or multi-rate coarse/fine time integration

## Migration note for maintainers

If older notes, examples, or branches describe SBP-SAT as “z-slab only” or
“general subgridding”, translate that language to the current contract before
reusing it:

- replace “general subgridding” with “experimental all-PEC arbitrary-box lane”
- replace “benchmark validated” with “proxy benchmark validated” unless the
  statement specifically cites a true R/T benchmark
- keep public wording aligned to `docs/guides/support_matrix.*`
