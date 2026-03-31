# 2026-03-30: Boundary-aperture waveguide multiport support generalized beyond 2 ports

## What changed

### High-level waveguide ports
- `Simulation.add_waveguide_port(...)` now accepts:
  - `direction`
  - `y_range`
  - `z_range`
- Ports can now describe disjoint rectangular apertures on the x-end boundary
  planes instead of always consuming the full y-z cross-section

### Multiport scattering scope
- `Simulation.compute_waveguide_s_matrix(...)` is no longer limited to exactly
  two ports
- The current sound scope is now:
  - x-directed guide network
  - ports lie on the two x-end boundary planes
  - `+x` ports share the left boundary plane
  - `-x` ports share the right boundary plane
  - apertures on the same boundary must be disjoint in y/z
- This supports >2 ports when they are physically meaningful boundary apertures
  (for example, multiple parallel guides or disjoint branches entering the same
  left/right boundary planes)

### Low-level assembly
- `extract_waveguide_s_matrix(...)` now assembles an `N x N` matrix for any
  number of ports in the supported boundary-aperture model
- The assembly remains one-driven-port-at-a-time
- Passive ports are measured, not concurrently driven

### API/run separation
- `Simulation.run()` still rejects multi-waveguide-port use and points users to
  `compute_waveguide_s_matrix()`
- This keeps single-run state/time-series semantics clean while allowing a
  dedicated scattering workflow for multiport waveguides

## Why this is the right generalization

The previous 2-port implementation was correct but still tied to a single
full-cross-section guide. The important realization was that the low-level
`WaveguidePort` model already supports arbitrary `y_slice/z_slice` apertures.
That means >2 ports become physically meaningful if they are interpreted as
**disjoint boundary apertures**, not arbitrary internal full-plane ports.

So the branch now supports the smallest correctness-first generalization:
- not arbitrary 3D branch normals
- not arbitrary internal interfaces
- but yes to multiple disjoint x-normal apertures on the left/right boundary
  planes, assembled one driven port at a time

## Verification

### New multiport behavior
- `pytest -q tests/test_api.py::test_waveguide_four_port_parallel_guides_through_api`
  - passed
- `pytest -q tests/test_api.py::test_waveguide_two_port_s_matrix_through_api tests/test_api.py::test_waveguide_two_port_s_matrix_rejects_bad_orientation tests/test_simulation.py::test_extract_waveguide_s_matrix_two_port_reciprocity`
  - passed

### Full targeted regression sweep
- `pytest -q tests/test_waveguide_port.py tests/test_api.py::test_waveguide_port_through_api tests/test_api.py::test_waveguide_port_geometry_changes_response tests/test_api.py::test_waveguide_port_rejects_unsupported_configuration tests/test_api.py::test_waveguide_two_port_s_matrix_through_api tests/test_api.py::test_waveguide_two_port_s_matrix_rejects_bad_orientation tests/test_api.py::test_waveguide_four_port_parallel_guides_through_api tests/test_api.py::test_waveguide_sparams_default_output tests/test_api.py::test_waveguide_sparams_deembedded_planes tests/test_api.py::test_waveguide_sparams_source_to_probe_preset tests/test_api.py::test_waveguide_sparams_report_snapped_planes_for_non_aligned_input tests/test_api.py::test_waveguide_port_allows_near_boundary_input_when_snapped_planes_fit tests/test_api.py::test_waveguide_port_rejects_invalid_reporting_plane tests/test_api.py::test_waveguide_port_rejects_measurement_planes_outside_domain tests/test_api.py::test_waveguide_port_rejects_nonfinite_numeric_inputs tests/test_api.py::test_waveguide_port_rejects_invalid_integer_like_inputs tests/test_api.py::test_waveguide_port_rejects_invalid_freq_arrays tests/test_simulation.py::test_compiled_runner_waveguide_port_matches_manual_loop tests/test_simulation.py::test_extract_waveguide_s_matrix_two_port_reciprocity tests/test_sparam.py tests/test_dft_probes.py tests/test_physics.py::test_two_port_reciprocity tests/test_verification.py::test_gradient_through_dft_plane`
  - 34 passed
- `python -m py_compile rfx/api.py rfx/simulation.py rfx/sources/waveguide_port.py rfx/probes/probes.py rfx/sources/__init__.py rfx/__init__.py tests/test_api.py tests/test_waveguide_port.py tests/test_simulation.py tests/test_dft_probes.py tests/test_sparam.py tests/test_physics.py tests/test_verification.py`
  - passed
- LSP diagnostics on modified files
  - 0 errors

## Remaining limitations
- Ports are still x-normal only
- The model still assumes rectangular aperture-local modal bases
- Generic branch/junction networks whose ports are not representable as disjoint
  x-boundary apertures remain a future extension
- Common DFT/port defaults remain conservative (rectangular) outside the
  waveguide path

## Next suggested step
1. decide whether to keep the boundary-aperture model explicit as the supported
   “multiport waveguide” scope, or extend to arbitrary port-normal axes and
   richer branch/junction geometries
2. only after that, revisit default window policies for the common DFT/port
   paths on a source-type-by-source-type basis
