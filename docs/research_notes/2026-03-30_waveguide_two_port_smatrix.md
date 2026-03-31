# 2026-03-30: True two-end-port waveguide scattering added; common DFT defaults remain conservative

## What changed

### True 2-port waveguide scattering
- `WaveguidePort` / `WaveguidePortConfig` now carry an explicit `direction`
  (`+x` for a left/inward launch, `-x` for a right/inward launch)
- Added `extract_waveguide_port_waves(...)` for port-local incident/outgoing
  wave extraction at a shifted reference plane
- Added `extract_waveguide_s_matrix(...)` for **exactly two** waveguide ports
  using one-driven-port-at-a-time assembly
- Added high-level `Simulation.compute_waveguide_s_matrix(...)`
  - requires exactly two waveguide ports
  - requires a left `+x` port and a right `-x` port
  - returns `WaveguideSMatrixResult`
    - `s_params`
    - `freqs`
    - `port_names`
    - `port_directions`
    - `reference_planes`

### API / runner behavior
- `Simulation.add_waveguide_port(...)` now accepts `direction`
- The high-level API allows up to two full-cross-section waveguide ports
- `Simulation.run()` still rejects multi-waveguide-port use and redirects users
  to `compute_waveguide_s_matrix()` for the theory-correct 2-port workflow
- Low-level `rfx.simulation.run(...)` no longer blocks multiple waveguide-port
  configs, because the one-driven-port-at-a-time assembly helper now uses it
  legitimately with passive ports set to zero excitation

### Common DFT / port policy
- The common DFT/port windowing infrastructure remains in place, but the shared
  probe/port defaults stay rectangular
- This is intentional: waveguide needed the default Tukey fix because it had an
  observed pathology, but point DFT / lumped-port paths should not be globally
  reconditioned by default without source-type-specific validation

## Why this is theoretically sound

The new waveguide multiport path does **not** excite both ports at once.
Instead it follows the same theoretical structure as a standard scattering
experiment:
1. drive one port only
2. measure local incident/outgoing waves at both ports
3. fill one column of the S-matrix
4. repeat for the other driven port

This is the smallest sound implementation for the current straight-guide,
full-cross-section port model. More than two such ports would be geometrically
ambiguous in the current abstraction and are still out of scope.

## Verification

### New waveguide 2-port path
- `pytest -q tests/test_api.py::test_waveguide_two_port_s_matrix_through_api tests/test_simulation.py::test_extract_waveguide_s_matrix_two_port_reciprocity`
  - passed
- Reciprocity and transmission are nontrivial and stable enough for the current
  straight-guide test case

### Existing waveguide path still good
- `pytest -q tests/test_waveguide_port.py tests/test_api.py::test_waveguide_port_through_api tests/test_api.py::test_waveguide_port_geometry_changes_response tests/test_api.py::test_waveguide_port_rejects_unsupported_configuration tests/test_api.py::test_waveguide_sparams_default_output tests/test_api.py::test_waveguide_sparams_deembedded_planes tests/test_api.py::test_waveguide_sparams_source_to_probe_preset tests/test_api.py::test_waveguide_sparams_report_snapped_planes_for_non_aligned_input tests/test_api.py::test_waveguide_port_allows_near_boundary_input_when_snapped_planes_fit tests/test_api.py::test_waveguide_port_rejects_invalid_reporting_plane tests/test_api.py::test_waveguide_port_rejects_measurement_planes_outside_domain tests/test_api.py::test_waveguide_port_rejects_nonfinite_numeric_inputs tests/test_api.py::test_waveguide_port_rejects_invalid_integer_like_inputs tests/test_api.py::test_waveguide_port_rejects_invalid_freq_arrays tests/test_simulation.py::test_compiled_runner_waveguide_port_matches_manual_loop tests/test_simulation.py::test_extract_waveguide_s_matrix_two_port_reciprocity tests/test_sparam.py tests/test_dft_probes.py tests/test_physics.py::test_two_port_reciprocity tests/test_verification.py::test_gradient_through_dft_plane`
  - 32 passed
- `python -m py_compile rfx/api.py rfx/simulation.py rfx/sources/waveguide_port.py rfx/probes/probes.py rfx/sources/__init__.py rfx/__init__.py tests/test_api.py tests/test_waveguide_port.py tests/test_simulation.py tests/test_dft_probes.py tests/test_sparam.py tests/test_physics.py tests/test_verification.py`
  - passed
- LSP diagnostics on modified files
  - 0 errors

## Remaining limitations
- The current theory-correct waveguide scattering support is **2-port only**
- The guide abstraction is still full-cross-section rectangular waveguide with
  x-directed ports; branched/custom-aperture generic N-port waveguide networks
  remain a future design task
- Common DFT/port defaults are still rectangular outside the waveguide path,
  by design

## Next suggested step
1. decide whether to extend waveguide scattering to richer but still sound
   geometries (custom aperture / branch ports) or keep the 2-end-port model
   explicit
2. only then revisit default window policies for the common DFT/port paths on a
   source-type-by-source-type basis
