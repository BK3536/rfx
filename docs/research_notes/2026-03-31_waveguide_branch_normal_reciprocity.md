# 2026-03-31: Mixed-normal waveguide reciprocity fixed for branch/junction ports

## Problem
The partial branch/junction-normal port work could produce non-reciprocal
mixed-axis waveguide S-matrices even though the underlying FDTD system is
linear and reciprocal.

Two concrete failure modes were present:
1. **Y-normal port parity mismatch**
   - The tangential sampling plane for a `y`-normal port is stored in physical
     `(x, z)` array order.
   - That plane ordering is left-handed with respect to `+y`, so reusing the
     x-normal H-profile sign convention made the modal current inner product
     inconsistent with +y-directed power.
   - Result: branch/junction couplings involving `±y` ports could violate
     reciprocity badly.
2. **High-level CPML-axis under-selection during port setup**
   - `Simulation._build_grid()` only considered already-registered waveguide
     ports.
   - When adding the first non-x-normal port to an existing waveguide setup,
     the temporary snapping/validation grid could omit the new normal axis from
     CPML padding.
   - Result: the public API could validate/snap mixed-normal ports against a
     grid that did not match the final scattering run.

## Fix
### 1) Y-normal local-frame correction
In `rfx/sources/waveguide_port.py`:
- keep the physical tangential field components for `y`-normal ports in
  `(Ex, Ez)` / `(Hx, Hz)` order
- flip the stored H mode profiles for `y`-normal ports so the modal-current
  projection measures **+y-directed power** correctly

This restores consistent incident/outgoing-wave extraction for `±y` ports
without changing the already-correct `x`- and `z`-normal conventions.

### 2) CPML axes from all active waveguide normals
In `rfx/api.py`:
- added `_waveguide_cpml_axes(...)`
- `_build_grid(...)` now uses the union of waveguide-port normal axes
- `add_waveguide_port(...)` passes the candidate port axis into the temporary
  snapping/validation grid via `extra_waveguide_axes`

This makes the public API use the same CPML-axis layout during validation,
config building, and final scattering assembly.

## Supported scope after the fix
The theory-correct public scope is now:
- axis-normal rectangular waveguide boundary ports
- one-driven-port-at-a-time S-matrix assembly
- mixed-axis boundary ports for branch/junction topologies, as long as the
  apertures are representable by the current rectangular port model and the
  geometry is provided explicitly (e.g. PEC T-junction walls)

The formulation is still not a claim of arbitrary oblique/internal waveguide
interfaces.

## Verification
### Mixed-normal regressions added
- `tests/test_api.py::test_waveguide_two_port_y_normal_s_matrix_through_api`
- `tests/test_api.py::test_waveguide_branch_junction_mixed_normals_reciprocal_through_api`
- `tests/test_simulation.py::test_extract_waveguide_s_matrix_mixed_normal_branch_reciprocity`

### Regression sweep
Command:
```bash
pytest -q tests/test_waveguide_port.py \
  tests/test_api.py::test_waveguide_port_through_api \
  tests/test_api.py::test_waveguide_port_geometry_changes_response \
  tests/test_api.py::test_waveguide_port_rejects_unsupported_configuration \
  tests/test_api.py::test_waveguide_two_port_s_matrix_through_api \
  tests/test_api.py::test_waveguide_multiport_same_direction_requires_shared_boundary_plane \
  tests/test_api.py::test_waveguide_four_port_parallel_guides_through_api \
  tests/test_api.py::test_waveguide_two_port_y_normal_s_matrix_through_api \
  tests/test_api.py::test_waveguide_branch_junction_mixed_normals_reciprocal_through_api \
  tests/test_api.py::test_waveguide_same_boundary_overlapping_apertures_reject \
  tests/test_api.py::test_waveguide_sparams_default_output \
  tests/test_api.py::test_waveguide_sparams_deembedded_planes \
  tests/test_api.py::test_waveguide_sparams_source_to_probe_preset \
  tests/test_api.py::test_waveguide_sparams_report_snapped_planes_for_non_aligned_input \
  tests/test_api.py::test_waveguide_port_allows_near_boundary_input_when_snapped_planes_fit \
  tests/test_api.py::test_waveguide_port_rejects_invalid_reporting_plane \
  tests/test_api.py::test_waveguide_port_rejects_measurement_planes_outside_domain \
  tests/test_api.py::test_waveguide_port_rejects_nonfinite_numeric_inputs \
  tests/test_api.py::test_waveguide_port_rejects_invalid_integer_like_inputs \
  tests/test_api.py::test_waveguide_port_rejects_invalid_freq_arrays \
  tests/test_simulation.py::test_compiled_runner_waveguide_port_matches_manual_loop \
  tests/test_simulation.py::test_extract_waveguide_s_matrix_two_port_reciprocity \
  tests/test_simulation.py::test_extract_waveguide_s_matrix_mixed_normal_branch_reciprocity \
  tests/test_sparam.py tests/test_dft_probes.py \
  tests/test_physics.py::test_two_port_reciprocity \
  tests/test_verification.py::test_gradient_through_dft_plane
```
Result:
- **38 passed**, 1 matplotlib warning

### Static verification
- `python -m py_compile rfx/api.py rfx/sources/waveguide_port.py tests/test_api.py tests/test_simulation.py`
  - passed
- LSP diagnostics on the modified files
  - 0 errors

## Practical note
For branch/junction scattering, the mixed-axis tests were most stable when the
run length was allowed to follow the grid-driven `num_periods=30` path rather
than forcing a short fixed step count. This is a measurement-fidelity issue,
not a reciprocity-formulation issue.
