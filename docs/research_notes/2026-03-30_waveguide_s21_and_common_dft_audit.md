# 2026-03-30: Waveguide S21 stabilized; waveguide multiport rejected loudly; common DFT/port windowing groundwork added

## What changed

### Waveguide S21 stabilization
- `rfx.sources.waveguide_port.update_waveguide_port_probe(...)` now applies a
  tapered DFT window when `dft_total_steps` is provided
- The current default for waveguide-port accumulation is Tukey (`alpha=0.25`)
- In the current TE10 propagation regression, above-cutoff `|S21|` improved
  from a highly unstable spread (~0.19…5.40) to a much tighter band
  (~0.87…1.17)

### Waveguide multiport correctness guard
- `rfx.api.Simulation.add_waveguide_port(...)` now rejects multiple waveguide
  ports
- `rfx.simulation.run(...)` also rejects multiple simultaneous waveguide-port
  configs defensively
- This is intentional: the current waveguide port model is still a single-port
  modal source/monitor path, not a true multiport scattering experiment

### Common DFT / port groundwork
- `DFTProbe`, `DFTPlaneProbe`, and `SParamProbe` now accept shared streaming DFT
  window metadata (`total_steps`, `window`, `window_alpha`)
- The common probe/port path currently defaults to rectangular accumulation so
  existing semantics stay unchanged unless a caller explicitly opts in
- `extract_s_matrix(...)` now threads `dft_total_steps=n_steps`, so the common
  machinery is in place for future opt-in spectral conditioning

## Why this design

The correctness-first requirement ruled out two tempting shortcuts:
1. pretending the old multi-waveguide-port path produced a valid multiport
   S-matrix just because the code could loop over several port configs
2. forcing the same windowing default onto every DFT/port path before proving
   it preserved existing lumped-port passivity / exact DFT parity expectations

So the current result is intentionally asymmetric:
- waveguide gets the Tukey-window fix because it directly solves the observed
  `S21` instability
- common DFT/port paths get the infrastructure, but keep rectangular default
  until a source-type-specific validation policy is designed

## Verification

### Waveguide correctness / fidelity
- `pytest -q tests/test_waveguide_port.py::test_waveguide_port_propagation -s`
  - TE10 above-cutoff `|S21|`: mean ~0.984, min ~0.870, max ~1.168
- `pytest -q tests/test_waveguide_port.py::test_te10_below_cutoff_evanescent -s`
  - below-cutoff `|S21|`: mean ~0.121 (-18.3 dB)
- `pytest -q tests/test_waveguide_port.py tests/test_api.py::test_waveguide_port_through_api tests/test_api.py::test_waveguide_port_geometry_changes_response tests/test_api.py::test_waveguide_port_rejects_unsupported_configuration tests/test_api.py::test_waveguide_sparams_default_output tests/test_api.py::test_waveguide_sparams_deembedded_planes tests/test_api.py::test_waveguide_sparams_source_to_probe_preset tests/test_api.py::test_waveguide_sparams_report_snapped_planes_for_non_aligned_input tests/test_api.py::test_waveguide_port_allows_near_boundary_input_when_snapped_planes_fit tests/test_api.py::test_waveguide_port_rejects_invalid_reporting_plane tests/test_api.py::test_waveguide_port_rejects_measurement_planes_outside_domain tests/test_api.py::test_waveguide_port_rejects_nonfinite_numeric_inputs tests/test_api.py::test_waveguide_port_rejects_invalid_integer_like_inputs tests/test_api.py::test_waveguide_port_rejects_invalid_freq_arrays tests/test_simulation.py::test_compiled_runner_waveguide_port_matches_manual_loop tests/test_simulation.py::test_compiled_runner_rejects_multiple_waveguide_ports`
  - 24 passed

### Common DFT / port paths
- `pytest -q tests/test_dft_probes.py tests/test_sparam.py tests/test_physics.py::test_two_port_reciprocity tests/test_simulation.py::test_compiled_runner_dft_plane_matches_manual_loop tests/test_verification.py::test_gradient_through_dft_plane`
  - 8 passed
- `python -m py_compile rfx/api.py rfx/simulation.py rfx/sources/waveguide_port.py rfx/probes/probes.py rfx/sources/__init__.py rfx/__init__.py tests/test_api.py tests/test_waveguide_port.py tests/test_simulation.py tests/test_dft_probes.py tests/test_sparam.py tests/test_physics.py tests/test_verification.py`
  - passed
- LSP diagnostics on modified files
  - 0 errors

## Interpretation

### What is now solid
- single-port waveguide modal S11/S21 extraction + de-embedding
- single-port waveguide S21 spectral stability is materially better
- lumped-port multiport S-matrix path remains theoretically cleaner because it
  already uses one-driven-port-at-a-time extraction

### What is intentionally blocked
- true waveguide multiport (>1 waveguide port in a single run)
- This remains unimplemented because a correct solution needs an explicit
  one-driven-port-at-a-time waveguide calibration flow and a deliberate
  `N x N` assembly step

### What remains approximate / future work
- common DFT/port spectral conditioning is not yet turned on by default
- measurement fidelity still depends on probe spacing, window choice, and CPML
  residuals

## Next suggested step
1. implement a true waveguide multiport scattering workflow (one driven port at
   a time, explicit matrix assembly)
2. only after that, decide which common DFT/port paths should opt into windowed
   accumulation by default and which should remain rectangular
