# 2026-03-30: Waveguide modal S-parameters now support wave decomposition, de-embedding, and high-level calibrated API output

## What changed

### Waveguide modal extraction
- `rfx.sources.waveguide_port.WaveguidePortConfig` now stores `mode_type`
- `WaveguidePortConfig` now also stores `dx`, so extraction helpers know the
  physical positions of the source/reference/probe planes
- Added internal guided-wave helpers:
  - `_compute_beta(...)`
  - `_compute_mode_impedance(...)`
  - `_extract_traveling_waves(...)`
- Added reference-plane helpers:
  - `waveguide_plane_positions(...)`
  - `_shift_modal_waves(...)`
  - `extract_waveguide_sparams(...)`
- `extract_waveguide_s21(...)` now uses forward-wave amplitudes at the
  reference and probe planes instead of a plain modal-voltage ratio, and it
  now accepts optional `ref_shift` / `probe_shift` de-embedding distances
- `extract_waveguide_s11(...)` now computes the reflected-to-forward modal-wave
  ratio at the reference plane instead of the old lossless placeholder, with
  optional reference-plane shift support
- waveguide DFT accumulation now supports a tapered window (default Tukey when
  `dft_total_steps` is provided), which materially stabilizes pointwise `S21`
  in finite-time runs

### High-level API / result surface
- `Simulation.add_waveguide_port(...)` now accepts optional
  `reference_plane` / `probe_plane` absolute x positions in the physical
  domain
- `Simulation.add_waveguide_port(...)` also supports
  `calibration_preset='source_to_probe'`, which auto-selects the snapped source
  plane as the reporting reference plane
- High-level input hardening now rejects:
  - non-finite `x_position`, `bandwidth`, `amplitude`, `f0`,
    `reference_plane`, `probe_plane`
  - non-integer or non-positive `probe_offset` / `ref_offset`
  - invalid `mode` tuples
  - invalid `n_freqs`
  - explicit `freqs` arrays containing non-finite or non-positive values
- Requested x-positions are interpreted in physical metres, while the solver
  still samples on the nearest snapped grid planes; the result metadata now
  reports those actual snapped measurement planes explicitly
- Near-boundary validation now uses the **snapped** source plane rather than
  the raw requested x-position, so valid non-grid-aligned requests are not
  rejected spuriously
- Measurement-plane overflow is now rejected eagerly at the high-level API
  layer, before `run()`, using snapped source-plane metadata
- multiple simultaneous waveguide ports are now rejected loudly instead of
  silently producing non-physical concurrent-drive results
- `Result.waveguide_sparams[name]` now exposes calibrated high-level output
  with:
  - `freqs`
  - `s11`
  - `s21`
  - `calibration_preset` (`measured`, `explicit`, `source_to_probe`)
  - `source_plane`
  - `measured_reference_plane`
  - `measured_probe_plane`
  - `reference_plane`
  - `probe_plane`
- Existing raw `Result.waveguide_ports` output remains unchanged for backward
  compatibility

### Test coverage
- Added a synthetic regression that reconstructs known TE and TM traveling
  waves from modal V/I spectra and verifies exact recovery of both `S11` and
  `S21`
- Added a de-embedding regression that simulates measured planes offset from
  the desired reporting planes and verifies exact recovery after applying
  `ref_shift` / `probe_shift`
- Added high-level API regressions that verify:
  - default calibrated output matches raw low-level extraction
  - requested reporting planes are de-embedded correctly
  - `source_to_probe` preset auto-selects the snapped source plane correctly
  - non-finite scalar inputs are rejected early
  - invalid integer-like inputs are rejected early
  - invalid explicit frequency arrays are rejected early
  - non-grid-aligned requested planes report the correct snapped measurement
    metadata
  - near-boundary non-grid-aligned sources are validated against snapped
    measurement planes
  - invalid reporting planes are rejected early
- Added explicit regression coverage that multiple waveguide ports are rejected
  until true waveguide multiport scattering is implemented
- Kept the physical propagation regression and confirmed the API / compiled
  runner waveguide paths still agree with the updated extraction

## Why this step

The low-level waveguide port path already accumulated modal voltage/current DFTs,
but the public extraction helpers still lagged behind the intended model:
- `S21` used a voltage-only ratio, which over-counted standing-wave structure
- `S11` remained a placeholder based on `sqrt(1 - |S21|^2)` rather than true
  backward-wave recovery

This change aligns the extraction helpers with the existing modal V/I data
model and lifts the new calibration capability into the high-level API, which
is a more commercial-simulator-like surface than returning only raw probe
accumulators. It also makes the requested-vs-actual plane semantics explicit,
which matters for trustworthy calibration metadata.

The remaining `S21` issue turned out to be dominated by finite-time measurement
quality rather than the scattering formula itself. A Tukey-windowed modal DFT
substantially reduces the worst leakage/ripple in the current single-port
waveguide workflow.

## Verification

- `pytest -q tests/test_waveguide_port.py` → 9 passed
- `pytest -q tests/test_waveguide_port.py tests/test_api.py::test_waveguide_port_through_api tests/test_api.py::test_waveguide_port_geometry_changes_response tests/test_api.py::test_waveguide_sparams_default_output tests/test_api.py::test_waveguide_sparams_deembedded_planes tests/test_api.py::test_waveguide_sparams_source_to_probe_preset tests/test_api.py::test_waveguide_sparams_report_snapped_planes_for_non_aligned_input tests/test_api.py::test_waveguide_port_allows_near_boundary_input_when_snapped_planes_fit tests/test_api.py::test_waveguide_port_rejects_invalid_reporting_plane tests/test_api.py::test_waveguide_port_rejects_measurement_planes_outside_domain tests/test_api.py::test_waveguide_port_rejects_nonfinite_numeric_inputs tests/test_api.py::test_waveguide_port_rejects_invalid_integer_like_inputs tests/test_api.py::test_waveguide_port_rejects_invalid_freq_arrays tests/test_simulation.py::test_compiled_runner_waveguide_port_matches_manual_loop` → 22 passed
- `pytest -q tests/test_waveguide_port.py::test_waveguide_port_propagation tests/test_waveguide_port.py::test_te10_below_cutoff_evanescent tests/test_simulation.py::test_compiled_runner_rejects_multiple_waveguide_ports tests/test_api.py::test_waveguide_port_rejects_unsupported_configuration -s` → passed
- TE10 above-cutoff propagation improved from about `|S21| = 0.19…5.40` to
  about `|S21| = 0.87…1.17` in the current regression after Tukey-windowed DFT
- `python -m py_compile rfx/api.py rfx/sources/waveguide_port.py rfx/sources/__init__.py rfx/__init__.py tests/test_api.py tests/test_waveguide_port.py` → passed
- LSP diagnostics on modified files → 0 errors
- `ruff check ...` unavailable in this environment (`ruff: command not found`)

## Interpretation / current limitation

The algebraic extraction is physically grounded in forward/backward modal
waves, and the Tukey-windowed DFT materially improves the current single-port
waveguide `S21` fidelity. The remaining issue is no longer catastrophic
spiking, but residual measurement-quality limits from finite window length,
probe placement, and CPML residuals.

True waveguide multiport scattering is still **not implemented**. The code now
fails loudly for multiple simultaneous waveguide ports instead of pretending
that concurrent driven ports produce a valid multiport S-matrix.

## Next suggested step

If waveguide fidelity is the next priority, the follow-up should target the
measurement setup and UX around the now-correct algebra:
1. tune reference / probe spacing and excitation window for cleaner modal
   estimates
2. add richer calibration presets / automatic reference-plane selection beyond
   the first `source_to_probe` preset so users do not need to reason about raw
   offsets directly
3. consider de-duplicating the snapped overflow validation that now exists in
   both `add_waveguide_port(...)` and `run(...)`
4. if multiport waveguide scattering is needed, implement a true one-driven-port
   calibration flow and assemble an `N x N` waveguide S-matrix explicitly
