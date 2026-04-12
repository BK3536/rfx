# Recipe: Far-field radiation pattern (antenna, scatterer)

## When to use

Any "what does this antenna / aperture / scatterer radiate at infinity"
question: antenna gain, directivity, beam patterns, RCS of a 3D
scatterer, NTFF (near-to-far-field transformation) in general.

## The right primitives

- **`Simulation.add_ntff_box`** — defines a closed Huygens surface
  that integrates tangential E, H on-the-fly during the simulation.
- **`rfx.farfield.compute_far_field`** — post-process the NTFF DFT
  data to get complex E_theta, E_phi at a list of angles.
- **`rfx.farfield.directivity`** — directivity (dBi) for a single
  frequency from a far-field result.
- **`rfx.farfield.radiation_pattern`** — normalized power pattern in
  dB for plotting.

## Canonical pattern

```python
from rfx import Simulation
from rfx.farfield import compute_far_field, directivity, radiation_pattern
import jax.numpy as jnp

sim = Simulation(
    freq_max=f_max, domain=(dom_x, dom_y, dom_z), dx=dx,
    boundary="cpml", cpml_layers=cpml_layers, mode="3d",
)
sim.add_material("metal", eps_r=1, sigma=1e7)  # or similar
sim.add(...)  # antenna geometry

sim.add_waveguide_port(  # or other excitation
    x_position=feed_x, direction="+x", f0=f0, bandwidth=0.5,
    x_range=(...), y_range=(...), name="feed",
)

# NTFF box — keep well inside the CPML region (margin ≥ cpml+2 cells)
ntff_margin = (cpml_layers + 3) * dx
sim.add_ntff_box(
    corner_lo=(ntff_margin, ntff_margin, ntff_margin),
    corner_hi=(dom_x - ntff_margin, dom_y - ntff_margin, dom_z - ntff_margin),
    freqs=jnp.array([f0]),
)

# Run (use num_periods for CW-like measurement, or until_decay for pulse)
result = sim.run(num_periods=20)

# Compute far-field on an angular grid
theta = jnp.linspace(0.0, jnp.pi, 181)     # 0 = +z, pi = -z
phi   = jnp.array([0.0, jnp.pi / 2])       # E-plane and H-plane
ff = compute_far_field(
    result.ntff_data, result.ntff_box, result.grid, theta, phi,
)

D_dbi = float(directivity(ff)[0])          # scalar directivity in dBi
pat_dB = radiation_pattern(ff)             # (n_freqs, n_theta, n_phi) normalized dB
```

## Axis conventions

The NTFF uses spherical coordinates with:
- `theta = 0` along +z axis (zenith)
- `theta = π` along -z axis (nadir)
- `phi = 0` in the xz half-plane (x>0)
- `phi = π/2` in the yz half-plane (y>0)

For a rectangular waveguide / horn radiating toward +z: E-plane is
the plane containing the E-field vector (typically φ=π/2 if Ey is
excited, φ=0 if Ex is excited). H-plane is perpendicular.

## Pitfalls

1. **NTFF box inside CPML**: the NTFF integrand picks up absorbing
   currents if the box crosses into the CPML region. Always keep
   `ntff_margin ≥ (cpml_layers + 2) * dx`.

2. **Finite simulation time**: NTFF accumulates a DFT over the
   simulation. For a pulse excitation, run until the near-field has
   fully radiated past the NTFF box. For CW, integrate an integer
   number of periods.

3. **Frequency list**: `add_ntff_box` accepts a list of frequencies.
   Only those are accumulated. For a broadband directivity sweep,
   pass a linspace; for a single-frequency pattern, use
   `jnp.array([f0])`.

4. **Directivity vs gain**: `directivity()` returns peak directivity
   (ratio of max radiation intensity to isotropic). It does NOT
   account for feed line losses or mismatch — for "gain" as
   typically reported, combine with waveguide port S11.

5. **Known issue** (2026-04-10): far-field `dS` per-face has been
   fixed for non-uniform z, but audit item #9 is still pending for
   some edge cases. Check the latest `docs/agent-memory/index.md`
   known-issues list before claiming results.

6. **Directivity needs a dense φ grid**. `rfx.farfield.directivity`
   integrates `∫∫ |E|² sin θ dθ dφ` using `np.gradient` for the
   step sizes. If you sample only a few φ values (e.g. `[0, π/2]`
   for plotting two cuts), the integration weight is NOT 2π — it is
   just the span of your samples. Compute the directivity with a
   **dedicated dense φ grid** (≥ 36 samples from 0 to 2π, endpoint
   excluded) to get the correct normalisation. Confirmed in
   crossval 11 (2026-04-10).

7. **Pattern normalisation pitfall** — `max(arr.max(), 1e-30)` is
   a Python built-in which compares two scalars. If your far-field
   magnitudes are naturally very small (e.g. 1e-31 for a single-cell
   dipole source at 10 GHz), `1e-30 > arr.max()` wins and the
   normaliser becomes the epsilon, scaling the whole pattern by a
   wrong factor. Use `arr.max() if arr.max() > 0 else 1.0` instead.

8. **Meep coordinate convention**: Meep cells are centred at the
   origin. When porting an rfx geometry, subtract `dom/2` from every
   coordinate before giving it to Meep. In particular, NTFF margin:
   rfx expects a distance from the cell-face corner; Meep expects a
   signed position inside a cell centred at the origin. Getting this
   wrong puts the NTFF box outside the interior and `get_farfield`
   returns all zeros. Confirmed in crossval 11.

## Canonical examples

- `examples/crossval/ (deleted — was broken)` — open-ended rectangular
  waveguide horn, NTFF directivity vs analytical formula.
- `tests/test_ntff*.py` — smaller validation cases.

## References

- Taflove & Hagness Ch. 8 (near-to-far-field transformation)
- Balanis, *Antenna Theory*, Ch. 12 (aperture antennas)
- rfx source: `rfx/farfield.py` — `compute_far_field`,
  `directivity`, `radiation_pattern`
