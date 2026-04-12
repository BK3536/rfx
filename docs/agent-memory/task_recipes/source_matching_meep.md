# Recipe: Matching rfx source parameters with Meep (for crossval)

## Why this matters

Almost every rfx vs Meep crossval is sensitive to getting the source
waveform right. Meep and rfx have different conventions for frequency,
bandwidth, amplitude and time zero; if you mix them up the two
simulators respond at different frequencies and the comparison is
useless.

## Waveform equivalents

| Meep                             | rfx                           | Notes |
|----------------------------------|-------------------------------|-------|
| `mp.GaussianSource(frequency=fcen, fwidth=df)` | `ModulatedGaussian(f0=fcen_hz, bandwidth=df/fcen, cutoff=5/√2)` | bandwidth is FRACTIONAL in rfx, ABSOLUTE in Meep |
| `mp.ContinuousSource(frequency=f, width=w)` | not directly equivalent; use `ModulatedGaussian` with large cutoff or a custom source | Meep CW has a soft ramp; rfx doesn't ship this |
| `mp.CustomSource(src_func=...)`  | `sim.add_source(..., waveform=...)` with a callable | both accept arbitrary functions |

## Unit conversion (Meep → rfx)

Meep uses dimensionless units where length is in units of a meter
(picked by the user, commonly 1 µm or 1 cm). Conversions:

```python
a_meep = 0.01  # meters per Meep length unit (e.g. 1 cm)

# Length
length_meep = length_physical / a_meep

# Time
time_meep = time_physical * C0 / a_meep

# Frequency
freq_meep = freq_physical * a_meep / C0

# In Meep, fwidth has the same units as frequency, so:
df_meep = bw_fractional * fcen_meep     # bw_fractional = bandwidth / fcen
```

## Gaussian envelope timing

Both Meep `GaussianSource` and rfx `ModulatedGaussian` use a
differentiated Gaussian by default (suppresses the DC component).
Peak times and cutoffs:

| Quantity                | rfx ModulatedGaussian    | Meep GaussianSource     |
|-------------------------|--------------------------|-------------------------|
| Pulse width (envelope σ) | `1 / (π · f0 · bandwidth)` | `1 / (π · fwidth)`      |
| Peak time               | `cutoff · σ`             | `5 / fwidth`            |
| Default cutoff          | `5 / √2`                 | 5                        |

For matching, set **rfx `cutoff = 5 / √2`** and make sure
`bandwidth_rfx = fwidth_meep / fcen_meep`. This gives identical
time-domain pulses.

## Plane wave vs line vs point source

| Intent                       | Meep                                                  | rfx                                           |
|------------------------------|-------------------------------------------------------|-----------------------------------------------|
| Plane wave at normal incidence | `Source(..., size=Vector3(0, sy, 0))` line source at fixed x (must couple via periodic y) | `sim.add_tfsf_source(f0=..., bandwidth=..., polarization="ez", direction="+x")` |
| Line source spanning a waveguide | `Source(..., size=Vector3(0, wg_width))`              | loop `sim.add_source` over y cells, 1/N amplitude |
| Point source                 | `Source(..., size=Vector3())`                         | `sim.add_source(position=..., component=..., waveform=...)` |

**Prefer `add_tfsf_source` over a manual line source** for plane waves:
it's unidirectional, decouples the source from the scatterer, and is
rigorously compared with Meep results that use bidirectional line
sources (they agree after the ref-subtraction step).

## Verification workflow

When you are debugging a source mismatch:

1. Record the probe time series in BOTH simulators at a probe far
   from the source (so the source has had time to fully emerge).
2. Plot both on the same axes — in the time domain, not frequency.
3. Check envelope, peak time, and oscillation count match before
   looking at spectra.
4. If envelopes differ: fwidth/bandwidth mismatch. If peak times
   differ: cutoff mismatch. If oscillations differ: fcen mismatch.

## Canonical examples

- `examples/crossval/02_ring_resonator.py` — `ModulatedGaussian`
  vs `mp.GaussianSource`, matched for Harminv analysis.
- `examples/crossval/03_straight_waveguide_flux.py` — line source
  spanning a waveguide width, flux monitor comparison.
- `examples/crossval/08_material_dispersion.py` — TFSF plane wave
  (rfx) vs bidirectional Meep line source, with field-level
  reference subtraction.

## References

- Meep docs: *Source* class, *GaussianSource*
- rfx source: `rfx/sources/sources.py::ModulatedGaussian`
