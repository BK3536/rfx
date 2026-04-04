# Sources and Ports

`rfx` distinguishes between:

- **sources** — excitation without impedance loading
- **ports** — excitation + measurement against a reference impedance

## Recommended rule of thumb

- Use **`add_source()`** for resonance studies and mode finding
- Use **`add_port()`** for S-parameters
- Use **`add_waveguide_port()`** for modal waveguide workflows

## Waveforms

Common waveform types:

- `GaussianPulse`
- `ModulatedGaussian`
- `CWSource`
- `CustomWaveform`

Example:

```python
from rfx import GaussianPulse, ModulatedGaussian

wideband = GaussianPulse(f0=2.4e9, bandwidth=0.8)
zero_dc  = ModulatedGaussian(f0=2.4e9, bandwidth=0.6)
```

`GaussianPulse` is the current high-level default when `waveform=None`.  
For resonance-focused source runs, `ModulatedGaussian` is often the better
explicit choice because it has zero DC content.

## Soft source

```python
sim.add_source((0.02, 0.02, 0.01), "ez", waveform=GaussianPulse(f0=3e9))
```

Best for:

- cavity resonance
- patch resonance
- general unloaded excitation

## Polarized source

```python
sim.add_polarized_source((0.02, 0.02, 0.01), polarization="rhcp")
sim.add_polarized_source((0.02, 0.02, 0.01), polarization=(1.0, 1j))
```

Useful for:

- circular polarization studies
- antenna polarization tests
- Jones-vector driven excitation

## Lumped port

```python
sim.add_port(
    position=(0.01, 0.02, 0.01),
    component="ez",
    impedance=50.0,
    waveform=GaussianPulse(f0=5e9),
)
```

Use this when you want:

- simple S11/S21 extraction
- a reference impedance
- multi-port circuit-style workflows

## Wire port

```python
sim.add_port(
    position=(0.01, 0.02, 0.0),
    component="ez",
    impedance=50.0,
    extent=0.0016,
    waveform=GaussianPulse(f0=2.4e9),
)
```

This is the better choice for:

- conductor-to-conductor feeds
- probe-feed style excitation
- ports spanning multiple cells

## Waveguide port

```python
sim.add_waveguide_port(
    x_position=0.01,
    y_range=(0.0, 0.023),
    z_range=(0.0, 0.010),
    mode=(1, 0),
    mode_type="TE",
    direction="+x",
    name="wg1",
)
```

Use for:

- TE/TM modal excitation
- calibrated modal S-matrices
- rectangular waveguide benchmarks

## Practical caution

For patch and microstrip tutorials:

- current-source resonance workflows are currently the clearest validated path
- lumped-port feed examples are useful, but feed/port interpretation should be
  presented carefully

See also:

- [Quick Start](quickstart.md)
- [Simulation API](simulation_api.md)
- [Validation](validation.md)
