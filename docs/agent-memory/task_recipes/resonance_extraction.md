# Recipe: Resonance frequency & Q extraction (Harminv)

## When to use

Any task that asks "what are the resonances / modes / Q factors of
this structure?" — ring resonators, cavities, photonic crystal
defects, dielectric spheres, waveguide modal beat frequencies, etc.

## The right primitive

**`rfx.harminv.harminv`** — Filter-diagonalization method. Extracts
complex-frequency modes from a short ringdown time series. Matches
Meep's `mp.Harminv` (same underlying algorithm).

```python
from rfx.harminv import harminv

modes = harminv(signal, dt, fmin_hz, fmax_hz)
for m in modes:
    print(m.freq, m.Q, m.amplitude, m.decay)
```

## Canonical pattern

```python
from rfx import Simulation
from rfx.sources.sources import ModulatedGaussian
from rfx.harminv import harminv

# 1. Broadband Gaussian pulse excitation (must overlap the expected
#    resonance band). Use ModulatedGaussian f0 = band center,
#    bandwidth = df/f0 ≥ (fmax - fmin)/fcen.
sim = Simulation(freq_max=..., domain=..., dx=..., boundary="upml",
                 cpml_layers=..., mode="2d_tmz")
sim.add(...)  # geometry (cavity / resonator)
sim.add_source(
    position=(src_x, src_y, src_z), component="ez",
    waveform=ModulatedGaussian(f0=fcen_hz, bandwidth=bw,
                               cutoff=5.0 / math.sqrt(2)),
)
sim.add_probe(position=(probe_x, probe_y, probe_z), component="ez")

# 2. Run long enough for (a) source to decay and (b) ringdown to sample
#    many cycles of the lowest-Q mode. Rule of thumb: at least
#    ~10*Q cycles after the source dies.
result = sim.run(n_steps=N)

# 3. Apply Harminv to the RINGDOWN portion only (skip the source-active
#    prefix — filter-diagonalization assumes free ringing).
ts = np.array(result.time_series).ravel()
dt = float(result.dt)
skip = int(len(ts) * 0.4)   # skip first 40%, keep ringdown
signal = ts[skip:]

modes = harminv(signal, dt, fmin_hz, fmax_hz)
modes_good = [(m.freq, m.Q, m.amplitude) for m in modes
              if m.Q > 1 and m.amplitude > 1e-10]
```

## Matching Meep for cross-validation

Meep's equivalent:
```python
import meep as mp
h = mp.Harminv(mp.Ez, mp.Vector3(probe_x, probe_y), fcen, df)
sim.run(mp.at_every(0.0, h), until_after_sources=N)
meep_modes = h.modes  # list with .freq, .Q, .amp, .decay
```

**Critical matching notes** (see also `source_matching_meep.md`):
- Meep `fcen`, `df` ↔ rfx `f0`, `bandwidth`. The conversion is
  `df_meep = f0_rfx * bandwidth_rfx * a_meep / C0` (Meep uses
  *absolute* df in Meep frequency units, rfx uses fractional).
- Meep Harminv's `fcen`, `df` specify the SEARCH band, not the source.
  Use fcen = (fmin+fmax)/2, df = fmax - fmin.
- Meep frequencies come out in Meep units (multiply by `C0/a` to get Hz).

## Pitfalls

1. **Source still active when Harminv runs**: Harminv assumes a free
   ringdown. Meep's `after_sources` waits for the source envelope to
   fall; in rfx you must manually skip the source portion of the time
   series (`signal = ts[skip:]`).

2. **Too few ringdown cycles**: Harminv needs many cycles per mode to
   resolve it. For a mode of frequency f and Q, the ringdown time is
   ~Q/(π·f). Run long enough to capture at least 5-10 Q-cycles.

3. **Cutoff filtering**: rfx's `ModulatedGaussian(cutoff=5/√2)` matches
   Meep's default Gaussian truncation. Changing cutoff changes the
   source bandwidth.

4. **Q from Meep is `2π·f·τ`**, same as rfx. No conversion needed.

## Canonical example

- `examples/crossval/02_ring_resonator.py` — ring resonator WGM
  extraction via Harminv, rfx vs Meep. Includes source-skip logic and
  mode matching across the two simulators.

## References

- Mandelshtam & Taylor, *J. Chem. Phys.* 107, 6756 (1997) — harmonic
  inversion algorithm
- Meep docs: `mp.Harminv`
- rfx source: `rfx/harminv.py`
