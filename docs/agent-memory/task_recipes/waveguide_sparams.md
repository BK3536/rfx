# Recipe: Waveguide S-parameters (TE10 mode decomposition)

## When to use

Multi-port S-parameters for **bounded rectangular waveguide** structures:
- dielectric-loaded sections
- discontinuities, irises, iris filters
- coaxial-waveguide transitions
- any 2N-port device in a rectangular waveguide

This recipe is for **GUIDED wave** measurements. For plane-wave R/T
through a slab in open space, see `rt_measurement.md` instead.

## The right primitives

- **`Simulation.add_waveguide_port(x_position, direction, mode, ...)`**
  — Each port acts as both source and receiver. rfx automatically
  launches the selected mode, records reflected + transmitted mode
  coefficients, and uses PEC walls on the y/z transverse axes.

- **`Simulation.compute_waveguide_s_matrix(normalize=True, num_periods=...)`**
  — Runs the sim once per port driving plus a vacuum reference run
  and returns an N×N×n_freqs complex S-matrix.

```python
sim = Simulation(
    freq_max=f_max,
    domain=(length, a_wg, b_wg),   # x-longitudinal waveguide
    dx=dx,
    boundary="cpml",
    cpml_layers=10,
)
sim.add_material("plug", eps_r=2.0)
sim.add(Box((plug_lo, 0, 0), (plug_hi, a_wg, b_wg)), material="plug")

sim.add_waveguide_port(
    x_position=0.003, direction="+x",
    mode=(1, 0), mode_type="TE",
    freqs=freqs, f0=fcen, bandwidth=bw,
    probe_offset=10, ref_offset=3,
    name="port1",
)
sim.add_waveguide_port(
    x_position=length - 0.003, direction="-x",
    mode=(1, 0), mode_type="TE",
    freqs=freqs, f0=fcen, bandwidth=bw,
    probe_offset=10, ref_offset=3,
    name="port2",
)

result = sim.compute_waveguide_s_matrix(num_periods=30.0, normalize=True)
S = np.asarray(result.s_params)   # (n_ports, n_ports, n_freqs)
S11 = S[0, 0, :]; S21 = S[1, 0, :]
```

## Critical parameters

| Parameter | Why it matters | Reasonable values |
|-----------|----------------|-------------------|
| `normalize=True` | Cancels Yee numerical-dispersion bias in |S21| | **Always True** for measurement |
| `num_periods` | Must be enough for ringdown + standing-wave decay, but **too large destabilizes** in rfx's current implementation | 20-40 typical; do NOT use 100+ |
| `f0`, `bandwidth` (per-port) | Default is `freq_max/2, bw=0.5` which may not cover your target band. Set explicitly to `(f_min+f_max)/2, (f_max-f_min)/fcen` | Band-matched |
| `probe_offset`, `ref_offset` | Distance in cells from port plane to probe and reference planes. Must be far from obstacles | 10, 3 typical |
| `a_wg`, `b_wg` | Waveguide cross-section. Below cutoff = no propagation | `f_min ≥ 1.3 · f_c` |
| Length | Ports must be `≥ probe_offset + 5` cells away from any obstacle | Empty guide + 2 × port_zone ≈ 80 mm min |

## Band selection

Choose `f_min` and `f_max` so that:
1. `f_min > 1.3 · f_cutoff` (TE10 cutoff `c/(2a)`)
2. No higher-order modes in the band — stay below `f_cutoff_TE20 = c/a`
3. The measurement band matches `f0 ± bandwidth·f0/2` of the source

## Known rfx baseline accuracy

rfx's `compute_waveguide_s_matrix(normalize=True)` has a **~5-10%
baseline error** relative to analytic for well-posed waveguide
problems. This is tracked by `tests/test_normalization.py::test_normalized_s21_straight_waveguide`
which asserts mean |S21| > 0.95 for an empty waveguide (i.e. up to 5%
deviation from the ideal 1.0). Crossvals should use a **15% PASS limit**
for individual S-params, not 5%.

## Meep counterpart (for crossval)

Meep's standard mode-decomposition flow:

```python
import meep as mp

src = mp.EigenModeSource(
    src=mp.GaussianSource(frequency=fcen, fwidth=df),
    center=..., size=...,
    direction=mp.X,
    eig_band=1,                # mode index (1 = fundamental)
    eig_match_freq=True,
)
sim = mp.Simulation(..., sources=[src])

mon_in = sim.add_mode_monitor(fcen, df, nfreq,
    mp.ModeRegion(center=..., size=...))
mon_out = sim.add_mode_monitor(fcen, df, nfreq,
    mp.ModeRegion(center=..., size=...))

# TWO RUNS (ref + device), then compute normalized S-params:
# Run 1 (ref):
sim.run(until_after_sources=mp.stop_when_fields_decayed(...))
c_ref_in = sim.get_eigenmode_coefficients(mon_in, [1]).alpha[0,:,0]
c_ref_out = sim.get_eigenmode_coefficients(mon_out, [1]).alpha[0,:,0]

sim.reset_meep()
# Run 2 (device with geometry):
# ... rebuild with geometry ...
c_dev_in  = sim.get_eigenmode_coefficients(mon_in, [1]).alpha   # [band,freq,dir]
c_dev_out = sim.get_eigenmode_coefficients(mon_out, [1]).alpha

S11 = c_dev_in[0, :, 1] / c_ref_in    # backward at in / fwd ref
S21 = c_dev_out[0, :, 0] / c_ref_out  # forward at out / fwd ref
```

**Meep pitfalls** (verified in crossval 09, 2026-04-10):

- **Band edges are noisy** — `get_eigenmode_coefficients` can produce
  non-physical |S| > 1 at frequencies near the source spectrum edge
  or near the waveguide cutoff. Narrow the comparison band to
  `[f_min + 0.25·Δ, f_max − 0.25·Δ]`.
- **Custom geometries are less validated** than the canonical taper
  in Meep's Mode Decomposition tutorial. A dielectric plug in a WR-90
  guide can trigger mode-matching artifacts that a smooth taper
  doesn't. For a clean Meep reference, imitate the tutorial example.

## Analytic reference

For a **guided-wave transmission line** (rectangular waveguide with a
dielectric section), use the Airy multi-reflection formula with
guided-wave impedance and propagation constant:

```python
def guided_beta(f, eps_r, a):
    k0 = 2*np.pi*f/C0
    kc = np.pi/a
    return np.sqrt(np.maximum(eps_r*k0**2 - kc**2, 0.0))

def guided_Z_TE(f, eps_r, a):
    eta = 377.0 / np.sqrt(eps_r)
    fc = C0 / (2*a*np.sqrt(eps_r))
    return eta / np.sqrt(1 - (fc/f)**2)

Z_vac = guided_Z_TE(f, 1.0, a_wg)
Z_d   = guided_Z_TE(f, eps_r, a_wg)
beta_d = guided_beta(f, eps_r, a_wg)
r12 = (Z_d - Z_vac) / (Z_d + Z_vac)
delta = beta_d * d_plug
e_2id = np.exp(2j*delta)
S11 = r12 * (1 - e_2id) / (1 - r12**2*e_2id)
S21 = (1 - r12**2) * np.exp(1j*delta) / (1 - r12**2*e_2id)
```

This is the 1D transmission-line reflection/transmission applied to
the single-mode TE10 guided wave.

## Canonical example

- `examples/crossval/09_mode_decomposition.py` — WR-90-style waveguide
  with dielectric plug, rfx vs Meep vs analytic. Demonstrates the
  recipe and documents the known edge-band artifacts.

## References

- Pozar, *Microwave Engineering* Ch. 4 (transmission-line S-params)
- Meep tutorial: *Mode Decomposition*
- rfx source: `rfx/api.py::compute_waveguide_s_matrix`,
  `rfx/ports/waveguide_port*.py`
- rfx tests: `tests/test_normalization.py` (baseline accuracy),
  `tests/test_conservation_laws.py` (reciprocity, unitarity)
