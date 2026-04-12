# Recipe: Patch-antenna resonance on a finite ground plane

## When to use

Any "what is the TM010 resonance of a rectangular microstrip patch
on substrate X?" question. Also applies to: microstrip filters,
coupled-line structures, and any planar metal geometry above a
finite PEC ground that you want to compare against Balanis TL or a
full-wave reference (OpenEMS, HFSS, Meep).

## The right primitives

- **Finite ground plane**: explicit `Box(material="pec")` placed
  **BELOW** the substrate, NOT `pec_faces={"z_lo"}`.
- **Substrate**: `Box(material="<dielectric>")` with physical z bounds.
- **Patch**: `Box(material="pec")` at the top of the substrate,
  1-cell thick in z.
- **Resonance extraction**: `add_source` (broadband Ez) +
  `add_probe` (field point) + `rfx.harminv.harminv` on the probe
  time series. See `resonance_extraction.md` for the harminv pattern.
- **Secondary check**: `add_port` (lumped) + `compute_s_params` for
  S11 passivity / local-dip confirmation.

## Canonical pattern

```python
from rfx import Simulation, Box
from rfx.sources.sources import GaussianPulse
from rfx.auto_config import smooth_grading
from rfx.harminv import harminv
import numpy as np

# Physical z-stack (bottom → top):
#   0 .. air_below                   : free-space air below the GP
#   [air_below - dz_sub, air_below]  : 1 PEC cell = finite ground plane
#   [air_below, air_below + h_sub]   : dielectric substrate
#   [+dz_sub]                        : 1 PEC cell = patch
#   ... air_above                    : free-space air above
dx = 1e-3                  # boundary cell size (CPML)
n_sub = 6                  # cells across substrate
dz_sub = h_sub / n_sub     # fine z in substrate
air_below = 12e-3          # ≥ λ/4 so bottom CPML clears the stack
air_above = 25e-3

raw_dz = np.concatenate([
    np.full(int(air_below/dx), dx),   # coarse air below
    np.full(1, dz_sub),                # ground PEC cell
    np.full(n_sub, dz_sub),            # substrate cells
    np.full(int(air_above/dx), dx),   # coarse air above
])
dz_profile = smooth_grading(raw_dz, max_ratio=1.3)

z_gnd_lo = air_below - dz_sub
z_gnd_hi = air_below
z_sub_lo, z_sub_hi = air_below, air_below + h_sub
z_patch_lo, z_patch_hi = z_sub_hi, z_sub_hi + dz_sub

sim = Simulation(
    freq_max=4e9, domain=(dom_x, dom_y, 0),
    dx=dx, dz_profile=dz_profile,
    boundary="cpml", cpml_layers=8,
    # NO pec_faces — the ground plane is a material Box below.
)
sim.add_material("fr4", eps_r=4.3)
sim.add(Box((gx_lo, gy_lo, z_gnd_lo), (gx_hi, gy_hi, z_gnd_hi)),
        material="pec")   # finite ground plane (60×55 mm, 1 cell)
sim.add(Box((gx_lo, gy_lo, z_sub_lo), (gx_hi, gy_hi, z_sub_hi)),
        material="fr4")
sim.add(Box((patch_x_lo, patch_y_lo, z_patch_lo),
            (patch_x_hi, patch_y_hi, z_patch_hi)), material="pec")
sim.add_source(position=(feed_x, feed_y, z_sub_lo + dz_sub*2.5),
               component="ez",
               waveform=GaussianPulse(f0=f_design, bandwidth=1.2))
sim.add_probe(position=(dom_x/2+5e-3, dom_y/2+5e-3, z_sub_lo + dz_sub*2.5),
              component="ez")
result = sim.run(num_periods=60)

ts = np.asarray(result.time_series).ravel()
modes = harminv(ts[int(len(ts)*0.3):], float(result.dt), 1.5e9, 3.5e9)
good = [m for m in modes if m.Q > 2 and m.amplitude > 1e-8]
tm010 = max((m for m in good if 0.9*f_design <= m.freq <= 1.1*f_design),
            key=lambda m: m.amplitude)
```

## Pitfalls

1. **Never use `pec_faces={"z_lo"}` as an antenna ground plane.**
   `pec_faces` creates a boundary-face PEC that covers the ENTIRE
   domain face at that side — i.e., an *infinite* PEC floor across
   the full simulation volume. That turns your antenna into a cavity
   and shifts the resonance ~+8 % high. Always use an explicit
   finite-size PEC `Box` for the ground.

2. **Place the ground plane BELOW the substrate, not overlapping it.**
   A PEC `Box` at `z=[0, dz_sub]` that coexists with a substrate
   `Box` at `z=[0, h_sub]` uses `|=` accumulation of the `pec_mask`,
   which makes the first substrate cell PEC — silently thinning the
   substrate by 1 cell (1/n_sub). Put the ground at `z=[-dz_sub, 0]`
   (or at `[air_below - dz_sub, air_below]` if you shift everything
   up for bottom-CPML clearance).

3. **Leave ≥ λ/4 of air below the ground plane.** A finite-GP patch
   radiates off its edges into the space below as well as above. If
   the bottom CPML butts up against the ground plane (no clearance),
   the absorber sees evanescent fields and the PEC cell is near-
   boundary, triggering the preflight warning "source/probe inside
   CPML region". Use `air_below ≥ 12 mm` at 2.4 GHz (≈ λ/4 in air).

4. **rfx single-cell lumped-port S11 dip is shallow (<2 dB) for
   high-Q multilayer structures** due to parasitic cell reactance.
   Always use `harminv` on a probe time series for the frequency
   measurement, and use the S11 dip only as a passivity / local-dip
   secondary check. See `tests/test_crossval_comprehensive.py::
   TestLumpedPortCavity` and `waveguide_sparams.md`.

5. **OpenEMS reference: use `PML_8` + ≥ λ/2 margin.** A naive
   `MUR` + λ/4 margin OpenEMS reference caused an 8 % downward bias
   in crossval 12 (wall reflections enlarged the effective cavity).
   Switch to `FDTD.SetBoundaryCond(['PML_8']*6)` and
   `margin_mm ≥ 50 mm at 2.4 GHz`.

6. **Analytic TL (Balanis) is ~5 % approximate.** The Balanis
   transmission-line model for `f_res` gives a great ballpark but
   misses finite-GP and non-ideal fringing effects. Don't tune rfx
   to match TL to better than 2 %; do compare against a full-wave
   reference.

## Canonical examples

- `examples/crossval/05_patch_antenna.py` — 2.4 GHz rectangular patch
  on FR4, finite 60×55 mm GP, rfx Harminv + OpenEMS PML_8 crossval.
  Agreement 0.99 % (both within 2 % of Balanis TL).

## References

- Balanis, *Antenna Theory*, Ch. 14 (transmission-line model for
  rectangular patch antennas, ΔL fringing formula).
- OpenEMS `Simple_Patch_Antenna.py` tutorial (canonical open-source
  patch benchmark — but use PML, not MUR).
- rfx source: `rfx/api.py::add_port`, `rfx/harminv.py::harminv`.
