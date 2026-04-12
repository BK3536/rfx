# Recipe: Side-by-side field snapshot comparison (rfx vs Meep)

## When to use

Any crossval where you want to **visually confirm that two simulators
see the same physics** before trusting the quantitative metrics.
Applies to: waveguide propagation, ring resonator modes, scattering
from dielectric slabs, antenna near-fields, basically everything.

**This is mandatory** — see the "field visual primary" memory:
*never* evaluate a crossval by scalar metrics (correlation, error
norms) alone. Read the field images and judge the physics visually
before looking at numbers.

## The right primitives

- **rfx**: `rfx.simulation.SnapshotSpec` + `sim.run(snapshot=snap)`
- **Meep**: `sim.get_array(component=mp.Ez, center=..., size=...)`
  called at specific simulation times (re-run a second time or use
  `sim.run(until=t)` with callback).

## Canonical pattern

```python
from rfx import Simulation, Box
from rfx.simulation import SnapshotSpec
import meep as mp
import numpy as np
import matplotlib.pyplot as plt

# Shared snapshot times (in physical seconds)
capture_ps = [0.05, 0.15, 0.30, 0.50]   # picoseconds, for example

# --- rfx: collect snapshots during the run ---
sim_rfx = Simulation(...)
sim_rfx.add(...); sim_rfx.add_source(...)
snap = SnapshotSpec(components=("ez",), slice_axis=2, slice_index=0)
res_rfx = sim_rfx.run(n_steps=n_steps, snapshot=snap)
ez_all = np.asarray(res_rfx.snapshots["ez"])  # shape (n_steps, nx, ny)

# Strip CPML padding (crossval plots should be INTERIOR only)
pad = res_rfx.grid.pad_x
n_dom_x = int(np.ceil(domain_x / dx)) + 1
n_dom_y = int(np.ceil(domain_y / dx)) + 1

dt_rfx = float(res_rfx.dt)
rfx_steps = [min(ez_all.shape[0]-1, int(t*1e-12/dt_rfx)) for t in capture_ps]

# --- Meep: second simulation to grab snapshots at matched times ---
sim_meep2 = mp.Simulation(cell_size=cell_meep, boundary_layers=pml_meep,
                          geometry=geo_meep, sources=src_meep,
                          resolution=resolution)
sim_meep2.init_sim()
meep_cap_times = [t * 1e-12 * C0 / a_meep for t in capture_ps]

# --- Plot side by side ---
fig, axes = plt.subplots(len(capture_ps), 3, figsize=(15, 4*len(capture_ps)))
for i, t_ps in enumerate(capture_ps):
    # rfx frame
    rf = ez_all[rfx_steps[i], pad:pad+n_dom_x, pad:pad+n_dom_y]

    # Meep frame — advance to target time
    remaining = meep_cap_times[i] - sim_meep2.meep_time()
    if remaining > 0:
        sim_meep2.run(until=remaining)
    ez_m = sim_meep2.get_array(center=mp.Vector3(), size=cell_meep,
                                component=mp.Ez)
    mf = ez_m[pml_cells:-pml_cells, :]

    # Match sizes (Meep and rfx may differ by ±1 cell)
    nc_x = min(rf.shape[0], mf.shape[0])
    nc_y = min(rf.shape[1], mf.shape[1])
    rf_c = rf[:nc_x, :nc_y]; mf_c = mf[:nc_x, :nc_y]

    vm = max(np.max(np.abs(rf_c)), np.max(np.abs(mf_c)), 1e-30) * 0.9
    axes[i, 0].imshow(rf_c.T, origin="lower", cmap="RdBu_r", vmin=-vm, vmax=vm)
    axes[i, 0].set_title(f"rfx Ez (t={t_ps:.2f}ps)")
    axes[i, 1].imshow(mf_c.T, origin="lower", cmap="RdBu_r", vmin=-vm, vmax=vm)
    axes[i, 1].set_title(f"Meep Ez (t={t_ps:.2f}ps)")
    axes[i, 2].imshow((rf_c - mf_c).T, origin="lower", cmap="bwr")
    axes[i, 2].set_title("rfx - Meep")

plt.savefig("XX_field_snapshots.png", dpi=150)
```

## Checklist

When reading the plots, check each frame:

- [ ] Pulse shape and position match in space
- [ ] Amplitude is of the same order of magnitude
- [ ] Wavefronts curve the same way (especially around scatterers)
- [ ] No spurious reflections from the CPML boundary
- [ ] No grid artifacts (checkerboard patterns = instability)
- [ ] Phase velocity looks right (pulse travels same distance per frame)

If any fail, **stop**. Quantitative metrics are meaningless if the
physics is visibly wrong.

## Pitfalls

1. **Meep needs a second simulation**: Meep doesn't have a built-in
   `SnapshotSpec` equivalent. You have to re-run and capture arrays
   at specific times, or use `sim.run(mp.at_every(t, save_fn))`.

2. **CPML padding strip**: rfx's `state.ez` includes the CPML
   padding cells. Always strip them (`ez[pad:pad+n_dom, ...]`)
   before plotting so you're comparing the *physical* domain.

3. **Unit conversion for times**: rfx is in SI seconds, Meep is in
   Meep time units (c=1). Conversion: `t_meep = t_physical * C0 / a_meep`.

4. **Color scale**: use a SYMMETRIC color scale (`vmin=-vm, vmax=vm`)
   for Ez, with the same `vm` for rfx and Meep frames in the same
   row. Otherwise the eye will latch onto spurious differences.

5. **Size mismatch**: rfx and Meep often differ by ±1 cell in each
   direction due to rounding. Crop to `min(shape)` before diffing.

## Canonical examples

- `examples/crossval/01_field_progression_review.py` — waveguide
  bend, 4 time points, rfx vs Meep side-by-side.
- `examples/crossval/02_deep_field_diagnostic.py` — finer-grained
  diagnostic with zoomed panels.
- `examples/crossval/02_ring_resonator.py` — narrowband mode
  pattern + broadband transient snapshots.
- `examples/crossval/03_straight_waveguide_flux.py` — straight
  waveguide propagation.
- `examples/crossval/08_field_snapshots.py` — 1D line-cut version
  (Ez(x) along beam axis) instead of full 2D.

## References

- rfx source: `rfx/simulation.py::SnapshotSpec`
- Meep docs: *get_array* method
- Memory: `feedback_field_visual_primary.md`
