# Quick Start

This guide walks through your first rfx simulation in 15 minutes.

## Minimal Workflow (5 lines)

A patch antenna simulation in five lines -- define the domain, add geometry, add a port, and run:

```python
import rfx

sim = rfx.Simulation(freq_max=4e9, domain=(0.08, 0.06, 0.02), boundary="cpml")
sim.add(rfx.Box((-19e-3, -14.5e-3, 0.8e-3), (19e-3, 14.5e-3, 0.8e-3)), material="pec")   # patch
sim.add(rfx.Box((-30e-3, -25e-3, 0), (30e-3, 25e-3, 1.6e-3)), material="fr4")             # substrate
sim.add_port(position=(5e-3, 0, 0.8e-3), component="ez")
result = sim.run(until_decay=1e-3)
```

`Box(corner_lo, corner_hi)` defines axis-aligned boxes in metres. Library materials
like `"pec"` and `"fr4"` are built in -- no manual definition needed. Use
`add_port()` for excitation with S-parameter extraction or `add_source()` for
unloaded excitation (e.g. resonance studies). Add `add_probe()` calls to record
time-domain field data at specific locations.

## 1. PEC Cavity Resonance

The simplest FDTD simulation: a rectangular PEC (perfect electric conductor) cavity.

```python
import numpy as np
import matplotlib.pyplot as plt
from rfx import Grid, GaussianPulse
from rfx.grid import C0
from rfx.core.yee import init_state, init_materials, update_e, update_h
from rfx.boundaries.pec import apply_pec

# Cavity dimensions
a, b, d = 0.10, 0.10, 0.05  # meters

# Analytical TM110 resonant frequency
f_analytical = (C0 / 2) * np.sqrt((1/a)**2 + (1/b)**2)

# Create grid (no CPML — PEC on all walls)
grid = Grid(freq_max=5e9, domain=(a, b, d), dx=0.001, cpml_layers=0)
state = init_state(grid.shape)
materials = init_materials(grid.shape)

# Source and probe
pulse = GaussianPulse(f0=f_analytical, bandwidth=0.8)
src = (grid.nx // 3, grid.ny // 3, grid.nz // 2)
probe = (2 * grid.nx // 3, 2 * grid.ny // 3, grid.nz // 2)

# Run
n_steps = grid.num_timesteps(num_periods=80)
ts = np.zeros(n_steps)
for n in range(n_steps):
    t = n * grid.dt
    state = update_h(state, materials, grid.dt, grid.dx)
    state = update_e(state, materials, grid.dt, grid.dx)
    state = apply_pec(state)
    state = state._replace(ez=state.ez.at[src].add(pulse(t)))
    ts[n] = float(state.ez[probe])

# FFT → find resonance
spectrum = np.abs(np.fft.rfft(ts))
freqs = np.fft.rfftfreq(len(ts), d=grid.dt)
print(f"Analytical: {f_analytical/1e9:.4f} GHz")
```

## 2. Using the High-Level API

For most workflows, use the `Simulation` builder:

```python
from rfx import Simulation, Box, GaussianPulse

sim = Simulation(
    freq_max=5e9,
    domain=(0.10, 0.04, 0.02),
    boundary="cpml",
    cpml_layers=8,
)

# Add geometry
sim.add_material("dielectric", eps_r=4.0)
sim.add(Box((0.03, 0.0, 0.0), (0.05, 0.04, 0.02)), material="dielectric")

# Add port and run
sim.add_port(
    position=(0.01, 0.02, 0.01),
    component="ez",
    pulse=GaussianPulse(f0=3e9),
)
result = sim.run(n_steps=500, compute_s_params=True)

# S-parameters are automatically extracted
s11 = result.s_params[0, 0, :]  # (n_freqs,) complex
print(f"|S11| mean: {np.mean(np.abs(s11)):.3f}")
```

## 3. Plotting

```python
from rfx import plot_s_params, plot_time_series

plot_s_params(result)     # S11 magnitude and phase vs frequency
plot_time_series(result)  # Time-domain probe signal
plt.show()
```

## Next Steps

- [Waveguide Ports](waveguide_ports.md) — multi-port S-matrix extraction
- [Inverse Design](inverse_design.md) — optimize structures with jax.grad
- [Far-Field & RCS](farfield_rcs.md) — antenna patterns and radar cross section
