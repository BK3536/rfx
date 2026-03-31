# Simulation API Reference

The `Simulation` class is the primary entry point for rfx. It provides a declarative builder pattern for configuring and running FDTD simulations.

## Creating a Simulation

```python
from rfx import Simulation

sim = Simulation(
    freq_max=10e9,          # Maximum frequency of interest (Hz)
    domain=(0.12, 0.04, 0.02),  # Physical size (x, y, z) in meters
    boundary="cpml",        # "cpml" or "pec"
    cpml_layers=10,         # Number of CPML absorber cells
    dx=0.002,               # Cell size (auto-calculated if omitted)
    mode="3d",              # "3d", "2d_tmz", or "2d_tez"
)
```

## Materials

### Built-in Library (11 materials)

```python
sim.add(Box(...), material="vacuum")     # eps_r=1
sim.add(Box(...), material="FR4")        # eps_r=4.4, tan_d=0.02
sim.add(Box(...), material="alumina")    # eps_r=9.8
sim.add(Box(...), material="silicon")    # eps_r=11.7
sim.add(Box(...), material="copper")     # sigma=5.8e7 S/m
sim.add(Box(...), material="pec")        # Perfect electric conductor
```

Full list: `vacuum`, `air`, `FR4`, `Rogers4003C`, `alumina`, `silicon`, `PTFE`, `copper`, `aluminum`, `pec`, `water_20C`

### Custom Materials

```python
sim.add_material("my_dielectric", eps_r=6.0, sigma=0.01)
sim.add_material("ferrite", eps_r=12.0, mu_r=100.0)
```

### Dispersive Materials

```python
from rfx import DebyePole, LorentzPole

sim.add_material("water", eps_r=5.0, debye_poles=[DebyePole(eps_s=80, tau=9.4e-12)])
sim.add_material("plasma", lorentz_poles=[LorentzPole(eps_s=1, omega_0=0, delta=1e9, omega_p=5e10)])
```

## Geometry (CSG)

```python
from rfx import Box, Sphere, Cylinder

sim.add(Box((0.01, 0.01, 0.01), (0.05, 0.03, 0.02)), material="FR4")
sim.add(Sphere((0.06, 0.02, 0.01), radius=0.005), material="alumina")
sim.add(Cylinder((0.08, 0.02, 0.01), radius=0.003, height=0.01, axis="z"), material="copper")
```

Boolean operations: `union(a, b)`, `difference(a, b)`, `intersection(a, b)`

## Sources

```python
from rfx import GaussianPulse, CWSource, CustomWaveform

# Lumped port (S-parameter extraction)
sim.add_port(position=(0.01, 0.02, 0.01), component="ez",
             pulse=GaussianPulse(f0=5e9))

# CW source (steady-state analysis)
sim.add_port(position=(0.01, 0.02, 0.01), component="ez",
             pulse=CWSource(f0=5e9, ramp_steps=100))
```

## Running

```python
# Fixed number of steps
result = sim.run(n_steps=500)

# Auto-determined by field decay
result = sim.run(until_decay=1e-3)

# With S-parameter extraction
result = sim.run(n_steps=500, compute_s_params=True)
```

## Result Object

```python
result.state          # Final FDTD field state
result.time_series    # (n_steps, n_probes) probe data
result.s_params       # (n_ports, n_ports, n_freqs) S-matrix
result.s_param_freqs  # (n_freqs,) frequency array
result.snapshots      # Field snapshots (if requested)
```
