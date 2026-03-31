# Advanced Features

## Dispersive Materials

### Debye Model (water, biological tissue)

```python
from rfx import DebyePole

sim.add_material("water", eps_r=5.0, debye_poles=[
    DebyePole(eps_s=80.0, tau=9.4e-12)  # Single-pole Debye
])
```

### Lorentz/Drude Model (metals, plasmas)

```python
from rfx import LorentzPole, drude_pole

# Drude metal
sim.add_material("gold", lorentz_poles=[
    drude_pole(omega_p=1.37e16, gamma=4.05e13)
])

# Lorentz resonance
sim.add_material("resonant", lorentz_poles=[
    LorentzPole(eps_s=2.0, omega_0=2*np.pi*5e9, delta=1e8, omega_p=2*np.pi*3e9)
])
```

## Magnetic Materials

```python
sim.add_material("ferrite", eps_r=12.0, mu_r=100.0)
```

Validated: Fresnel reflection |R| = 3.1% error, phase velocity 0.0% error for mu_r=4.

## CFS-CPML (Enhanced Absorption)

Standard CPML struggles with evanescent waves near cutoff. CFS-CPML adds kappa stretching:

```python
from rfx.boundaries.cpml import init_cpml

# Standard CPML (default)
cp, cs = init_cpml(grid, kappa_max=1.0)

# CFS-CPML for better evanescent absorption (1.6x improvement)
cp, cs = init_cpml(grid, kappa_max=7.0)
```

## Subpixel Smoothing

Anisotropic permittivity averaging at dielectric interfaces for second-order convergence:

```python
result = sim.run(n_steps=500, subpixel_smoothing=True)
```

Uses arithmetic mean (parallel to interface) and harmonic mean (perpendicular) per Farjadpour et al. (2006). Gives ~1.2x error reduction at equivalent resolution.

## Field Decay Convergence

Auto-determine simulation length based on field energy decay:

```python
# Stop when |field|² < 0.1% of peak
result = sim.run(until_decay=1e-3)

# With min/max step limits
result = sim.run(until_decay=1e-3, decay_min_steps=200, decay_max_steps=10000)
```

## Touchstone I/O

Import/export S-parameters for interoperability with commercial tools:

```python
from rfx import write_touchstone, read_touchstone

# Export
write_touchstone("device.s2p", freqs, s_matrix, z0=50.0, format="RI")

# Import
freqs, s_matrix, z0 = read_touchstone("device.s2p")
```

## HDF5 Checkpointing

Save and restore simulation state:

```python
from rfx import save_state, load_state, save_materials, load_materials

save_state("checkpoint.h5", result.state)
state = load_state("checkpoint.h5", grid.shape)

save_materials("materials.h5", materials)
materials = load_materials("materials.h5", grid.shape)
```

## 2D Modes

For thin structures (waveguides, transmission lines):

```python
sim = Simulation(freq_max=10e9, domain=(0.1, 0.04, 0.001),
                 boundary="cpml", mode="2d_tmz")  # or "2d_tez"
```

## GPU Acceleration

rfx runs on GPU automatically when JAX detects CUDA:

```python
import jax
print(jax.default_backend())  # "gpu" if available, "cpu" otherwise
```

No code changes needed. Same script runs on CPU or GPU transparently. Typical speedup: 10-50x on NVIDIA RTX 4090 / A6000.
