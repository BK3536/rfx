# `rfx`

```text
██████╗ ███████╗██╗  ██╗
██╔══██╗██╔════╝╚██╗██╔╝
██████╔╝█████╗   ╚███╔╝
██╔══██╗██╔══╝   ██╔██╗
██║  ██║██║     ██╔╝ ██╗
╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝
```

JAX-based differentiable 3D FDTD electromagnetic simulator for RF and microwave engineering.

[![PyPI](https://img.shields.io/badge/PyPI-coming_soon-lightgrey)](#installation)
[![Tests](https://img.shields.io/badge/tests-placeholder-lightgrey)](#)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#license)

## What is rfx?

`rfx` is a JAX-native finite-difference time-domain (FDTD) simulator built for differentiable electromagnetic modeling in RF and microwave engineering. It supports end-to-end automatic differentiation, so gradients can flow through the entire simulation with `jax.grad` for optimization and inverse design workflows. The project targets practical RF and microwave regimes up to X-band (10 GHz), with a focus on simulation workflows that remain programmable, composable, and accelerator-friendly. It is positioned to be competitive with tools such as Meep and OpenEMS while adding native autodiff as a first-class capability.

## Key Features

- **3D and 2D FDTD** with CFS-CPML absorbing boundaries
- **Differentiable** simulation with `jax.grad` and `jax.checkpoint` for reverse-mode AD
- **Sources:** Gaussian pulse, CW, custom waveforms, TFSF plane wave, lumped ports, and waveguide ports
- **Materials:** dispersive (Debye, Lorentz/Drude), magnetic (`mu_r`), thin conductors, and subpixel smoothing
- **S-parameters:** N-port extraction, waveguide modal decomposition, and two-run normalization
- **Far-field:** NTFF, radiation patterns, and radar cross section (RCS) computation
- **Inverse design:** Adam optimizer with design regions
- **I/O:** Touchstone (`.sNp`) files and HDF5 checkpoints
- **GPU accelerated** via JAX
- **187 tests**, cross-validated against Meep and OpenEMS

## Installation

Install the base package from PyPI:

```bash
pip install rfx
```

For GPU-enabled workflows:

```bash
pip install rfx[gpu]
```

> **Note:** For CUDA or other accelerator backends, complete the appropriate
> [JAX GPU setup](https://jax.readthedocs.io/en/latest/installation.html)
> for your platform before running large simulations.

### GPU Performance (RTX 4090)

| Grid | Steps | GPU Time | Throughput |
|------|-------|----------|------------|
| 23³ | 200 | 0.087s | 28 Mcells/s |
| 33³ | 300 | 0.086s | 125 Mcells/s |
| 43³ | 400 | 0.103s | 310 Mcells/s |
| 63³ | 500 | 0.095s | **1,310 Mcells/s** |

Gradient computation (reverse-mode AD): **~0.31s** for all grid sizes on GPU.

## Quick Start

The example below creates a PEC-walled cavity with a dielectric slab, excites it with a lumped port, runs the simulation, and plots S11 versus frequency.

```python
import numpy as np
import matplotlib.pyplot as plt
from rfx import Simulation, Box, GaussianPulse

# 48 mm × 32 mm × 32 mm PEC cavity, simulate up to 10 GHz
sim = Simulation(freq_max=10e9, domain=(0.048, 0.032, 0.032), boundary="pec")

# Dielectric slab filling the centre third of the cavity
sim.add_material("slab", eps_r=2.2)
sim.add(Box((0.016, 0.008, 0.008), (0.032, 0.024, 0.024)), material="slab")

# Lumped port at one end, probe at the other
sim.add_port((0.006, 0.016, 0.016), "ez", impedance=50.0,
             waveform=GaussianPulse(f0=5e9, bandwidth=0.8))
sim.add_probe((0.042, 0.016, 0.016), "ez")

result = sim.run(num_periods=30)

# S11 versus frequency
freqs = result.freqs
s11_db = 20 * np.log10(np.abs(result.s_params[0, 0, :]) + 1e-12)
plt.plot(freqs / 1e9, s11_db)
plt.xlabel("Frequency [GHz]")
plt.ylabel("|S11| [dB]")
plt.title("PEC cavity with dielectric slab")
plt.show()
```

## Inverse Design Example

This example optimises the permittivity inside a design region to minimise reflected power at a lumped port.

```python
import numpy as np
import matplotlib.pyplot as plt
from rfx import Simulation, Box, GaussianPulse
from rfx.optimize import DesignRegion, optimize

sim = Simulation(freq_max=10e9, domain=(0.048, 0.032, 0.032), boundary="pec")
sim.add_port((0.006, 0.016, 0.016), "ez", impedance=50.0,
             waveform=GaussianPulse(f0=5e9, bandwidth=0.8))
sim.add_probe((0.042, 0.016, 0.016), "ez")

# Optimisable region between the port and the far wall
region = DesignRegion(
    corner_lo=(0.016, 0.008, 0.008),
    corner_hi=(0.032, 0.024, 0.024),
    eps_range=(1.0, 4.4),
)

def objective(result):
    # Minimise reflected power averaged over all S-param frequencies
    return float(np.mean(np.abs(result.s_params[0, 0, :]) ** 2))

opt_result = optimize(sim, region, objective, n_iters=40, lr=0.05)

plt.plot(opt_result.loss_history)
plt.xlabel("Iteration")
plt.ylabel("Mean |S11|²")
plt.title("Inverse design convergence")
plt.show()
```

## Documentation

Documentation is currently centered around in-code docstrings and runnable examples in the `examples/` directory. Start with the builder API examples and then inspect individual object and method docstrings for parameter details and simulation options.

## Citation

If you use `rfx` in academic work, please cite:

```bibtex
@software{kim_rfx,
  author       = {Byungkwan Kim},
  title        = {rfx: JAX-based differentiable 3D FDTD simulator},
  institution  = {REMI Lab, Chungnam National University},
  year         = {2026},
  url          = {https://github.com/BK3536/rfx},
  note         = {BibTeX placeholder -- update after formal release}
}
```

## License

`rfx` is released under the [MIT License](LICENSE).

## Acknowledgments

Developed by [Byungkwan Kim](https://github.com/BK3536) at the **Radar & ElectroMagnetic Intelligence (REMI) Laboratory**, Chungnam National University.

### AI Development Credits

This project was developed with significant contributions from AI coding assistants:

- **Claude** (Anthropic) — Architecture design, code review, quality management, validation strategy, physics verification
- **Codex** (OpenAI) — Feature implementation, test generation, documentation drafting
