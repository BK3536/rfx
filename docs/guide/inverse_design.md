# Inverse Design

rfx is fully differentiable — `jax.grad` computes gradients through the entire FDTD simulation, enabling gradient-based inverse design of RF structures.

## How It Works

JAX traces the computation graph through all FDTD time steps. `jax.checkpoint` reduces memory from O(n_steps) to O(√n_steps) by re-computing forward states during backpropagation.

```
Forward:  eps_r → FDTD steps → S-parameters → objective
Backward: jax.grad(objective)(eps_r) → gradient of eps_r
```

## Manual Gradient Loop

The most flexible approach:

```python
import jax
import jax.numpy as jnp
from rfx.grid import Grid
from rfx.core.yee import MaterialArrays
from rfx.simulation import run, make_source, make_probe, ProbeSpec
from rfx.sources.sources import GaussianPulse

grid = Grid(freq_max=8e9, domain=(0.04, 0.01, 0.01), dx=0.001, cpml_layers=6)
src = make_source(grid, (0.008, 0.005, 0.005), "ez", GaussianPulse(f0=4e9), n_steps=150)
probe = ProbeSpec(i=30, j=5, k=5, component="ez")

sigma = jnp.zeros(grid.shape, dtype=jnp.float32)
mu_r = jnp.ones(grid.shape, dtype=jnp.float32)

def objective(eps_r):
    mats = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=mu_r)
    result = run(grid, mats, 150, sources=[src], probes=[probe],
                 boundary="pec", checkpoint=True)
    return -jnp.sum(result.time_series ** 2)  # maximize transmission

# Adam optimizer
eps_r = jnp.ones(grid.shape, dtype=jnp.float32)
lr, m, v = 0.05, jnp.zeros_like(eps_r), jnp.zeros_like(eps_r)

for i in range(20):
    loss, grad = jax.value_and_grad(objective)(eps_r)
    m = 0.9 * m + 0.1 * grad
    v = 0.999 * v + 0.001 * grad ** 2
    eps_r = eps_r - lr * m / (jnp.sqrt(v) + 1e-8)
    print(f"iter {i}: loss = {loss:.6e}")
```

## Pre-Built Objectives

```python
from rfx import minimize_s11, maximize_s21, target_impedance, maximize_bandwidth

# Minimize S11 at 5 GHz (target: -10 dB)
obj = minimize_s11(freqs=jnp.array([5e9]), target_db=-10)

# Maximize S21 across a band
obj = maximize_s21(freqs=jnp.linspace(4e9, 6e9, 20))

# Match to 50 ohm
obj = target_impedance(freq=5e9, z_target=50.0)

# Maximize -10 dB bandwidth around 5 GHz
obj = maximize_bandwidth(f_center=5e9, f_bw=2e9, s11_threshold=-10)
```

## Design Region API

```python
from rfx import Simulation, DesignRegion

sim = Simulation(freq_max=10e9, domain=(0.1, 0.04, 0.02), boundary="cpml")
sim.add_port(...)

region = DesignRegion(
    corner_lo=(0.03, 0.0, 0.0),
    corner_hi=(0.07, 0.04, 0.02),
    eps_range=(1.0, 6.0),  # Permittivity bounds
)

from rfx import optimize
result = optimize(sim, region, objective=minimize_s11(...), n_iters=50, lr=0.01)
print(f"Final loss: {result.loss_history[-1]:.4f}")
```

## Tips

- **Always use `checkpoint=True`** — saves 10-100x memory
- **Start with small grids** for iteration, scale up for final design
- **Learning rate**: 0.01-0.1 for eps_r optimization
- **Sigmoid projection**: maps unbounded latent to [eps_min, eps_max]
- **GPU acceleration**: same code, 10-50x faster on NVIDIA GPUs
