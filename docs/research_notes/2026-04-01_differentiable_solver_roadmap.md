# rfx Differentiable Solver Roadmap: Limitations, Gaps, and AI/ML Strategy

## 1. Current rfx Limitations vs Commercial Solvers

| Feature | rfx (v0.1) | Meep | CST | HFSS | Feasibility |
|---------|-----------|------|-----|------|-------------|
| **Mesh** | Uniform Yee (staircase) | Uniform Yee | Hex + conformal | Tet FEM | Subgridding [1] |
| **Curved geometry** | Staircase + subpixel | Subpixel smoothing | Conformal (PBA) | Native curved | Conformal FDTD [2] |
| **Broadband** | Single run | Single run | Single run | Sweep needed | **rfx advantage** |
| **Port modes** | Analytical TE/TM | MPB eigenmode | Numerical eigenmode | Numerical eigenmode | Eigenmode solver (in progress) |
| **Multi-scale** | Single dx | Single dx | Subgridding | Adaptive mesh | Subgridding [3] |
| **MPI distributed** | None (single GPU) | Yes | Yes | Yes | JAX pmap/sharding [4] |
| **Nonlinear** | None | Yes (χ²,χ³) | Yes | None | ADE nonlinear [5] |
| **Thin features** | Thin conductor only | Conductivity | TSA, thin sheets | Impedance BC | Subcell models [6] |
| **GUI** | None | None | Yes | Yes | Not priority (API-first) |
| **Autodiff** | **Native jax.grad** | None | None (adjoint only) | None | **rfx unique** |

### References for gap closure

- [1] Berenger (1999), Chevalier & Luebbers (1997) — Subgridding with interpolation at coarse/fine interface
- [2] Dey & Mittra (1997) "A locally conformal FDTD algorithm" — PEC conformal boundary correction
- [3] JAX `pjit` / `sharding` for domain decomposition — same halo exchange pattern as Meep MPI
- [4] Taflove Ch. 9 — ADE for Kerr, Raman nonlinear materials
- [5] Holland & Simpson (1981) — Thin-wire subcell FDTD models
- [6] Li et al. (2022) — Differentiable rendering/rasterization for inverse photonics

### Assessment

Most gaps are closable with known techniques. The key insight: **rfx does not
need to match CST/HFSS on every axis.** Its differentiability is the unique
value proposition. The gaps that matter most are those that block the
differentiable workflow, not those that block general-purpose simulation.

---

## 2. rfx's Role: Simulator, Not ML Model

### Clear boundary definition

rfx is a **physics simulator** that happens to be differentiable. It is NOT:
- A neural network that approximates Maxwell's equations
- A learned surrogate that replaces simulation
- An ML model that needs training data

rfx IS:
- A first-principles FDTD solver built on JAX
- A differentiable computation graph for gradient-based optimization
- A **data source** for ML pipelines (training data generation, in-loop evaluation)
- An **environment** for reinforcement learning and active learning agents

### Three distinct usage modes

```
Mode 1: DIRECT OPTIMIZATION (no ML)
  params → rfx FDTD → objective → jax.grad → updated params
  ↳ Classical inverse design. rfx provides the gradient directly.
  ↳ No neural network involved. Pure physics + optimization.

Mode 2: DATA GENERATOR (ML training)
  parameter_space → rfx batch simulations → (input, output) dataset
  ↳ rfx generates physics-accurate labeled data
  ↳ Downstream ML model trains on this data
  ↳ rfx is the oracle, not the model

Mode 3: IN-LOOP EVALUATOR (RL / active learning)
  agent proposes design → rfx evaluates → reward/metric → agent updates
  ↳ rfx is the environment, agent is the policy
  ↳ Each rfx call is expensive but exact
  ↳ Active learning: agent chooses WHICH simulations to run
```

### What rfx must provide for each mode

| Mode | rfx requirement | Status |
|------|----------------|--------|
| **Mode 1** | jax.grad through FDTD | Done |
| **Mode 1** | Differentiable geometry parameterization | **Missing (critical)** |
| **Mode 1** | Physics-constrained loss functions | Partial (objectives exist) |
| **Mode 2** | Fast batch simulation (GPU) | Done (1,310 Mcells/s) |
| **Mode 2** | Parameter sweep utilities | **Missing** |
| **Mode 2** | Structured output format (dataset export) | **Missing** |
| **Mode 3** | Gym-like environment interface | **Missing** |
| **Mode 3** | Fast single evaluation | Done |
| **Mode 3** | Reward shaping utilities | **Missing** |

---

## 3. Feature Roadmap for AI/ML Novelty

### Priority 1: Differentiable Geometry Parameterization (CRITICAL)

**Problem**: Current optimization works on raw eps_r voxel arrays. Results are
non-manufacturable, resolution-dependent, and don't generalize.

**Solution**: Level-set based differentiable geometry

```python
# Current (voxel-level, non-parametric):
eps_r = jnp.ones(grid.shape)
eps_r = eps_r.at[10:20, :, :].set(4.0)
jax.grad(objective)(eps_r)  # gradient per voxel

# Needed (parametric, differentiable):
params = {"width": 0.01, "gap": 0.002, "eps": 4.0}
def parametric_to_eps(params, grid):
    phi = level_set_from_params(params, grid)  # signed distance
    return eps_min + (eps_max - eps_min) * sigmoid(beta * phi)
jax.grad(lambda p: objective(parametric_to_eps(p, grid)))(params)
```

**Why critical**: Without this, rfx is a "voxel optimizer" — with this, it
becomes a "parametric RF design tool" that outputs manufacturable geometries.

**References**:
- Li et al. (2022) — differentiable rasterizer for photonics
- Hughes et al. (2018) — forward-mode AD for Maxwell's (Ceviche)
- Minkov et al. (2020) — inverse design of photonic crystals via AD

**Effort**: 2-3 days

### Priority 2: Batch Simulation & Dataset Generation (Mode 2)

**Problem**: No utility for sweeping parameter spaces and exporting structured
datasets for ML training.

**Solution**:

```python
from rfx.ml import ParameterSweep, SimulationDataset

# Define parameter space
sweep = ParameterSweep(
    width=np.linspace(0.005, 0.020, 10),
    length=np.linspace(0.010, 0.050, 10),
    eps_r=np.linspace(2.0, 10.0, 5),
)

# Run all combinations (GPU-accelerated, jax.vmap where possible)
dataset = SimulationDataset.from_sweep(
    sim_factory=lambda w, l, e: build_filter(w, l, e),
    sweep=sweep,
    outputs=["s11", "s21", "z_in"],
)

# Export for ML training
dataset.to_hdf5("filter_dataset.h5")
dataset.to_numpy()  # (X: params, Y: S-params) arrays
```

**Why important**: Most ML-for-EM papers spend 80% of effort on data
generation. A clean pipeline from parameter space → simulation → dataset
dramatically lowers the barrier.

**Effort**: 1-2 days

### Priority 3: RL/Active Learning Environment Interface (Mode 3)

**Problem**: No standard interface for treating rfx as an RL environment.

**Solution**: Gymnasium-compatible wrapper

```python
from rfx.ml import RFDesignEnv

env = RFDesignEnv(
    sim_template=base_simulation,
    design_region=Box((0.02, 0, 0), (0.08, 0.04, 0.02)),
    action_space="continuous",  # eps_r values in design region
    reward_fn=lambda result: -mean(abs(result.s_params[0,0,:])),
    max_steps=50,  # max design iterations
)

# Standard RL loop
obs = env.reset()
for _ in range(100):
    action = agent.predict(obs)
    obs, reward, done, info = env.step(action)
```

**Why important**: Connects rfx to the entire RL ecosystem (Stable-Baselines3,
CleanRL, etc.) without requiring users to understand FDTD.

**Effort**: 1 day

### Priority 4: Physics-Constrained Loss Utilities

**Problem**: Users must manually implement fabrication and physics constraints.

**Solution**: Pre-built constraint functions, all JAX-differentiable.

```python
from rfx.ml import (
    fabrication_penalty,    # min feature size, connectivity
    symmetry_constraint,    # enforce x/y/z mirror symmetry
    total_variation,        # smooth geometry (reduce stairstepping)
    passivity_constraint,   # Σ|S|² ≤ 1
    binary_penalty,         # push eps_r toward discrete values
    volume_fraction,        # material usage constraint
)

loss = s11_loss + 0.1 * fabrication_penalty(eps_r, min_feature=3*dx) \
                + 0.01 * total_variation(eps_r) \
                + 0.1 * binary_penalty(eps_r, targets=[1.0, 4.0])
```

**Effort**: 4 hours

### Priority 5: Fabrication-Aware Optimization

**Problem**: Optimized designs often have features smaller than manufacturing
resolution or disconnected islands.

**Solution**: Density filtering + projection (standard topology optimization)

```python
from rfx.ml import density_filter, threshold_projection

def manufacturable_objective(latent):
    # 1. Density filter (enforce min feature size)
    filtered = density_filter(latent, radius=3*dx)
    # 2. Threshold projection (push toward binary 0/1)
    projected = threshold_projection(filtered, beta=8.0, eta=0.5)
    # 3. Map to eps_r
    eps_r = 1.0 + 5.0 * projected
    return fdtd_objective(eps_r)
```

**References**:
- Lazarov & Sigmund (2011) — density filtering for topology optimization
- Wang et al. (2011) — projection methods for feature size control

**Effort**: 4 hours

---

## 4. What rfx Should NOT Do

To maintain clear scope, rfx should not:

- **Train ML models internally** — rfx generates data/gradients, external
  frameworks (Flax, PyTorch, sklearn) train models
- **Include surrogate model implementations** — surrogates are downstream
  consumers of rfx data, not part of the solver
- **Implement RL algorithms** — rfx provides the environment, not the agent
- **Replace domain expertise** — rfx accelerates the design loop, it doesn't
  eliminate the need for RF engineering knowledge

### Boundary principle

```
rfx scope boundary:
  ┌─────────────────────────────────────────────┐
  │ Physics simulation (FDTD)                    │
  │ Gradient computation (jax.grad)              │
  │ Geometry parameterization (level-set)        │
  │ Dataset generation (parameter sweep)         │
  │ Environment interface (Gym-like)             │
  │ Physics constraints (passivity, fabrication) │
  └─────────────────────────────────────────────┘
            ↕ clean API boundary ↕
  ┌─────────────────────────────────────────────┐
  │ ML model training (Flax, PyTorch)           │
  │ RL agents (SB3, CleanRL)                    │
  │ Surrogate models (user-built)               │
  │ Active learning strategies (user-defined)   │
  │ Visualization dashboards (external)         │
  └─────────────────────────────────────────────┘
```

rfx provides the **physics oracle + gradient oracle + data pipeline**.
Everything above the boundary is the user's domain.

---

## 5. Implementation Priority for v1.0 → v2.0

### v1.0 (current sprint)
- [x] Core FDTD + autodiff
- [x] S-parameters, waveguide ports, RCS
- [x] Objective library
- [ ] Eigenmode solver (in progress)
- [ ] Clean public repo

### v1.1 (differentiable geometry)
- [ ] Level-set parameterization with differentiable rasterization
- [ ] Density filtering + threshold projection
- [ ] Fabrication constraint utilities
- [ ] Example: parametric waveguide filter optimization

### v1.2 (ML data pipeline)
- [ ] ParameterSweep + SimulationDataset
- [ ] Batch simulation with jax.vmap
- [ ] HDF5/NumPy dataset export
- [ ] Example: train surrogate model on rfx data

### v2.0 (RL/active learning)
- [ ] Gymnasium environment wrapper
- [ ] Reward shaping utilities
- [ ] Example: RL agent designs antenna
- [ ] Conformal PEC boundaries
- [ ] Multi-GPU support
