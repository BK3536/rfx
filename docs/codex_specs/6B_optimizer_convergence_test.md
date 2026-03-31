# Codex Spec 6B: End-to-End Optimization Convergence Test

## Goal
Create `tests/test_optimize_convergence.py` that proves the differentiable
optimization pipeline actually converges on real EM problems.

## Context
- `rfx/optimize.py` has `optimize()` with Adam gradient descent and
  `DesignRegion` for eps_r parameterization via sigmoid projection.
- `rfx/api.py` has `Simulation` builder with `run()` returning differentiable results.
- No existing test verifies convergence — only bounds/mapping are tested in `test_optimize.py`.
- JAX reverse-mode AD through `jax.lax.scan` with `jax.checkpoint` is validated.

## Deliverable
`tests/test_optimize_convergence.py` with two tests:

### Test 1: `test_optimize_dielectric_slab_s11`
Optimize a dielectric slab thickness/permittivity to minimize |S11| at 5 GHz.

Setup:
- Domain: (0.06, 0.02, 0.02), boundary="cpml", cpml_layers=8
- Single lumped port at x=0.01
- Design region: Box((0.025, 0, 0), (0.035, 0.02, 0.02))
- eps_bounds: (1.0, 6.0)
- Objective: `sum(|S11(f)|^2)` over 3-5 frequencies around 5 GHz
- Use `rfx.optimize.optimize()` or manual `jax.grad` + Adam loop

Assertions:
- Objective decreases over iterations (not necessarily monotonic due to Adam, but final < initial * 0.5)
- Converges within 30 iterations
- No NaN in gradients or objective

### Test 2: `test_optimize_waveguide_transmission`
Optimize eps_r in a design region between source and probe to maximize
time-integrated |Ez|^2 at the probe.

Setup:
- Domain: (0.04, 0.01, 0.01), boundary="cpml", cpml_layers=6
- GaussianPulse source at (0.008, 0.005, 0.005)
- Probe at (0.030, 0.005, 0.005)
- Design region: Box((0.015, 0, 0), (0.025, 0.01, 0.01))
- eps_bounds: (1.0, 4.0)
- Objective: `-sum(time_series^2)` (negative because we maximize)
- n_steps: 100-150 (keep short for test speed)

Assertions:
- Final objective < initial objective (transmission increased)
- Gradient norm is non-zero at step 0
- No NaN

## Constraints
- Each test must complete in < 120 seconds on CPU
- Use `checkpoint=True` in all `run()` calls
- Import from `rfx.optimize` if useful, but manual grad loop is also fine
- Do NOT modify any existing files — only create the new test file

## Verification
Run: `pytest -xvs tests/test_optimize_convergence.py`
Both tests must pass.
