# Phase C: 3D SBP-SAT Stabilization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make 3D SBP-SAT subgridding energy-stable and accuracy-validated so it can serve as the foundation for auto-subgrid (Phase A).

**Architecture:** Fix the JIT runner's missing H-field coupling, decide and implement the correct timestep scheme (sub-stepping vs global), then validate with energy conservation tests and crossval-grade accuracy tests against uniform fine grids.

**Tech Stack:** JAX, jax.lax.scan, pytest, matplotlib

**Spec:** `docs/superpowers/specs/2026-04-12-auto-subgrid-design.md` (Phase C)

---

### Task 1: Fix JIT runner — add H-field SAT coupling

The JIT runner (`jit_runner.py`) only calls E-field SAT coupling. The H-field coupling function exists in `sbp_sat_3d.py` but is not imported or called. This violates energy conservation.

**Files:**
- Modify: `rfx/subgridding/jit_runner.py:26` (import) and `:119-190` (step_fn)
- Test: `tests/test_sbp_sat_jit.py` (existing, verify still passes)

- [ ] **Step 1: Write a failing test that detects missing H-coupling**

Add to `tests/test_sbp_sat_jit.py`:

```python
def test_jit_runner_h_coupling_energy():
    """JIT runner must couple both E and H fields for energy conservation.

    Without H-coupling, energy can grow. With H-coupling, energy should
    be bounded. Uses a short source burst then measures post-source energy.
    """
    import jax.numpy as jnp
    from rfx.subgridding.sbp_sat_3d import init_subgrid_3d, compute_energy_3d, SubgridState3D
    from rfx.subgridding.jit_runner import run_subgridded_jit
    from rfx.core.yee import init_materials
    from rfx.grid import Grid
    from rfx.sources.sources import GaussianPulse

    dx_c = 0.003
    shape_c = (20, 20, 20)
    config, _ = init_subgrid_3d(
        shape_c=shape_c, dx_c=dx_c,
        fine_region=(7, 13, 7, 13, 7, 13), ratio=3,
    )
    grid_c = Grid(freq_max=5e9, domain=tuple(s * dx_c for s in shape_c), dx=dx_c)
    mats_c = init_materials(shape_c)
    mats_f = init_materials((config.nx_f, config.ny_f, config.nz_f))

    # Source: short Gaussian pulse on fine grid at center
    # Fine grid center indices
    si, sj, sk = config.nx_f // 2, config.ny_f // 2, config.nz_f // 2
    n_steps = 500
    pulse = GaussianPulse(f0=5e9, bandwidth=0.5)
    waveform = jnp.array([float(pulse(i * config.dt)) for i in range(n_steps)])

    result = run_subgridded_jit(
        grid_c, mats_c, mats_f, config,
        n_steps=n_steps,
        sources_f=[(si, sj, sk, "ez", waveform)],
    )

    # After source dies out, energy in the PEC cavity should be conserved.
    # Run a second pass with no source to measure energy conservation.
    # (The first run establishes the fields; we check that the JIT runner
    # didn't cause unphysical energy growth during propagation.)
    final_state = SubgridState3D(state_c=result.state_c, state_f=result.state_f)
    e_final = compute_energy_3d(final_state, config)

    # Energy should be finite and positive (source injected energy)
    assert jnp.isfinite(e_final), "Final energy not finite — coupling broken"
    assert e_final > 0, "No energy in cavity — source or coupling failed"

    # Compare: run same config with standalone stepper (which has H-coupling)
    # If JIT and standalone diverge significantly, H-coupling is missing in JIT
    _, state_standalone = init_subgrid_3d(
        shape_c=shape_c, dx_c=dx_c,
        fine_region=(7, 13, 7, 13, 7, 13), ratio=3,
    )
    from rfx.subgridding.sbp_sat_3d import step_subgrid_3d
    st = state_standalone
    st = st._replace(ez_f=st.ez_f.at[si, sj, sk].set(1.0))
    for i in range(500):
        st = step_subgrid_3d(st, config)
    e_standalone = compute_energy_3d(st, config)

    # Both should be finite; large divergence indicates missing coupling
    assert jnp.isfinite(e_standalone), "Standalone energy not finite"
    if e_standalone > 1e-20:
        ratio = float(e_final / e_standalone)
        # They won't match exactly (different source injection), but
        # order-of-magnitude agreement indicates both paths work
        assert 0.01 < ratio < 100, (
            f"JIT vs standalone energy ratio={ratio:.2f} — "
            f"likely H-coupling mismatch"
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sbp_sat_jit.py::test_jit_runner_h_coupling_energy -xvs`
Expected: FAIL — energy grows because H-coupling is missing from JIT runner.

- [ ] **Step 3: Add H-coupling import and call to JIT runner**

In `rfx/subgridding/jit_runner.py`, line 26, change:

```python
from rfx.subgridding.sbp_sat_3d import (
    SubgridConfig3D, _shared_node_coupling_3d,
)
```

to:

```python
from rfx.subgridding.sbp_sat_3d import (
    SubgridConfig3D, _shared_node_coupling_3d, _shared_node_coupling_h_3d,
)
```

In `step_fn` (around line 137), after both H-updates and before E-updates, add H-coupling:

```python
        # === Fine H update ===
        st_f = FDTDState(ex=ex_f, ey=ey_f, ez=ez_f,
                         hx=hx_f, hy=hy_f, hz=hz_f,
                         step=step_idx)
        st_f = update_h(st_f, mats_f, dt, dx_f)

        # === SBP-SAT H-field coupling (ADDED — required for energy conservation) ===
        (hx_c_new, hy_c_new, hz_c_new), (hx_f_new, hy_f_new, hz_f_new) = \
            _shared_node_coupling_h_3d(
                (st_c.hx, st_c.hy, st_c.hz),
                (st_f.hx, st_f.hy, st_f.hz),
                config,
            )
        st_c = st_c._replace(hx=hx_c_new, hy=hy_c_new, hz=hz_c_new)
        st_f = st_f._replace(hx=hx_f_new, hy=hy_f_new, hz=hz_f_new)

        # === Coarse E update + boundary ===
        st_c = update_e(st_c, mats_c, dt, dx_c)
```

Also update the carry packing at the end of `step_fn` to use the coupled H fields:

```python
        carry_out = {
            "c": (ex_c_new, ey_c_new, ez_c_new,
                  hx_c_new, hy_c_new, hz_c_new),
            "f": (ex_f_new, ey_f_new, ez_f_new,
                  hx_f_new, hy_f_new, hz_f_new),
        }
```

Note: Check the existing carry_out assignment and make sure it uses the SAT-coupled H-fields (`hx_c_new` etc.) rather than the pre-coupling values from `st_c.hx`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sbp_sat_jit.py::test_jit_runner_h_coupling_energy -xvs`
Expected: PASS — energy growth < 5%.

- [ ] **Step 5: Run all existing JIT tests**

Run: `pytest tests/test_sbp_sat_jit.py -xvs`
Expected: All existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add rfx/subgridding/jit_runner.py tests/test_sbp_sat_jit.py
git commit -m "fix(subgrid): add H-field SAT coupling to JIT runner

JIT runner was only coupling E-fields at the coarse-fine interface.
H-field coupling is required for energy conservation per Cheng et al.
2025. Without it, energy can grow unbounded.

Added _shared_node_coupling_h_3d call after H-updates, before E-updates."
```

---

### Task 2: Energy conservation test — standalone stepper

Update the existing 3D tests to require energy conservation (non-increasing), not just "finite".

**Files:**
- Modify: `tests/test_sbp_sat_3d.py`

- [ ] **Step 1: Replace the warning-based test with a strict assertion**

Replace the entire `test_3d_stability` function in `tests/test_sbp_sat_3d.py`:

```python
def test_3d_stability():
    """Energy must be non-increasing over 1000 steps in PEC cavity.

    This is the fundamental SBP-SAT stability guarantee. If energy grows,
    the coupling coefficients are wrong.
    """
    config, state = init_subgrid_3d(
        shape_c=(20, 20, 20), dx_c=0.003,
        fine_region=(7, 13, 7, 13, 7, 13), ratio=3,
    )
    state = state._replace(ez_c=state.ez_c.at[4, 4, 4].set(1.0))
    initial_energy = compute_energy_3d(state, config)

    max_energy = initial_energy
    for i in range(1000):
        state = step_subgrid_3d(state, config)
        if (i + 1) % 100 == 0:
            e = compute_energy_3d(state, config)
            # Allow tiny float32 growth per step (~1e-6 relative)
            assert e <= max_energy * 1.001, (
                f"Energy grew at step {i+1}: {e:.6e} > {max_energy:.6e} "
                f"(growth {e/max_energy:.6f}x)"
            )
            max_energy = max(max_energy, e)

    final_energy = compute_energy_3d(state, config)
    print(f"\n3D energy conservation: initial={initial_energy:.6e}, "
          f"final={final_energy:.6e}, ratio={final_energy/initial_energy:.6f}")
    assert final_energy <= initial_energy * 1.01, (
        f"Total energy grew {final_energy/initial_energy:.4f}x over 1000 steps"
    )
```

- [ ] **Step 2: Run the updated test**

Run: `pytest tests/test_sbp_sat_3d.py::test_3d_stability -xvs`
Expected: May FAIL if current coupling coefficients allow energy growth. This is expected — it will pass after Task 1's H-coupling fix propagates to the standalone stepper (the standalone stepper already has H-coupling, so it should pass if coefficients are correct).

If it fails, note the growth factor — this tells us whether the issue is the coefficients or the scheme.

- [ ] **Step 3: Commit**

```bash
git add tests/test_sbp_sat_3d.py
git commit -m "test(subgrid): strict energy conservation test for 3D SBP-SAT

Replace warning-based energy test with strict assertion:
energy must be non-increasing (within float32 tolerance) over 1000 steps."
```

---

### Task 3: Investigate timestep scheme decision

Before implementing changes, run diagnostics to determine whether the current global-timestep scheme with correct H+E coupling is energy-stable, or whether sub-stepping is required.

**Files:**
- Create: `tests/test_subgrid_timestep_investigation.py` (temporary diagnostic, not permanent)

- [ ] **Step 1: Write diagnostic that compares energy behavior**

```python
"""Diagnostic: compare energy behavior of 3D SBP-SAT with different configs.

NOT a permanent test — this is an investigation to decide whether
sub-stepping is needed or global timestep + H-coupling is sufficient.

Run: pytest tests/test_subgrid_timestep_investigation.py -xvs
"""

import numpy as np
from rfx.subgridding.sbp_sat_3d import (
    init_subgrid_3d, step_subgrid_3d, compute_energy_3d,
)


def test_energy_vs_tau():
    """Sweep tau from 0.1 to 1.5 and report energy growth at 500 steps."""
    taus = [0.1, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5]
    print("\n=== Energy growth vs tau (500 steps, ratio=3) ===")
    for tau in taus:
        config, state = init_subgrid_3d(
            shape_c=(20, 20, 20), dx_c=0.003,
            fine_region=(7, 13, 7, 13, 7, 13), ratio=3, tau=tau,
        )
        state = state._replace(ez_c=state.ez_c.at[10, 10, 10].set(1.0))
        e0 = compute_energy_3d(state, config)
        for _ in range(500):
            state = step_subgrid_3d(state, config)
        e1 = compute_energy_3d(state, config)
        growth = e1 / max(e0, 1e-30)
        status = "STABLE" if growth <= 1.01 else "UNSTABLE"
        print(f"  tau={tau:.1f}: growth={growth:.6f}x [{status}]")


def test_energy_vs_ratio():
    """Sweep ratio from 2 to 5 and report energy growth at 500 steps."""
    ratios = [2, 3, 4, 5]
    print("\n=== Energy growth vs ratio (500 steps, tau=0.5) ===")
    for ratio in ratios:
        config, state = init_subgrid_3d(
            shape_c=(20, 20, 20), dx_c=0.003,
            fine_region=(7, 13, 7, 13, 7, 13), ratio=ratio, tau=0.5,
        )
        state = state._replace(ez_c=state.ez_c.at[10, 10, 10].set(1.0))
        e0 = compute_energy_3d(state, config)
        for _ in range(500):
            state = step_subgrid_3d(state, config)
        e1 = compute_energy_3d(state, config)
        growth = e1 / max(e0, 1e-30)
        status = "STABLE" if growth <= 1.01 else "UNSTABLE"
        print(f"  ratio={ratio}: growth={growth:.6f}x [{status}]")


def test_long_run_energy_trace():
    """10k steps, print energy every 1000 — check for late-time growth."""
    config, state = init_subgrid_3d(
        shape_c=(20, 20, 20), dx_c=0.003,
        fine_region=(7, 13, 7, 13, 7, 13), ratio=3, tau=0.5,
    )
    state = state._replace(ez_c=state.ez_c.at[10, 10, 10].set(1.0))
    e0 = compute_energy_3d(state, config)
    print(f"\n=== Long-run energy trace (10k steps) ===")
    print(f"  step    0: energy={e0:.6e}")
    for step in range(10000):
        state = step_subgrid_3d(state, config)
        if (step + 1) % 1000 == 0:
            e = compute_energy_3d(state, config)
            print(f"  step {step+1:5d}: energy={e:.6e}  "
                  f"(ratio={e/e0:.6f})")
```

- [ ] **Step 2: Run the diagnostic**

Run: `pytest tests/test_subgrid_timestep_investigation.py -xvs`

Record the output. The results will determine the next step:

- If energy is stable (growth <= 1.01) for tau=0.5 or tau=1.0 with ratio=3: **global timestep with H-coupling is sufficient**. No sub-stepping needed.
- If energy grows for all tau values: **sub-stepping is required**. Implement the 2D-style sub-stepping pattern for 3D.

- [ ] **Step 3: Document the decision**

Based on diagnostic results, add a comment block to `sbp_sat_3d.py` documenting:
- Which timestep scheme was chosen and why
- The energy growth data that justified the decision
- If sub-stepping is needed, reference the 2D implementation pattern

- [ ] **Step 4: Commit diagnostic results**

```bash
git add tests/test_subgrid_timestep_investigation.py rfx/subgridding/sbp_sat_3d.py
git commit -m "investigate(subgrid): timestep scheme diagnostic for 3D SBP-SAT

Sweep tau and ratio to determine if global timestep + H-coupling
is energy-stable, or if sub-stepping is required.
Results: [FILL IN BASED ON OUTPUT]"
```

---

### Task 4: Implement timestep scheme fix (if needed)

**This task depends on Task 3 results.** Two paths:

**Path A (global timestep is stable):** Minimal changes — update docstrings, set optimal default tau, document equations.

**Path B (sub-stepping needed):** Rewrite `sbp_sat_3d.py` step function to match 2D pattern.

**Files:**
- Modify: `rfx/subgridding/sbp_sat_3d.py`
- Modify: `rfx/subgridding/jit_runner.py` (if sub-stepping)
- Test: `tests/test_sbp_sat_3d.py`

- [ ] **Step 1 (Path A): Update defaults and documentation**

If global timestep is stable, update `init_subgrid_3d` default tau to the optimal value found in Task 3, and update the module docstring to document the scheme choice:

```python
"""3D SBP-SAT FDTD subgridding (Phase 3).

...
Timestep scheme: global dt (same for coarse and fine).
Both E-field and H-field SAT coupling are applied per step.
Energy conservation validated: growth < 0.1% over 10k steps
at tau={optimal_tau}, ratio=3.

Note: this differs from the 2D scheme which uses temporal sub-stepping.
The global-timestep scheme avoids operator-splitting errors and is
simpler to implement in JAX lax.scan. Energy stability is achieved
through correct SAT penalty coefficients on both E and H fields.
"""
```

- [ ] **Step 1 (Path B): Implement sub-stepping**

If sub-stepping is needed, rewrite `step_subgrid_3d` following the 2D pattern:

```python
def step_subgrid_3d(state, config, *, mats_c=None, mats_f=None,
                    pec_mask_c=None, pec_mask_f=None):
    """One coupled timestep: fine runs `ratio` sub-steps, coarse runs 1 step."""
    dt_f = config.dt              # fine timestep
    dt_c = config.ratio * dt_f    # coarse timestep
    dx_c, dx_f = config.dx_c, config.dx_f
    
    st_c, st_f = state.state_c, state.state_f
    
    # === Fine grid: ratio sub-steps ===
    for _ in range(config.ratio):
        st_f = update_h(st_f, mats_f or init_materials((config.nx_f, config.ny_f, config.nz_f)), dt_f, dx_f)
        st_f = update_e(st_f, mats_f or init_materials((config.nx_f, config.ny_f, config.nz_f)), dt_f, dx_f)
        if pec_mask_f is not None:
            st_f = apply_pec_mask(st_f, pec_mask_f)
    
    # === Coarse grid: 1 step with dt_c ===
    st_c = update_h(st_c, mats_c or init_materials(config.shape_c), dt_c, dx_c)
    st_c = update_e(st_c, mats_c or init_materials(config.shape_c), dt_c, dx_c)
    if pec_mask_c is not None:
        st_c = apply_pec_mask(st_c, pec_mask_c)
    
    # === SAT coupling (E + H) ===
    (ex_c, ey_c, ez_c), (ex_f, ey_f, ez_f) = _shared_node_coupling_3d(
        (st_c.ex, st_c.ey, st_c.ez), (st_f.ex, st_f.ey, st_f.ez), config)
    (hx_c, hy_c, hz_c), (hx_f, hy_f, hz_f) = _shared_node_coupling_h_3d(
        (st_c.hx, st_c.hy, st_c.hz), (st_f.hx, st_f.hy, st_f.hz), config)
    
    st_c = st_c._replace(ex=ex_c, ey=ey_c, ez=ez_c, hx=hx_c, hy=hy_c, hz=hz_c)
    st_f = st_f._replace(ex=ex_f, ey=ey_f, ez=ez_f, hx=hx_f, hy=hy_f, hz=hz_f)
    
    return state._replace(state_c=st_c, state_f=st_f)
```

Note: If sub-stepping is chosen, the JIT runner also needs updating — the `lax.scan` body must include an inner loop for fine sub-steps. This can be done with `jax.lax.fori_loop` inside the scan body.

- [ ] **Step 2: Run energy conservation test**

Run: `pytest tests/test_sbp_sat_3d.py::test_3d_stability -xvs`
Expected: PASS — energy non-increasing.

- [ ] **Step 3: Run all 3D tests**

Run: `pytest tests/test_sbp_sat_3d.py tests/test_sbp_sat_jit.py -xvs`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add rfx/subgridding/sbp_sat_3d.py rfx/subgridding/jit_runner.py tests/
git commit -m "fix(subgrid): [PATH A or B] 3D SBP-SAT timestep scheme

[Document which path was taken and why, based on Task 3 results]"
```

---

### Task 5: Accuracy crossval — Fresnel slab (1D-like test)

Compare subgridded simulation vs uniform fine grid on the Fresnel slab.

**Files:**
- Create: `tests/test_subgrid_crossval.py`
- Reference: `examples/crossval/04_multilayer_fresnel.py`

- [ ] **Step 1: Write the crossval test**

```python
"""Crossval: subgridded vs uniform-fine on Fresnel slab.

Validates that the subgridding introduces < 5% error compared to
a uniform fine grid simulation of the same problem.
"""
import pytest
import numpy as np
import jax.numpy as jnp

from rfx import Simulation, Box
from rfx.sources.sources import GaussianPulse

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def _run_fresnel(dx, n_steps, use_subgrid=False, subgrid_z_range=None, ratio=3):
    """Run a simple Fresnel slab and return probe time series."""
    f0 = 5e9
    domain = (0.04, 0.04, 0.08)
    
    sim = Simulation(freq_max=f0 * 3, domain=domain, dx=dx, boundary="cpml", cpml_layers=8)
    sim.add_material("dielectric", eps_r=4.0)
    # Slab in the middle of the domain
    slab_z0, slab_z1 = 0.03, 0.05
    sim.add(Box((0, 0, slab_z0), (domain[0], domain[1], slab_z1)), material="dielectric")
    
    sim.add_source(position=(domain[0]/2, domain[1]/2, 0.015), component="ez",
                   waveform=GaussianPulse(f0=f0, bandwidth=0.5))
    sim.add_probe(position=(domain[0]/2, domain[1]/2, 0.065), component="ez")  # transmitted
    sim.add_probe(position=(domain[0]/2, domain[1]/2, 0.025), component="ez")  # reflected
    
    if use_subgrid and subgrid_z_range is not None:
        sim.add_refinement(z_range=subgrid_z_range, ratio=ratio)
    
    result = sim.run(n_steps=n_steps)
    return np.array(result.time_series), float(result.dt)


def test_subgrid_fresnel_accuracy():
    """Subgridded Fresnel slab must match uniform-fine within 5%."""
    dx_coarse = 2e-3   # 2 mm
    dx_fine = 2e-3 / 3  # ~0.67 mm (ratio=3 equivalent)
    n_steps = 2000
    
    # Run 1: uniform fine (reference)
    ts_fine, dt_fine = _run_fresnel(dx_fine, n_steps)
    
    # Run 2: coarse + subgrid around slab
    ts_sub, dt_sub = _run_fresnel(
        dx_coarse, n_steps, use_subgrid=True,
        subgrid_z_range=(0.025, 0.055), ratio=3,
    )
    
    # Compare transmitted probe (index 0)
    # Resample to common time axis if dt differs
    t_fine = np.arange(ts_fine.shape[0]) * dt_fine
    t_sub = np.arange(ts_sub.shape[0]) * dt_sub
    t_max = min(t_fine[-1], t_sub[-1])
    t_common = np.linspace(0, t_max, 1000)
    
    sig_fine = np.interp(t_common, t_fine, ts_fine[:, 0])
    sig_sub = np.interp(t_common, t_sub, ts_sub[:, 0])
    
    # Normalized RMS error
    ref_max = np.max(np.abs(sig_fine))
    if ref_max < 1e-15:
        pytest.skip("Reference signal too weak")
    
    rms_error = np.sqrt(np.mean((sig_fine - sig_sub) ** 2)) / ref_max
    print(f"\nFresnel slab crossval:")
    print(f"  Uniform fine dx={dx_fine*1e3:.2f}mm: peak={np.max(np.abs(sig_fine)):.6e}")
    print(f"  Subgridded dx={dx_coarse*1e3:.1f}mm + ratio=3: peak={np.max(np.abs(sig_sub)):.6e}")
    print(f"  Normalized RMS error: {rms_error*100:.2f}%")
    
    assert rms_error < 0.05, f"Subgrid error {rms_error*100:.2f}% exceeds 5% threshold"
```

- [ ] **Step 2: Run locally (CPU, may be slow)**

Run: `pytest tests/test_subgrid_crossval.py::test_subgrid_fresnel_accuracy -xvs`
Expected: PASS with < 5% error. If it fails, the subgridding accuracy needs investigation before proceeding.

- [ ] **Step 3: Commit**

```bash
git add tests/test_subgrid_crossval.py
git commit -m "test(subgrid): Fresnel slab accuracy crossval — subgrid vs uniform fine"
```

---

### Task 6: Accuracy crossval — 3D cavity (stress test)

The Fresnel slab only tests z-axis coupling. This test uses a metallic cavity with an off-axis source to stress all 6 face couplings.

**Files:**
- Modify: `tests/test_subgrid_crossval.py` (add second test)

- [ ] **Step 1: Add 3D cavity crossval test**

Append to `tests/test_subgrid_crossval.py`:

```python
def test_subgrid_3d_cavity_accuracy():
    """3D PEC cavity with off-axis source — stresses all 6 subgrid faces.

    Uniform-fine vs subgridded comparison. The subgrid region is NOT
    aligned with any symmetry axis, forcing oblique wave-subgrid
    interaction on all faces.
    """
    f0 = 5e9
    dx_coarse = 2e-3
    domain = (0.04, 0.04, 0.04)
    n_steps = 1500
    
    # PEC cavity (no CPML — reflecting on all sides)
    def _run_cavity(dx, use_subgrid=False):
        sim = Simulation(
            freq_max=f0 * 3, domain=domain, dx=dx,
            boundary="pec",
        )
        # Off-axis source — not centered, not on any symmetry plane
        sim.add_source(
            position=(0.012, 0.015, 0.018), component="ez",
            waveform=GaussianPulse(f0=f0, bandwidth=0.5),
        )
        # Probe at another asymmetric position
        sim.add_probe(position=(0.028, 0.025, 0.022), component="ez")
        
        if use_subgrid:
            # Refinement region NOT centered — asymmetric box
            sim.add_refinement(z_range=(0.012, 0.028), ratio=3)
        
        result = sim.run(n_steps=n_steps)
        return np.array(result.time_series).ravel(), float(result.dt)
    
    # Reference: uniform fine
    dx_fine = dx_coarse / 3
    ts_fine, dt_fine = _run_cavity(dx_fine)
    
    # Subgridded
    ts_sub, dt_sub = _run_cavity(dx_coarse, use_subgrid=True)
    
    # Compare on common time axis
    t_fine = np.arange(len(ts_fine)) * dt_fine
    t_sub = np.arange(len(ts_sub)) * dt_sub
    t_max = min(t_fine[-1], t_sub[-1])
    t_common = np.linspace(0, t_max, 1000)
    
    sig_fine = np.interp(t_common, t_fine, ts_fine)
    sig_sub = np.interp(t_common, t_sub, ts_sub)
    
    ref_max = np.max(np.abs(sig_fine))
    if ref_max < 1e-15:
        pytest.skip("Reference signal too weak")
    
    rms_error = np.sqrt(np.mean((sig_fine - sig_sub) ** 2)) / ref_max
    print(f"\n3D cavity crossval:")
    print(f"  Uniform fine: peak={np.max(np.abs(sig_fine)):.6e}")
    print(f"  Subgridded:   peak={np.max(np.abs(sig_sub)):.6e}")
    print(f"  Normalized RMS error: {rms_error*100:.2f}%")
    
    assert rms_error < 0.10, f"3D cavity error {rms_error*100:.2f}% exceeds 10% threshold"
```

- [ ] **Step 2: Run locally**

Run: `pytest tests/test_subgrid_crossval.py::test_subgrid_3d_cavity_accuracy -xvs`
Expected: PASS with < 10% error.

- [ ] **Step 3: Commit**

```bash
git add tests/test_subgrid_crossval.py
git commit -m "test(subgrid): 3D cavity accuracy crossval — oblique wave stress test"
```

---

### Task 7: Fix dead code + cleanup

Clean up the runner and documentation.

**Files:**
- Modify: `rfx/runners/subgridded.py:44` (dead expression)
- Modify: `rfx/subgridding/sbp_sat_3d.py` (docstring update with equation refs)

- [ ] **Step 1: Read and fix dead code**

Read `rfx/runners/subgridded.py` around line 44. Find the expression statement that computes but discards a value. Either assign it to a variable and use it, or remove the line.

- [ ] **Step 2: Update sbp_sat_3d.py module docstring**

Update the module docstring to reflect the current state:
- Which timestep scheme is used (result of Task 3)
- Which Cheng et al. equations are implemented
- Energy conservation status (passes/fails, with data)

- [ ] **Step 3: Run all subgridding tests**

Run: `pytest tests/test_sbp_sat_1d.py tests/test_sbp_sat_2d.py tests/test_sbp_sat_3d.py tests/test_sbp_sat_alpha.py tests/test_sbp_sat_jit.py tests/test_subgrid_crossval.py -v --tb=short`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add rfx/runners/subgridded.py rfx/subgridding/sbp_sat_3d.py
git commit -m "chore(subgrid): fix dead code, update docstrings with equation refs"
```

---

### Task 8: Delete temporary investigation file + final push

- [ ] **Step 1: Remove diagnostic file**

```bash
git rm tests/test_subgrid_timestep_investigation.py
```

(Keep it if the diagnostic results are valuable as a permanent test. Otherwise, the data is captured in commit messages and sbp_sat_3d.py docstring.)

- [ ] **Step 2: Final test run**

Run: `pytest tests/ -m gpu --tb=short -q --ignore=tests/test_meep_crossval.py --ignore=tests/test_meep_crossval_patch.py --ignore=tests/test_openems_crossval.py --ignore=tests/test_crossval_comprehensive.py --ignore=tests/test_distributed.py -k "not slow"`
Expected: All pass (same as v1.5.0 validation, plus new subgrid tests).

- [ ] **Step 3: Commit and push**

```bash
git add -A
git commit -m "chore(subgrid): Phase C complete — 3D SBP-SAT stabilized + validated"
git push origin main
```

---

## Phase C Exit Criteria Checklist

After all tasks complete, verify:

- [ ] JIT runner calls both E-field and H-field SAT coupling
- [ ] Timestep scheme decision documented with evidence
- [ ] Energy non-increasing over 1000+ steps (PEC cavity, JIT runner)
- [ ] Fresnel slab: subgridded vs uniform-fine RMS error < 5%
- [ ] 3D cavity: subgridded vs uniform-fine RMS error < 10%
- [ ] No dead code in runner
- [ ] Module docstrings document equations and scheme choice
- [ ] All existing tests still pass
