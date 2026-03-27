#!/usr/bin/env python
"""Benchmark: JAX (JIT) vs pure NumPy FDTD performance.

Runs the same PEC cavity simulation with:
1. rfx (JAX with @jax.jit on update_h/update_e)
2. Pure NumPy reference implementation (no JIT)

Reports wall-clock time and speedup factor.
"""

import time
import numpy as np

from rfx.grid import Grid
from rfx.core.yee import (
    init_state, init_materials, update_e, update_h,
    EPS_0, MU_0, _shift_fwd, _shift_bwd,
)
from rfx.boundaries.pec import apply_pec
from rfx.sources.sources import GaussianPulse


# ---------- Pure NumPy reference ----------

def _np_shift_fwd(arr, axis):
    """NumPy equivalent of _shift_fwd."""
    pad_widths = [(0, 0)] * arr.ndim
    pad_widths[axis] = (0, 1)
    padded = np.pad(arr, pad_widths)
    slices = [slice(None)] * arr.ndim
    slices[axis] = slice(1, None)
    return padded[tuple(slices)]


def _np_shift_bwd(arr, axis):
    """NumPy equivalent of _shift_bwd."""
    pad_widths = [(0, 0)] * arr.ndim
    pad_widths[axis] = (1, 0)
    padded = np.pad(arr, pad_widths)
    slices = [slice(None)] * arr.ndim
    slices[axis] = slice(None, -1)
    return padded[tuple(slices)]


def np_update_h(ex, ey, ez, hx, hy, hz, mu, dt, dx):
    """Pure NumPy H update."""
    curl_x = (_np_shift_fwd(ez, 1) - ez) / dx - (_np_shift_fwd(ey, 2) - ey) / dx
    curl_y = (_np_shift_fwd(ex, 2) - ex) / dx - (_np_shift_fwd(ez, 0) - ez) / dx
    curl_z = (_np_shift_fwd(ey, 0) - ey) / dx - (_np_shift_fwd(ex, 1) - ex) / dx
    hx = hx - (dt / mu) * curl_x
    hy = hy - (dt / mu) * curl_y
    hz = hz - (dt / mu) * curl_z
    return hx, hy, hz


def np_update_e(ex, ey, ez, hx, hy, hz, eps, sigma, dt, dx):
    """Pure NumPy E update."""
    sigma_dt_2eps = sigma * dt / (2.0 * eps)
    ca = (1.0 - sigma_dt_2eps) / (1.0 + sigma_dt_2eps)
    cb = (dt / eps) / (1.0 + sigma_dt_2eps)
    curl_x = (hz - _np_shift_bwd(hz, 1)) / dx - (hy - _np_shift_bwd(hy, 2)) / dx
    curl_y = (hx - _np_shift_bwd(hx, 2)) / dx - (hz - _np_shift_bwd(hz, 0)) / dx
    curl_z = (hy - _np_shift_bwd(hy, 0)) / dx - (hx - _np_shift_bwd(hx, 1)) / dx
    ex = ca * ex + cb * curl_x
    ey = ca * ey + cb * curl_y
    ez = ca * ez + cb * curl_z
    return ex, ey, ez


def np_apply_pec(ex, ey, ez):
    """Pure NumPy PEC."""
    ey[0, :, :] = ey[-1, :, :] = 0.0
    ez[0, :, :] = ez[-1, :, :] = 0.0
    ex[:, 0, :] = ex[:, -1, :] = 0.0
    ez[:, 0, :] = ez[:, -1, :] = 0.0
    ex[:, :, 0] = ex[:, :, -1] = 0.0
    ey[:, :, 0] = ey[:, :, -1] = 0.0
    return ex, ey, ez


# ---------- JAX fori_loop variant ----------

def make_jax_fori_step(materials, dt, dx, cx, cy, cz, pulse):
    """Create a fully JIT-compiled step function for jax.lax.fori_loop."""
    import jax
    import jax.numpy as jnp

    # Pre-extract pulse parameters as static values for JIT
    p_amp = pulse.amplitude
    p_t0 = pulse.t0
    p_tau = pulse.tau

    @jax.jit
    def run_n_steps(state, n_steps):
        def body(n, s):
            t = n * dt
            s = update_h(s, materials, dt, dx)
            s = update_e(s, materials, dt, dx)
            s = apply_pec(s)
            # Inline the differentiated Gaussian pulse
            arg = (t - p_t0) / p_tau
            val = p_amp * (-2.0 * arg) * jnp.exp(-(arg ** 2))
            ez = s.ez.at[cx, cy, cz].add(val)
            return s._replace(ez=ez)
        return jax.lax.fori_loop(0, n_steps, body, state)

    return run_n_steps


# ---------- Benchmark runner ----------

def run_benchmark(grid_sizes=None, num_steps=200):
    import jax

    if grid_sizes is None:
        grid_sizes = [
            (0.05, 0.05, 0.05, 0.005),   # ~10³ = 1k cells
            (0.05, 0.05, 0.05, 0.002),    # ~25³ = 15k cells
            (0.10, 0.10, 0.05, 0.002),    # ~50x50x25 = 62k cells
            (0.10, 0.10, 0.10, 0.002),    # ~50³ = 125k cells
        ]

    device = jax.devices()[0]
    print(f"{'='*65}")
    print(f"  rfx Benchmark: JAX (@jax.jit) vs Pure NumPy")
    print(f"  {num_steps} timesteps per run")
    print(f"  JAX device: {device}")
    print(f"{'='*65}")

    for domain_x, domain_y, domain_z, dx_val in grid_sizes:
        grid = Grid(freq_max=5e9,
                    domain=(domain_x, domain_y, domain_z),
                    dx=dx_val, cpml_layers=0)
        shape = grid.shape
        n_cells = shape[0] * shape[1] * shape[2]
        dt, dx = grid.dt, grid.dx

        pulse = GaussianPulse(f0=2e9, bandwidth=0.5)
        cx, cy, cz = shape[0] // 2, shape[1] // 2, shape[2] // 2

        # --- JAX Python-loop run (with warmup) ---
        state = init_state(shape)
        materials = init_materials(shape)

        # Warmup: first call triggers JIT compilation
        state = update_h(state, materials, dt, dx)
        state = update_e(state, materials, dt, dx)
        state = apply_pec(state)
        float(state.ez[cx, cy, cz])

        state = init_state(shape)
        t0 = time.perf_counter()
        for n in range(num_steps):
            t = n * dt
            state = update_h(state, materials, dt, dx)
            state = update_e(state, materials, dt, dx)
            state = apply_pec(state)
            ez = state.ez.at[cx, cy, cz].add(pulse(t))
            state = state._replace(ez=ez)
        float(state.ez[cx, cy, cz])
        t_jax_loop = time.perf_counter() - t0

        # --- JAX fori_loop run (fully compiled) ---
        run_fori = make_jax_fori_step(materials, dt, dx, cx, cy, cz, pulse)
        state = init_state(shape)
        # Warmup (includes tracing + compilation)
        _ = run_fori(state, 2)
        float(_.ez[cx, cy, cz])

        state = init_state(shape)
        t0 = time.perf_counter()
        state = run_fori(state, num_steps)
        float(state.ez[cx, cy, cz])
        t_jax_fori = time.perf_counter() - t0

        # --- NumPy run ---
        ex = np.zeros(shape, dtype=np.float32)
        ey = np.zeros(shape, dtype=np.float32)
        ez_np = np.zeros(shape, dtype=np.float32)
        hx = np.zeros(shape, dtype=np.float32)
        hy = np.zeros(shape, dtype=np.float32)
        hz = np.zeros(shape, dtype=np.float32)
        mu = np.full(shape, MU_0, dtype=np.float32)
        eps = np.full(shape, EPS_0, dtype=np.float32)
        sigma = np.zeros(shape, dtype=np.float32)

        t0 = time.perf_counter()
        for n in range(num_steps):
            t = n * dt
            hx, hy, hz = np_update_h(ex, ey, ez_np, hx, hy, hz, mu, dt, dx)
            ex, ey, ez_np = np_update_e(ex, ey, ez_np, hx, hy, hz, eps, sigma, dt, dx)
            ex, ey, ez_np = np_apply_pec(ex, ey, ez_np)
            ez_np[cx, cy, cz] += pulse(t)
        t_numpy = time.perf_counter() - t0

        sp_loop = t_numpy / t_jax_loop if t_jax_loop > 0 else float('inf')
        sp_fori = t_numpy / t_jax_fori if t_jax_fori > 0 else float('inf')

        print(f"\n  Grid: {shape[0]}x{shape[1]}x{shape[2]} = {n_cells:,} cells")
        print(f"  NumPy:              {t_numpy:.3f}s  ({num_steps/t_numpy:.0f} steps/s)")
        print(f"  JAX (Python loop):  {t_jax_loop:.3f}s  ({num_steps/t_jax_loop:.0f} steps/s)  {sp_loop:.2f}x")
        print(f"  JAX (fori_loop):    {t_jax_fori:.3f}s  ({num_steps/t_jax_fori:.0f} steps/s)  {sp_fori:.2f}x")

    print(f"\n{'='*65}")
    print("  Speedup values > 1.0 mean JAX is faster than NumPy.")
    print("  fori_loop eliminates Python loop overhead (full XLA fusion).")
    print("  GPU would give much larger speedups at bigger grid sizes.")
    print(f"{'='*65}")


if __name__ == "__main__":
    run_benchmark()
