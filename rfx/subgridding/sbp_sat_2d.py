"""2D TM SBP-SAT FDTD subgridding (Phase 2).

Extends the 1D prototype to 2D TMz mode (Ez, Hx, Hy) with a rectangular
refinement region. The coarse-fine interface has four sides.

Based on: Cheng et al., IEEE TAP 2023
"Toward the 2-D Stable FDTD Subgridding Method With SBP-SAT and
Arbitrary Grid Ratio"
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from rfx.core.yee import EPS_0, MU_0

C0 = 1.0 / np.sqrt(EPS_0 * MU_0)


class SubgridConfig2D(NamedTuple):
    """Configuration for 2D TM SBP-SAT subgridding."""
    # Coarse grid covers the full domain
    nx_c: int
    ny_c: int
    dx_c: float
    dt: float
    # Fine region bounds (in coarse grid indices)
    fi_lo: int          # coarse i-index where fine region starts
    fi_hi: int          # coarse i-index where fine region ends
    fj_lo: int
    fj_hi: int
    # Fine grid
    nx_f: int
    ny_f: int
    dx_f: float
    ratio: int
    tau: float


class SubgridState2D(NamedTuple):
    """State for 2D TM subgridded domain."""
    # Coarse grid: full domain
    ez_c: jnp.ndarray   # (nx_c, ny_c)
    hx_c: jnp.ndarray   # (nx_c, ny_c-1)
    hy_c: jnp.ndarray   # (nx_c-1, ny_c)
    # Fine grid: refinement region only
    ez_f: jnp.ndarray   # (nx_f, ny_f)
    hx_f: jnp.ndarray   # (nx_f, ny_f-1)
    hy_f: jnp.ndarray   # (nx_f-1, ny_f)
    step: int


def init_subgrid_2d(
    nx_c: int = 60,
    ny_c: int = 60,
    dx_c: float = 0.003,
    fine_region: tuple[int, int, int, int] = (20, 40, 20, 40),
    ratio: int = 3,
    courant: float = 0.45,
) -> tuple[SubgridConfig2D, SubgridState2D]:
    """Initialize 2D TM subgridded domain.

    Parameters
    ----------
    nx_c, ny_c : coarse grid dimensions
    dx_c : coarse cell size
    fine_region : (i_lo, i_hi, j_lo, j_hi) in coarse indices
    ratio : grid refinement ratio
    courant : Courant number (< 1/sqrt(2) for 2D stability)
    """
    fi_lo, fi_hi, fj_lo, fj_hi = fine_region
    dx_f = dx_c / ratio
    dt = courant * dx_f / (C0 * np.sqrt(2))

    # Fine grid dimensions
    nx_f = (fi_hi - fi_lo) * ratio
    ny_f = (fj_hi - fj_lo) * ratio

    config = SubgridConfig2D(
        nx_c=nx_c, ny_c=ny_c, dx_c=dx_c, dt=dt,
        fi_lo=fi_lo, fi_hi=fi_hi, fj_lo=fj_lo, fj_hi=fj_hi,
        nx_f=nx_f, ny_f=ny_f, dx_f=dx_f,
        ratio=ratio, tau=0.5,
    )

    state = SubgridState2D(
        ez_c=jnp.zeros((nx_c, ny_c), dtype=jnp.float32),
        hx_c=jnp.zeros((nx_c, ny_c - 1), dtype=jnp.float32),
        hy_c=jnp.zeros((nx_c - 1, ny_c), dtype=jnp.float32),
        ez_f=jnp.zeros((nx_f, ny_f), dtype=jnp.float32),
        hx_f=jnp.zeros((nx_f, ny_f - 1), dtype=jnp.float32),
        hy_f=jnp.zeros((nx_f - 1, ny_f), dtype=jnp.float32),
        step=0,
    )

    return config, state


def _update_h_2d(ez, hx, hy, dt, dx):
    """2D TM H update: Hx, Hy from curl(Ez)."""
    # Hx = Hx - dt/mu0 * dEz/dy
    dez_dy = (ez[:, 1:] - ez[:, :-1]) / dx
    hx_new = hx - (dt / MU_0) * dez_dy
    # Hy = Hy + dt/mu0 * dEz/dx
    dez_dx = (ez[1:, :] - ez[:-1, :]) / dx
    hy_new = hy + (dt / MU_0) * dez_dx
    return hx_new, hy_new


def _update_e_2d(ez, hx, hy, dt, dx):
    """2D TM E update: Ez from curl(H). Interior only."""
    # dHy/dx - dHx/dy
    dhy_dx = (hy[1:, :] - hy[:-1, :]) / dx  # (nx-2, ny)
    dhx_dy = (hx[:, 1:] - hx[:, :-1]) / dx  # (nx, ny-2)

    # Interior update (1:-1 in both dims to avoid boundary)
    # Shapes must match: use the overlap region
    nx, ny = ez.shape
    curl_h = dhy_dx[:, 1:-1] - dhx_dy[1:-1, :]
    ez_new = ez.at[1:-1, 1:-1].add((dt / EPS_0) * curl_h)
    return ez_new


def _apply_pec_2d(ez):
    """Zero Ez on all boundaries (PEC walls)."""
    ez = ez.at[0, :].set(0.0)
    ez = ez.at[-1, :].set(0.0)
    ez = ez.at[:, 0].set(0.0)
    ez = ez.at[:, -1].set(0.0)
    return ez


def step_subgrid_2d(
    state: SubgridState2D,
    config: SubgridConfig2D,
    source_val: float = 0.0,
    source_pos_c: tuple[int, int] = (-1, -1),
) -> SubgridState2D:
    """One coupled timestep of coarse + fine 2D TM grids."""
    dt = config.dt
    dx_c, dx_f = config.dx_c, config.dx_f
    ratio = config.ratio
    tau = config.tau

    ez_c, hx_c, hy_c = state.ez_c, state.hx_c, state.hy_c
    ez_f, hx_f, hy_f = state.ez_f, state.hx_f, state.hy_f

    # === Fine grid: ratio sub-steps ===
    for _ in range(ratio):
        hx_f, hy_f = _update_h_2d(ez_f, hx_f, hy_f, dt, dx_f)
        ez_f = _update_e_2d(ez_f, hx_f, hy_f, dt, dx_f)
        ez_f = _apply_pec_2d(ez_f)

    # === Coarse grid: one step with dt_c = ratio * dt ===
    dt_c = ratio * dt
    hx_c, hy_c = _update_h_2d(ez_c, hx_c, hy_c, dt_c, dx_c)
    ez_c = _update_e_2d(ez_c, hx_c, hy_c, dt_c, dx_c)
    ez_c = _apply_pec_2d(ez_c)

    # === Source injection ===
    if source_pos_c[0] >= 0:
        ez_c = ez_c.at[source_pos_c].add(source_val)

    # === SAT interface coupling (vectorized, 4 sides) ===
    # SAT penalty in field space: correction = tau * (field_diff)
    # No dt/eps scaling — the penalty acts directly on field values.
    sat_scale_c = tau
    sat_scale_f = tau

    # Downsample fine boundary → coarse resolution by averaging blocks of `ratio`
    def _downsample(fine_edge, n_coarse_cells):
        """Average fine boundary values into coarse cell blocks."""
        trimmed = fine_edge[:n_coarse_cells * ratio]
        return jnp.mean(trimmed.reshape(n_coarse_cells, ratio), axis=1)

    def _upsample_diff(diff_coarse, n_fine_cells, r):
        """Broadcast coarse difference to fine cells (constant per block)."""
        return jnp.repeat(diff_coarse, r)[:n_fine_cells]

    ni = config.fi_hi - config.fi_lo  # coarse cells along i at interface
    nj = config.fj_hi - config.fj_lo  # coarse cells along j at interface

    # Left side: coarse ez_c[fi_lo, fj_lo:fj_hi] ↔ fine ez_f[0, :]
    ec_left = ez_c[config.fi_lo, config.fj_lo:config.fj_hi]
    ef_left = _downsample(ez_f[0, :], nj)
    diff_left = ec_left - ef_left
    ez_c = ez_c.at[config.fi_lo, config.fj_lo:config.fj_hi].add(-sat_scale_c * diff_left)
    ez_f = ez_f.at[0, :config.ny_f].add(sat_scale_f * _upsample_diff(diff_left, config.ny_f, ratio) / ratio)

    # Right side: coarse ez_c[fi_hi-1, fj_lo:fj_hi] ↔ fine ez_f[-1, :]
    ec_right = ez_c[config.fi_hi - 1, config.fj_lo:config.fj_hi]
    ef_right = _downsample(ez_f[-1, :], nj)
    diff_right = ec_right - ef_right
    ez_c = ez_c.at[config.fi_hi - 1, config.fj_lo:config.fj_hi].add(-sat_scale_c * diff_right)
    ez_f = ez_f.at[-1, :config.ny_f].add(sat_scale_f * _upsample_diff(diff_right, config.ny_f, ratio) / ratio)

    # Bottom side: coarse ez_c[fi_lo:fi_hi, fj_lo] ↔ fine ez_f[:, 0]
    ec_bot = ez_c[config.fi_lo:config.fi_hi, config.fj_lo]
    ef_bot = _downsample(ez_f[:, 0], ni)
    diff_bot = ec_bot - ef_bot
    ez_c = ez_c.at[config.fi_lo:config.fi_hi, config.fj_lo].add(-sat_scale_c * diff_bot)
    ez_f = ez_f.at[:config.nx_f, 0].add(sat_scale_f * _upsample_diff(diff_bot, config.nx_f, ratio) / ratio)

    # Top side: coarse ez_c[fi_lo:fi_hi, fj_hi-1] ↔ fine ez_f[:, -1]
    ec_top = ez_c[config.fi_lo:config.fi_hi, config.fj_hi - 1]
    ef_top = _downsample(ez_f[:, -1], ni)
    diff_top = ec_top - ef_top
    ez_c = ez_c.at[config.fi_lo:config.fi_hi, config.fj_hi - 1].add(-sat_scale_c * diff_top)
    ez_f = ez_f.at[:config.nx_f, -1].add(sat_scale_f * _upsample_diff(diff_top, config.nx_f, ratio) / ratio)

    return SubgridState2D(
        ez_c=ez_c, hx_c=hx_c, hy_c=hy_c,
        ez_f=ez_f, hx_f=hx_f, hy_f=hy_f,
        step=state.step + 1,
    )


def compute_energy_2d(state: SubgridState2D, config: SubgridConfig2D) -> float:
    """Total discrete energy for 2D TM subgridded domain."""
    e_c = (float(jnp.sum(state.ez_c ** 2)) * EPS_0 * config.dx_c ** 2 +
           float(jnp.sum(state.hx_c ** 2)) * MU_0 * config.dx_c ** 2 +
           float(jnp.sum(state.hy_c ** 2)) * MU_0 * config.dx_c ** 2)
    e_f = (float(jnp.sum(state.ez_f ** 2)) * EPS_0 * config.dx_f ** 2 +
           float(jnp.sum(state.hx_f ** 2)) * MU_0 * config.dx_f ** 2 +
           float(jnp.sum(state.hy_f ** 2)) * MU_0 * config.dx_f ** 2)
    return e_c + e_f
