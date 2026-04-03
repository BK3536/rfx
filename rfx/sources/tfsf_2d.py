"""2D auxiliary grid for oblique-incidence TFSF plane waves.

Uses a 2D TMz Yee grid with the same cell size ``dx`` as the 3D
simulation so numerical dispersion matches exactly at any angle.

The 2D grid is **periodic in y** (matching the 3D grid) and uses
CFS-CPML only on the x-lo and x-hi boundaries.  This avoids the
grazing-angle CPML reflection problem that arises when all four
edges use absorbing boundaries.

The oblique plane wave source is a soft line source at a fixed x
with per-y-cell time delay that encodes the oblique angle.

Reference: S. C. Tan and G. E. Potter, IEEE T-AP 58(9), 2010.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from rfx.core.yee import EPS_0, MU_0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class TFSF2DState(NamedTuple):
    """2D auxiliary grid state for oblique TFSF."""
    ez_2d: jnp.ndarray   # (n2x, n2y)
    hx_2d: jnp.ndarray   # (n2x, n2y)
    hy_2d: jnp.ndarray   # (n2x, n2y)
    # CFS-CPML psi arrays (x-direction only, 4 total)
    psi_ez_xlo: jnp.ndarray  # (n_cpml, n2y)
    psi_ez_xhi: jnp.ndarray  # (n_cpml, n2y)
    psi_hy_xlo: jnp.ndarray  # (n_cpml, n2y)
    psi_hy_xhi: jnp.ndarray  # (n_cpml, n2y)
    step: jnp.ndarray


class TFSF2DConfig(NamedTuple):
    """Configuration for 2D auxiliary grid TFSF."""
    x_lo: int
    x_hi: int
    n2x: int
    n2y: int          # == ny (3D), periodic in y
    i0_x: int         # 2D x-index mapping to 3D x_lo
    i0_y: int         # always 0 (periodic y, no padding)
    src_x: int
    src_amp: float
    src_t0: float
    src_tau: float
    theta: float
    cos_theta: float
    sin_theta: float
    electric_component: str
    magnetic_component: str
    curl_sign: float
    direction: str
    direction_sign: float
    transverse_axis: str
    n_cpml: int
    b_cpml: jnp.ndarray
    c_cpml: jnp.ndarray
    kappa_cpml: jnp.ndarray
    grid_pad: int
    angle_deg: float
    dx_1d: float


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_tfsf_2d(
    nx: int,
    ny: int,
    dx: float,
    dt: float,
    *,
    cpml_layers: int = 0,
    tfsf_margin: int = 3,
    f0: float = 5e9,
    bandwidth: float = 0.5,
    amplitude: float = 1.0,
    polarization: str = "ez",
    direction: str = "+x",
    theta_deg: float = 0.0,
) -> tuple[TFSF2DConfig, TFSF2DState]:
    """Initialize 2D auxiliary grid for oblique-incidence TFSF.

    The 2D grid uses periodic BC in y (matching the 3D grid) and
    CFS-CPML only on x boundaries.
    """
    if polarization not in ("ez", "ey"):
        raise ValueError(f"polarization must be 'ez' or 'ey', got {polarization!r}")
    if direction not in ("+x", "-x"):
        raise ValueError(f"direction must be '+x' or '-x', got {direction!r}")
    if abs(theta_deg) >= 90.0:
        raise ValueError(f"|angle_deg| must be < 90, got {theta_deg}")

    theta = np.radians(theta_deg)
    cos_theta = float(np.cos(theta))
    sin_theta = float(np.sin(theta))

    # TFSF box boundaries in x
    offset = cpml_layers + tfsf_margin
    x_lo = offset
    x_hi = nx - offset - 1

    if x_lo <= 0 or x_hi >= nx - 1 or x_lo >= x_hi:
        raise ValueError(
            "TFSF margin/cpml_layers too large for the grid: "
            f"nx={nx}, cpml_layers={cpml_layers}, tfsf_margin={tfsf_margin}"
        )

    # ---- 2D grid sizing ----
    # y-direction: periodic, same size as 3D grid
    n2y = ny

    # x-direction: TFSF span + margins + CPML on both ends
    n_cpml_2d = 30
    n_tfsf_x = x_hi - x_lo + 2
    n_margin_x = 25

    n2x = n_cpml_2d + n_margin_x + n_tfsf_x + n_margin_x + n_cpml_2d

    # i0_x: 2D x-index that maps to 3D x_lo
    i0_x = n_cpml_2d + n_margin_x
    i0_y = 0  # periodic y, no padding

    # Source position
    if direction == "+x":
        src_x = n_cpml_2d + 3
    else:
        src_x = n2x - n_cpml_2d - 4

    # CFS-CPML profile (x-direction only), 4th order
    cpml_order = 4
    kappa_max = 7.0
    eta = np.sqrt(MU_0 / EPS_0)
    sigma_max = 0.8 * (cpml_order + 1) / (eta * dx) * kappa_max
    rho = 1.0 - np.arange(n_cpml_2d, dtype=np.float64) / max(n_cpml_2d - 1, 1)
    sigma_prof = sigma_max * rho ** cpml_order
    kappa_prof = 1.0 + (kappa_max - 1.0) * rho ** cpml_order
    alpha_prof = 0.05 * (1.0 - rho)
    denom = sigma_prof * kappa_prof + kappa_prof ** 2 * alpha_prof
    b_prof = np.exp(-(sigma_prof / kappa_prof + alpha_prof) * dt / EPS_0)
    c_prof = np.where(denom > 1e-30, sigma_prof * (b_prof - 1.0) / denom, 0.0)

    # Source waveform
    tau = 1.0 / (f0 * bandwidth * np.pi)
    t0 = 3.0 * tau

    if polarization == "ez":
        electric_component = "ez"
        magnetic_component = "hy"
        curl_sign = 1.0
        transverse_axis = "y"
    else:
        electric_component = "ey"
        magnetic_component = "hz"
        curl_sign = -1.0
        transverse_axis = "z"

    direction_sign = 1.0 if direction == "+x" else -1.0

    config = TFSF2DConfig(
        x_lo=x_lo, x_hi=x_hi,
        n2x=n2x, n2y=n2y,
        i0_x=i0_x, i0_y=i0_y,
        src_x=src_x,
        src_amp=float(amplitude),
        src_t0=float(t0),
        src_tau=float(tau),
        theta=float(theta),
        cos_theta=cos_theta,
        sin_theta=sin_theta,
        electric_component=electric_component,
        magnetic_component=magnetic_component,
        curl_sign=float(curl_sign),
        direction=direction,
        direction_sign=float(direction_sign),
        transverse_axis=transverse_axis,
        n_cpml=n_cpml_2d,
        b_cpml=jnp.array(b_prof, dtype=jnp.float32),
        c_cpml=jnp.array(c_prof, dtype=jnp.float32),
        kappa_cpml=jnp.array(kappa_prof, dtype=jnp.float32),
        grid_pad=cpml_layers,
        angle_deg=float(theta_deg),
        dx_1d=float(dx),
    )

    state = TFSF2DState(
        ez_2d=jnp.zeros((n2x, n2y), dtype=jnp.float32),
        hx_2d=jnp.zeros((n2x, n2y), dtype=jnp.float32),
        hy_2d=jnp.zeros((n2x, n2y), dtype=jnp.float32),
        psi_ez_xlo=jnp.zeros((n_cpml_2d, n2y), dtype=jnp.float32),
        psi_ez_xhi=jnp.zeros((n_cpml_2d, n2y), dtype=jnp.float32),
        psi_hy_xlo=jnp.zeros((n_cpml_2d, n2y), dtype=jnp.float32),
        psi_hy_xhi=jnp.zeros((n_cpml_2d, n2y), dtype=jnp.float32),
        step=jnp.array(0, dtype=jnp.int32),
    )

    return config, state


# ---------------------------------------------------------------------------
# 2D Yee update: TMz mode (Ez, Hx, Hy) — periodic y, CPML x
# ---------------------------------------------------------------------------

def update_tfsf_2d_h(cfg: TFSF2DConfig, st: TFSF2DState,
                      dx: float, dt: float) -> TFSF2DState:
    """Advance 2D auxiliary H: H^{n-1/2} -> H^{n+1/2}.

    TMz: Hx -= (dt/mu0)*dEz/dy,  Hy += (dt/mu0)*dEz/dx
    y-direction uses periodic (roll), x-direction uses zero-pad + CFS-CPML.
    """
    n = cfg.n_cpml
    ez = st.ez_2d
    hx, hy = st.hx_2d, st.hy_2d
    coeff_h = dt / MU_0

    # dEz/dy — periodic (roll -1 along y)
    dez_dy = (jnp.roll(ez, -1, axis=1) - ez) / dx

    # dEz/dx — zero-pad forward difference along x
    dez_dx = (jnp.concatenate([ez[1:, :], jnp.zeros((1, ez.shape[1]))], axis=0) - ez) / dx

    hx = hx - coeff_h * dez_dy
    hy = hy + coeff_h * dez_dx

    # CFS-CPML for Hy (x-direction only)
    b_lo = cfg.b_cpml[:, None]
    c_lo = cfg.c_cpml[:, None]
    k_lo = cfg.kappa_cpml[:, None]

    psi_hy_xlo = b_lo * st.psi_hy_xlo + c_lo * dez_dx[:n, :]
    hy = hy.at[:n, :].add(coeff_h * psi_hy_xlo)
    hy = hy.at[:n, :].add(coeff_h * (1.0 / k_lo - 1.0) * dez_dx[:n, :])

    b_hi = jnp.flip(cfg.b_cpml)[:, None]
    c_hi = jnp.flip(cfg.c_cpml)[:, None]
    k_hi = jnp.flip(cfg.kappa_cpml)[:, None]

    psi_hy_xhi = b_hi * st.psi_hy_xhi + c_hi * dez_dx[-n:, :]
    hy = hy.at[-n:, :].add(coeff_h * psi_hy_xhi)
    hy = hy.at[-n:, :].add(coeff_h * (1.0 / k_hi - 1.0) * dez_dx[-n:, :])

    # No y-CPML: y is periodic
    return st._replace(
        hx_2d=hx, hy_2d=hy,
        psi_hy_xlo=psi_hy_xlo, psi_hy_xhi=psi_hy_xhi,
    )


def update_tfsf_2d_e(cfg: TFSF2DConfig, st: TFSF2DState,
                      dx: float, dt: float, t: float) -> TFSF2DState:
    """Advance 2D auxiliary Ez: Ez^n -> Ez^{n+1} + source injection.

    TMz: Ez += (dt/eps0) * (dHy/dx - dHx/dy)
    y-direction uses periodic (roll), x-direction uses zero-pad + CFS-CPML.
    """
    n = cfg.n_cpml
    ez = st.ez_2d
    hx, hy = st.hx_2d, st.hy_2d
    coeff_e = dt / EPS_0

    # dHy/dx — zero-pad backward difference along x
    dhy_dx = (hy - jnp.concatenate([jnp.zeros((1, hy.shape[1])), hy[:-1, :]], axis=0)) / dx

    # dHx/dy — periodic (roll +1 along y for backward difference)
    dhx_dy = (hx - jnp.roll(hx, 1, axis=1)) / dx

    ez = ez + coeff_e * (dhy_dx - dhx_dy)

    # CFS-CPML for Ez (x-direction, from dHy/dx)
    b_lo = cfg.b_cpml[:, None]
    c_lo = cfg.c_cpml[:, None]
    k_lo = cfg.kappa_cpml[:, None]

    psi_ez_xlo = b_lo * st.psi_ez_xlo + c_lo * dhy_dx[:n, :]
    ez = ez.at[:n, :].add(coeff_e * psi_ez_xlo)
    ez = ez.at[:n, :].add(coeff_e * (1.0 / k_lo - 1.0) * dhy_dx[:n, :])

    b_hi = jnp.flip(cfg.b_cpml)[:, None]
    c_hi = jnp.flip(cfg.c_cpml)[:, None]
    k_hi = jnp.flip(cfg.kappa_cpml)[:, None]

    psi_ez_xhi = b_hi * st.psi_ez_xhi + c_hi * dhy_dx[-n:, :]
    ez = ez.at[-n:, :].add(coeff_e * psi_ez_xhi)
    ez = ez.at[-n:, :].add(coeff_e * (1.0 / k_hi - 1.0) * dhy_dx[-n:, :])

    # No y-CPML: y is periodic

    # ---- Inject oblique plane-wave source along a line ----
    c0 = 1.0 / jnp.sqrt(jnp.float32(EPS_0 * MU_0))
    j_indices = jnp.arange(ez.shape[1], dtype=jnp.float32)
    y_offset = j_indices * dx  # i0_y=0, so y_offset = j * dx
    t_delay = cfg.direction_sign * cfg.sin_theta * y_offset / c0
    arg = (t - t_delay - cfg.src_t0) / cfg.src_tau
    src_line = cfg.src_amp * (-2.0 * arg) * jnp.exp(-(arg ** 2))
    ez = ez.at[cfg.src_x, :].add(src_line)

    return st._replace(
        ez_2d=ez,
        psi_ez_xlo=psi_ez_xlo, psi_ez_xhi=psi_ez_xhi,
        step=st.step + 1,
    )


def update_tfsf_2d(cfg: TFSF2DConfig, st: TFSF2DState,
                    dx: float, dt: float, t: float) -> TFSF2DState:
    """Advance 2D auxiliary grid by one full timestep (convenience)."""
    st = update_tfsf_2d_h(cfg, st, dx, dt)
    st = update_tfsf_2d_e(cfg, st, dx, dt, t)
    return st


# ---------------------------------------------------------------------------
# Sample incident fields from 2D grid
# ---------------------------------------------------------------------------

def _sample_ez_at_x(cfg: TFSF2DConfig, st: TFSF2DState,
                     ix_2d: int, ny_3d: int) -> jnp.ndarray:
    """Sample Ez from 2D grid at x-index for all 3D y cells. Returns (ny_3d,)."""
    return st.ez_2d[ix_2d, :ny_3d]


def _sample_hy_at_x(cfg: TFSF2DConfig, st: TFSF2DState,
                     ix_2d: int, ny_3d: int) -> jnp.ndarray:
    """Sample Hy from 2D grid at x-index for all 3D y cells. Returns (ny_3d,)."""
    return st.hy_2d[ix_2d, :ny_3d]


# ---------------------------------------------------------------------------
# 3D TFSF corrections using 2D auxiliary grid
# ---------------------------------------------------------------------------

def apply_tfsf_2d_e(state, cfg: TFSF2DConfig, tfsf_st: TFSF2DState,
                     dx: float, dt: float):
    """Apply E-field TFSF correction using 2D aux grid (call AFTER update_e).

    Ez[x_lo, j, :] -= (dt/(eps0*dx)) * Hy_inc[x_lo-0.5, j]
    Ez[x_hi+1, j, :] += (dt/(eps0*dx)) * Hy_inc[x_hi+0.5, j]
    """
    coeff = dt / (EPS_0 * dx)
    i0 = cfg.i0_x
    ny_3d = state.ez.shape[1]
    nz_3d = state.ez.shape[2]

    h_inc_lo = _sample_hy_at_x(cfg, tfsf_st, i0 - 1, ny_3d)
    h_inc_hi = _sample_hy_at_x(cfg, tfsf_st, i0 + (cfg.x_hi - cfg.x_lo), ny_3d)

    h_lo_3d = jnp.broadcast_to(h_inc_lo[:, None], (ny_3d, nz_3d))
    h_hi_3d = jnp.broadcast_to(h_inc_hi[:, None], (ny_3d, nz_3d))

    e_field = getattr(state, cfg.electric_component)
    e_field = e_field.at[cfg.x_lo, :, :].add(-cfg.curl_sign * coeff * h_lo_3d)
    e_field = e_field.at[cfg.x_hi + 1, :, :].add(cfg.curl_sign * coeff * h_hi_3d)

    return state._replace(**{cfg.electric_component: e_field})


def apply_tfsf_2d_h(state, cfg: TFSF2DConfig, tfsf_st: TFSF2DState,
                     dx: float, dt: float):
    """Apply H-field TFSF correction using 2D aux grid (call AFTER update_h).

    Hy[x_lo-1, j, :] -= (dt/(mu0*dx)) * Ez_inc[x_lo, j]
    Hy[x_hi, j, :]   += (dt/(mu0*dx)) * Ez_inc[x_hi+1, j]
    """
    coeff = dt / (MU_0 * dx)
    i0 = cfg.i0_x
    ny_3d = state.hy.shape[1]
    nz_3d = state.hy.shape[2]

    e_inc_lo = _sample_ez_at_x(cfg, tfsf_st, i0, ny_3d)
    e_inc_hi = _sample_ez_at_x(cfg, tfsf_st, i0 + (cfg.x_hi + 1 - cfg.x_lo), ny_3d)

    e_lo_3d = jnp.broadcast_to(e_inc_lo[:, None], (ny_3d, nz_3d))
    e_hi_3d = jnp.broadcast_to(e_inc_hi[:, None], (ny_3d, nz_3d))

    h_field = getattr(state, cfg.magnetic_component)
    h_field = h_field.at[cfg.x_lo - 1, :, :].add(-cfg.curl_sign * coeff * e_lo_3d)
    h_field = h_field.at[cfg.x_hi, :, :].add(cfg.curl_sign * coeff * e_hi_3d)

    return state._replace(**{cfg.magnetic_component: h_field})
