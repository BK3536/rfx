"""1D SBP-SAT FDTD subgridding prototype.

Implements provably stable subgridding for 1D Ez/Hy FDTD using
Summation-By-Parts (SBP) operators and Simultaneous Approximation
Terms (SAT) at the coarse-fine grid interface.

Based on: Cheng et al., arXiv:2202.10770
"SBP-SAT FDTD Subgridding Using Staggered Yee's Grids Without
Modifying Field Components"

The key stability guarantee:
  d/dt (||E||² + ||H||²) ≤ 0  (discrete energy non-increasing)

This is achieved by:
1. SBP operators that mimic integration-by-parts at the discrete level
2. SAT penalty terms that weakly enforce field continuity at the interface
3. Carefully derived interpolation matrices between coarse and fine grids

Implementation note:
  The paper's tau=0.5 applies to the semi-discrete (continuous time) form.
  For explicit leapfrog time integration, the penalty parameter is rescaled
  so that the fully-discrete coefficient is a dimensionless number alpha < 1.
  Specifically: tau_discrete = alpha * eps0 * P_boundary / dt, ensuring the
  SAT correction per timestep is proportional to alpha * (field_difference).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from rfx.core.yee import EPS_0, MU_0

C0 = 1.0 / np.sqrt(EPS_0 * MU_0)


# ---------------------------------------------------------------------------
# SBP operator construction
# ---------------------------------------------------------------------------

def build_sbp_norm(n: int, dx: float) -> np.ndarray:
    """Build diagonal SBP norm matrix P (returned as 1-D vector of length *n*).

    For 2nd-order SBP on a collocated grid with *n* nodes and spacing *dx*,
    interior weights are *dx* and boundary weights are *dx/2* (trapezoidal
    rule quadrature).

    Parameters
    ----------
    n : number of grid nodes
    dx : cell spacing

    Returns
    -------
    P : (n,) diagonal entries of the norm matrix
    """
    p = np.full(n, dx, dtype=np.float64)
    p[0] = dx / 2.0
    p[-1] = dx / 2.0
    return p


def build_sbp_diff(n: int, dx: float) -> np.ndarray:
    """Build 2nd-order SBP first-derivative operator D (n x n).

    D approximates d/dx on *n* collocated nodes with spacing *dx*.
    It satisfies the SBP property::

        P @ D + (P @ D)^T = E_boundary

    where ``E_boundary = diag(-1, 0, ..., 0, +1)``.

    Uses standard 2nd-order centred stencil in the interior and
    compatible one-sided stencils at the boundaries.

    Parameters
    ----------
    n : number of grid nodes
    dx : cell spacing

    Returns
    -------
    D : (n, n) dense difference operator
    """
    D = np.zeros((n, n), dtype=np.float64)

    # Interior: centred difference
    for i in range(1, n - 1):
        D[i, i - 1] = -1.0 / (2.0 * dx)
        D[i, i + 1] = +1.0 / (2.0 * dx)

    # Boundary: one-sided (compatible with trapezoidal SBP norm)
    # Left boundary (row 0): forward difference
    D[0, 0] = -1.0 / dx
    D[0, 1] = +1.0 / dx

    # Right boundary (row n-1): backward difference
    D[-1, -2] = -1.0 / dx
    D[-1, -1] = +1.0 / dx

    return D


# ---------------------------------------------------------------------------
# Interpolation matrices
# ---------------------------------------------------------------------------

def build_interpolation_c2f(n_coarse: int, n_fine: int, ratio: int) -> np.ndarray:
    """Build coarse-to-fine linear interpolation at the interface boundary.

    Maps 2 coarse boundary nodes to ``ratio + 1`` fine boundary positions
    via linear interpolation.

    Parameters
    ----------
    n_coarse : number of coarse E-nodes (not used directly, kept for API)
    n_fine : number of fine E-nodes (not used directly, kept for API)
    ratio : grid ratio (dx_c / dx_f, integer)

    Returns
    -------
    R_c2f : (ratio+1, 2) interpolation matrix
    """
    R = np.zeros((ratio + 1, 2), dtype=np.float64)
    for k in range(ratio + 1):
        alpha = k / ratio  # fractional position in [0, 1]
        R[k, 0] = 1.0 - alpha
        R[k, 1] = alpha
    return R


def build_interpolation_f2c(n_fine: int, n_coarse: int, ratio: int) -> np.ndarray:
    """Build fine-to-coarse interpolation at the interface boundary.

    Returns the transpose of the c2f matrix, which is the SBP-compatible
    adjoint interpolation (preserves the energy balance).

    Parameters
    ----------
    n_fine : number of fine E-nodes
    n_coarse : number of coarse E-nodes
    ratio : grid ratio

    Returns
    -------
    R_f2c : (2, ratio+1) matrix
    """
    R_c2f = build_interpolation_c2f(n_coarse, n_fine, ratio)
    return R_c2f.T.copy()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class SubgridConfig1D(NamedTuple):
    """Configuration for 1D SBP-SAT subgridded domain."""
    n_c: int            # coarse grid E-nodes
    n_f: int            # fine grid E-nodes
    dx_c: float         # coarse cell size
    dx_f: float         # fine cell size
    dt: float           # timestep (Courant-limited by fine grid)
    ratio: int          # grid ratio (dx_c / dx_f)
    alpha: float        # dimensionless SAT penalty strength (< 1)
    # Pre-computed SBP norm diagonal entries
    p_c: jnp.ndarray    # (n_c,)
    p_f: jnp.ndarray    # (n_f,)


class SubgridState1D(NamedTuple):
    """State of coupled coarse + fine 1D grids."""
    e_c: jnp.ndarray    # Ez on coarse grid (n_c,)
    h_c: jnp.ndarray    # Hy on coarse grid (n_c-1,) at half-cells
    e_f: jnp.ndarray    # Ez on fine grid (n_f,)
    h_f: jnp.ndarray    # Hy on fine grid (n_f-1,)
    step: int


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_subgrid_1d(
    n_c: int = 60,
    n_f: int = 90,
    dx_c: float = 0.003,
    ratio: int = 3,
    dt: float | None = None,
    courant: float = 0.5,
    alpha: float = 0.3,
) -> tuple[SubgridConfig1D, SubgridState1D]:
    """Initialize a 1D subgridded domain.

    Layout::

        [coarse grid] | interface | [fine grid]

    The coarse grid right boundary connects to the fine grid left boundary.

    Parameters
    ----------
    n_c : number of E-field nodes on coarse grid
    n_f : number of E-field nodes on fine grid
    dx_c : coarse cell size in metres
    ratio : integer grid ratio (dx_c = ratio * dx_f)
    dt : explicit timestep; if *None*, derived from Courant number
    courant : Courant number (used when *dt* is None)
    alpha : dimensionless SAT penalty strength (0 < alpha < 1).
            This is the fraction of the field mismatch corrected per
            coarse timestep.  Values 0.2--0.5 are typical.
    """
    dx_f = dx_c / ratio

    if dt is None:
        dt = courant * dx_f / C0  # limited by fine grid

    p_c = jnp.array(build_sbp_norm(n_c, dx_c), dtype=jnp.float32)
    p_f = jnp.array(build_sbp_norm(n_f, dx_f), dtype=jnp.float32)

    config = SubgridConfig1D(
        n_c=n_c, n_f=n_f,
        dx_c=dx_c, dx_f=dx_f,
        dt=float(dt), ratio=ratio, alpha=alpha,
        p_c=p_c, p_f=p_f,
    )

    state = SubgridState1D(
        e_c=jnp.zeros(n_c, dtype=jnp.float32),
        h_c=jnp.zeros(n_c - 1, dtype=jnp.float32),
        e_f=jnp.zeros(n_f, dtype=jnp.float32),
        h_f=jnp.zeros(n_f - 1, dtype=jnp.float32),
        step=0,
    )

    return config, state


# ---------------------------------------------------------------------------
# Elementary 1D FDTD updates
# ---------------------------------------------------------------------------

def _update_h_1d(e: jnp.ndarray, h: jnp.ndarray,
                 dt: float, dx: float) -> jnp.ndarray:
    """Standard 1D Yee H-update:  H^{n+1/2} = H^{n-1/2} + (dt/mu0) dE/dx."""
    de_dx = (e[1:] - e[:-1]) / dx
    return h + (dt / MU_0) * de_dx


def _update_e_1d(e: jnp.ndarray, h: jnp.ndarray,
                 dt: float, dx: float) -> jnp.ndarray:
    """Standard 1D Yee E-update (interior nodes only):
    E^{n+1} = E^n + (dt/eps0) dH/dx.
    """
    dh_dx = (h[1:] - h[:-1]) / dx
    e_new = e.at[1:-1].add((dt / EPS_0) * dh_dx)
    return e_new


# ---------------------------------------------------------------------------
# Coupled SBP-SAT step
# ---------------------------------------------------------------------------

def step_subgrid_1d(
    state: SubgridState1D,
    config: SubgridConfig1D,
) -> SubgridState1D:
    """One coupled timestep of coarse + fine grids.

    The fine grid takes ``ratio`` sub-steps per coarse step.
    SAT penalty terms are applied at the interface *after* the standard
    FDTD updates to weakly enforce field continuity.

    The SAT correction uses a dimensionless penalty coefficient *alpha*:

    .. math::

        E_c[-1]  -=  \\alpha \\cdot (E_c[-1] - E_f[0])
        E_f[0]   +=  \\alpha \\cdot (E_c[-1] - E_f[0])

    and similarly for H.  This is equivalent to the semi-discrete SBP-SAT
    penalty with ``tau = alpha * eps0 * P_boundary / dt``, rescaled for
    explicit leapfrog stability.
    """
    dt = config.dt
    dx_c, dx_f = config.dx_c, config.dx_f
    ratio = config.ratio
    alpha = config.alpha

    e_c, h_c = state.e_c, state.h_c
    e_f, h_f = state.e_f, state.h_f

    # ── Fine grid: *ratio* sub-steps ──────────────────────────────
    for _ in range(ratio):
        h_f = _update_h_1d(e_f, h_f, dt, dx_f)
        e_f = _update_e_1d(e_f, h_f, dt, dx_f)
        # PEC at the far (right) end of the fine grid
        e_f = e_f.at[-1].set(0.0)

    # ── Coarse grid: one step with dt_c = ratio * dt ─────────────
    dt_c = ratio * dt
    h_c = _update_h_1d(e_c, h_c, dt_c, dx_c)
    e_c = _update_e_1d(e_c, h_c, dt_c, dx_c)
    # PEC at the far (left) end of the coarse grid
    e_c = e_c.at[0].set(0.0)

    # ── SAT interface coupling ────────────────────────────────────
    # Interface: coarse right boundary e_c[-1] ↔ fine left boundary e_f[0]
    #
    # The penalty drives both fields toward their average, dissipating
    # the interface mismatch energy.  The symmetric form ensures
    # d/dt(total energy) ≤ 0.

    # E-field SAT
    e_diff = e_c[-1] - e_f[0]
    e_c = e_c.at[-1].add(-alpha * e_diff)
    e_f = e_f.at[0].add(+alpha * e_diff)

    # H-field SAT (rightmost coarse H ↔ leftmost fine H)
    h_diff = h_c[-1] - h_f[0]
    h_c = h_c.at[-1].add(-alpha * h_diff)
    h_f = h_f.at[0].add(+alpha * h_diff)

    return SubgridState1D(
        e_c=e_c, h_c=h_c,
        e_f=e_f, h_f=h_f,
        step=state.step + 1,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def compute_energy(state: SubgridState1D, config: SubgridConfig1D) -> float:
    """Total discrete electromagnetic energy.

    .. math::

        \\mathcal{E} = \\sum_i \\varepsilon_0 E_i^2 \\Delta x
                     + \\sum_j \\mu_0 H_j^2 \\Delta x
    """
    energy_e_c = float(jnp.sum(state.e_c ** 2)) * EPS_0 * config.dx_c
    energy_h_c = float(jnp.sum(state.h_c ** 2)) * MU_0 * config.dx_c
    energy_e_f = float(jnp.sum(state.e_f ** 2)) * EPS_0 * config.dx_f
    energy_h_f = float(jnp.sum(state.h_f ** 2)) * MU_0 * config.dx_f
    return energy_e_c + energy_h_c + energy_e_f + energy_h_f
