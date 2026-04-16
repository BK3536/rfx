"""Phase B: Non-uniform FDTD kernels for the shard_map distributed runner.

This module supplies the NU analogues of the uniform kernels used by
``rfx/runners/distributed_v2.py`` so the distributed path can accept
``NonUniformGrid`` without silently dropping the profile. The uniform
kernels in ``distributed.py`` remain the reference baseline and must
not be modified from this module.

Scope (Phase B minimal):
- x-axis shard only (1D slab decomposition).
- Global grading ratio <= 5:1 (shared single dt).
- x-axis CPML cells are uniform (guaranteed by make_nonuniform_grid
  boundary padding).
- TFSF single-device only (enforced upstream).
- Dispersive (Debye/Lorentz) E on NU distributed is NOT implemented
  here. The public entry point in ``distributed_v2`` falls back when
  dispersion is active.

Key helper: ``_build_sharded_inv_dx_arrays`` returns per-device
slabs of ``inv_dx`` / ``inv_dx_h`` whose slab boundary entry of
``inv_dx_h`` is derived from the global spacing straddling the slab
seam (NOT from the local slab alone) so H-field mean-spacing math
remains consistent across the shard boundary.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from rfx.core.yee import (
    FDTDState,
    MaterialArrays,
    MU_0,
    EPS_0,
    _shift_fwd,
    _shift_bwd,
)


# ---------------------------------------------------------------------------
# Sharded inv-spacing arrays
# ---------------------------------------------------------------------------

def _build_sharded_inv_dx_arrays(grid, n_devices, pad_x=0):
    """Build per-device x-axis inverse-spacing slabs for the shard_map runner.

    The caller has already padded the global x-extent by ``pad_x`` cells
    (to align ``nx`` on ``n_devices``).  We replicate that padding onto
    the cell-size profile using the boundary cell value (matching how
    ``make_nonuniform_grid`` pads CPML cells) and rebuild the global
    ``inv_dx`` and ``inv_dx_h`` from the padded profile, then reshape to
    per-device slabs.

    For ``inv_dx_h``, the last entry of each device's slab is the global
    mean-spacing straddling the slab seam with the next device (or 0 at
    the domain boundary), NOT derived from the local slab alone.

    Parameters
    ----------
    grid : NonUniformGrid
    n_devices : int
    pad_x : int
        Number of PEC-padded cells appended to the high-x end of the
        domain so that ``(nx + pad_x) % n_devices == 0``.

    Returns
    -------
    inv_dx_global : (nx_padded,) np.ndarray
        Replicated — every device sees the whole thing when used with
        ``P("x")`` (see caller packing).
    inv_dx_h_global : (nx_padded,) np.ndarray
    dx_padded : (nx_padded,) np.ndarray
        The padded cell-size profile (float32) — useful for diagnostics
        and the unit test.
    """
    dx_arr = np.asarray(grid.dx_arr, dtype=np.float64)
    if pad_x > 0:
        # pad at the high-x end with boundary-cell-size value
        dx_arr = np.concatenate(
            [dx_arr, np.full(pad_x, float(dx_arr[-1]))]
        )
    nx = dx_arr.shape[0]
    if nx % n_devices != 0:
        raise ValueError(
            f"After padding nx={nx} is not divisible by n_devices={n_devices}"
        )

    inv_dx = 1.0 / dx_arr
    # inv_dx_h[i] = 2 / (dx[i] + dx[i+1]) for i<N-1 ; 0 at end.
    inv_dx_h_mean = 2.0 / (dx_arr[:-1] + dx_arr[1:])
    inv_dx_h = np.concatenate([inv_dx_h_mean, np.zeros(1, dtype=np.float64)])

    return (
        inv_dx.astype(np.float32),
        inv_dx_h.astype(np.float32),
        dx_arr.astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Local NU update kernels (operate on per-device slab including ghosts)
# ---------------------------------------------------------------------------

def _update_h_local_nu(state, materials, dt,
                      inv_dx_slab, inv_dy_full, inv_dz_full,
                      inv_dx_h_slab, inv_dy_h_full, inv_dz_h_full):
    """H update on a local slab using NU inverse spacings.

    Mirrors ``rfx/core/yee.py::update_h_nu`` but accepts pre-sliced
    per-device ``inv_dx`` / ``inv_dx_h`` (length nx_local), while
    y/z spacings are replicated (full-axis).
    """
    ex, ey, ez = state.ex, state.ey, state.ez
    mu = materials.mu_r * MU_0

    curl_x = (
        (_shift_fwd(ez, 1) - ez) * inv_dy_h_full[None, :, None]
        - (_shift_fwd(ey, 2) - ey) * inv_dz_h_full[None, None, :]
    )
    curl_y = (
        (_shift_fwd(ex, 2) - ex) * inv_dz_h_full[None, None, :]
        - (_shift_fwd(ez, 0) - ez) * inv_dx_h_slab[:, None, None]
    )
    curl_z = (
        (_shift_fwd(ey, 0) - ey) * inv_dx_h_slab[:, None, None]
        - (_shift_fwd(ex, 1) - ex) * inv_dy_h_full[None, :, None]
    )

    hx = state.hx - (dt / mu) * curl_x
    hy = state.hy - (dt / mu) * curl_y
    hz = state.hz - (dt / mu) * curl_z

    return state._replace(hx=hx, hy=hy, hz=hz)


def _update_e_local_nu(state, materials, dt,
                      inv_dx_slab, inv_dy_full, inv_dz_full):
    """E update on a local slab using NU inverse (cell-local) spacings.

    Mirrors ``rfx/core/yee.py::update_e_nu``.
    """
    hx, hy, hz = state.hx, state.hy, state.hz
    eps = materials.eps_r * EPS_0
    sigma = materials.sigma

    sigma_dt_2eps = sigma * dt / (2.0 * eps)
    ca = (1.0 - sigma_dt_2eps) / (1.0 + sigma_dt_2eps)
    cb = (dt / eps) / (1.0 + sigma_dt_2eps)

    curl_x = (
        (hz - _shift_bwd(hz, 1)) * inv_dy_full[None, :, None]
        - (hy - _shift_bwd(hy, 2)) * inv_dz_full[None, None, :]
    )
    curl_y = (
        (hx - _shift_bwd(hx, 2)) * inv_dz_full[None, None, :]
        - (hz - _shift_bwd(hz, 0)) * inv_dx_slab[:, None, None]
    )
    curl_z = (
        (hy - _shift_bwd(hy, 0)) * inv_dx_slab[:, None, None]
        - (hx - _shift_bwd(hx, 1)) * inv_dy_full[None, :, None]
    )

    ex = ca * state.ex + cb * curl_x
    ey = ca * state.ey + cb * curl_y
    ez = ca * state.ez + cb * curl_z

    return state._replace(ex=ex, ey=ey, ez=ez, step=state.step + 1)


# ---------------------------------------------------------------------------
# Sharded NU grid metadata — Phase 2A
# ---------------------------------------------------------------------------

from typing import NamedTuple as _NamedTuple


class ShardedNUGrid(_NamedTuple):
    """Metadata describing a non-uniform grid that has been sliced into
    x-axis slabs for the shard_map distributed runner.

    **Coordinate mapping convention** (used by Phase 3 probe/source routing):

    Physical positions are always resolved on the *full-domain*
    ``NonUniformGrid`` first via ``position_to_index(grid, pos)`` which
    returns a global triple ``(i_global, j, k)``.  The x-index is then
    mapped to a rank and a local index:

        rank      = i_global // nx_per_rank
        local_i   = (i_global % nx_per_rank) + ghost_width

    where ``nx_per_rank = nx_local_real`` (the per-rank real cell count,
    *not* the padded/ghost count) and ``ghost_width`` is the ghost cell
    offset stored in this object.  No per-rank physical coordinate system
    is introduced — the global cumulative x-positions are the only
    reference frame.

    Fields
    ------
    nx : int
        Original (unpadded) global x cell count.
    ny : int
        Global y cell count (unchanged by sharding).
    nz : int
        Global z cell count (unchanged by sharding).
    n_devices : int
        Number of ranks / devices.
    nx_padded : int
        Global x count after PEC padding so ``nx_padded % n_devices == 0``.
    pad_x : int
        Number of PEC cells appended at the high-x end
        (``nx_padded - nx``).
    nx_per_rank : int
        Real cells per rank (``nx_padded // n_devices``).
    nx_local : int
        Per-rank cell count including ghost cells
        (``nx_per_rank + 2 * ghost_width``).
    ghost_width : int
        Number of ghost cells on each side of a rank's slab (always 1).
    cpml_layers : int
        CPML layer count from the source grid (replicated, same on every rank).
    dt : float
        Global shared timestep (same on every rank; not recomputed).
    inv_dx_global : np.ndarray  shape (nx_padded,)
        Cell-local inverse x-spacings for the padded full domain.
    inv_dx_h_global : np.ndarray  shape (nx_padded,)
        Mean-spacing inverse x-spacings (seam-aware) for the padded full domain.
    dx_padded : np.ndarray  shape (nx_padded,)
        Padded cell-size profile (float32); useful for diagnostics.
    inv_dy : np.ndarray  shape (ny,)
        Replicated y inverse spacings (every rank receives the full array).
    inv_dy_h : np.ndarray  shape (ny,)
        Replicated y mean-spacing inverse spacings.
    inv_dz : np.ndarray  shape (nz,)
        Replicated z inverse spacings.
    inv_dz_h : np.ndarray  shape (nz,)
        Replicated z mean-spacing inverse spacings.
    rank_has_high_x_pad : int
        Index of the rank that owns the high-x PEC padding cells
        (always ``n_devices - 1``; stored for Phase 3 trim logic).
    nx_trim : int
        Number of padded cells that must be trimmed from the high-x rank's
        slab when assembling the full-domain result (equals ``pad_x``).
    x_starts : tuple[int, ...]
        Global x start index (inclusive) of the real cells for each rank.
    x_stops : tuple[int, ...]
        Global x stop index (exclusive) of the real cells for each rank.
    """

    nx: int
    ny: int
    nz: int
    n_devices: int
    nx_padded: int
    pad_x: int
    nx_per_rank: int
    nx_local: int
    ghost_width: int
    cpml_layers: int
    dt: float
    inv_dx_global: object   # np.ndarray (nx_padded,) float32
    inv_dx_h_global: object  # np.ndarray (nx_padded,) float32
    dx_padded: object        # np.ndarray (nx_padded,) float32
    inv_dy: object           # np.ndarray (ny,) float32
    inv_dy_h: object         # np.ndarray (ny,) float32
    inv_dz: object           # np.ndarray (nz,) float32
    inv_dz_h: object         # np.ndarray (nz,) float32
    rank_has_high_x_pad: int
    nx_trim: int
    x_starts: tuple
    x_stops: tuple


def split_1d_with_ghost(arr: "np.ndarray", n_devices: int, nx_per: int,
                        nx_local: int, ghost: int,
                        pad_value: float) -> "np.ndarray":
    """Split a 1-D inverse-spacing array into per-device slabs with ghost cells.

    This is the canonical split helper shared between the NU metadata builder
    and the distributed_v2 runner.  It produces a ``(n_devices, nx_local)``
    NumPy array where each row is one rank's slab including ``ghost`` cells on
    each side.

    Parameters
    ----------
    arr : np.ndarray  shape (n_devices * nx_per,)
        Padded global inverse-spacing array (output of
        ``_build_sharded_inv_dx_arrays``).
    n_devices : int
    nx_per : int
        Real cells per device (``arr.shape[0] // n_devices``).
    nx_local : int
        ``nx_per + 2 * ghost``.
    ghost : int
        Ghost width (typically 1).
    pad_value : float
        Value to fill boundary ghost cells (1.0 for inv_dx, 0.0 for inv_dx_h).

    Returns
    -------
    slabs : np.ndarray  shape (n_devices, nx_local)
    """
    slabs = np.zeros((n_devices, nx_local), dtype=arr.dtype)
    for d in range(n_devices):
        lo = d * nx_per
        hi = lo + nx_per
        slabs[d, ghost:ghost + nx_per] = arr[lo:hi]
        # left ghost
        if d > 0:
            slabs[d, 0] = arr[lo - 1]
        else:
            slabs[d, 0] = pad_value
        # right ghost
        if d < n_devices - 1:
            slabs[d, -1] = arr[hi]
        else:
            slabs[d, -1] = pad_value
    return slabs


def build_sharded_nu_grid(
    grid,
    n_devices: int,
    exchange_interval: int = 1,
) -> ShardedNUGrid:
    """Build a :class:`ShardedNUGrid` from a full-domain :class:`NonUniformGrid`.

    This is the Phase 2A metadata-only helper.  It does **not** touch
    JAX device placement or shard_map; callers (e.g. the Phase 2B scan
    body) are responsible for calling ``jax.device_put`` on the returned
    arrays.

    Parameters
    ----------
    grid : NonUniformGrid
        Full-domain non-uniform grid produced by ``make_nonuniform_grid``.
    n_devices : int
        Number of ranks / devices for the x-slab decomposition.
    exchange_interval : int
        Ghost exchange interval.  Currently only ``exchange_interval == 1``
        is supported (one exchange per FDTD step).  The parameter is
        accepted for forward-compatibility with Phase 2E batched exchange.
        Ghost width is always ``1 * exchange_interval`` cells, so passing
        a larger value will increase ``ghost_width`` accordingly if support
        is added in a later phase.

    Returns
    -------
    ShardedNUGrid
        Immutable metadata object.  All numpy arrays are float32 and live
        on the host (CPU) at this stage.

    Notes
    -----
    **Coordinate mapping convention** (important for Phase 3):

    Probe and source physical positions must be converted to
    ``(i_global, j, k)`` using ``position_to_index(grid, pos)`` on the
    *full-domain* grid **before** sharding.  The resulting global ``i``
    is then mapped to a (rank, local_i) pair as::

        rank    = i_global // sharded.nx_per_rank
        local_i = (i_global % sharded.nx_per_rank) + sharded.ghost_width

    No per-rank physical coordinate system should be created.
    """
    if exchange_interval != 1:
        raise NotImplementedError(
            "exchange_interval > 1 is reserved for Phase 2E; "
            "only exchange_interval=1 is supported in Phase 2A."
        )

    ghost = exchange_interval  # ghost_width = exchange_interval cells

    nx, ny, nz = grid.nx, grid.ny, grid.nz

    # Pad nx to nearest multiple of n_devices (PEC cells on high-x end)
    pad_x = 0
    if nx % n_devices != 0:
        pad_x = n_devices - (nx % n_devices)
    nx_padded = nx + pad_x

    nx_per = nx_padded // n_devices
    nx_local = nx_per + 2 * ghost

    # Build padded inverse-spacing arrays (reuses existing Phase B helper)
    inv_dx_global, inv_dx_h_global, dx_padded = _build_sharded_inv_dx_arrays(
        grid, n_devices, pad_x=pad_x
    )

    # Replicate y/z inverse spacings (unchanged by x-sharding)
    inv_dy = np.asarray(grid.inv_dy, dtype=np.float32)
    inv_dy_h = np.asarray(grid.inv_dy_h, dtype=np.float32)
    inv_dz = np.asarray(grid.inv_dz, dtype=np.float32)
    inv_dz_h = np.asarray(grid.inv_dz_h, dtype=np.float32)

    # Rank x-range bookkeeping
    x_starts = tuple(d * nx_per for d in range(n_devices))
    x_stops = tuple(min((d + 1) * nx_per, nx) for d in range(n_devices))

    return ShardedNUGrid(
        nx=nx,
        ny=ny,
        nz=nz,
        n_devices=n_devices,
        nx_padded=nx_padded,
        pad_x=pad_x,
        nx_per_rank=nx_per,
        nx_local=nx_local,
        ghost_width=ghost,
        cpml_layers=grid.cpml_layers,
        dt=float(grid.dt),
        inv_dx_global=inv_dx_global,
        inv_dx_h_global=inv_dx_h_global,
        dx_padded=dx_padded,
        inv_dy=inv_dy,
        inv_dy_h=inv_dy_h,
        inv_dz=inv_dz,
        inv_dz_h=inv_dz_h,
        rank_has_high_x_pad=n_devices - 1,
        nx_trim=pad_x,
        x_starts=x_starts,
        x_stops=x_stops,
    )
