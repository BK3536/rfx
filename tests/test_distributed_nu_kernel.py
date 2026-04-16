"""Unit tests for Phase B distributed_nu kernels.

These are pure-python tests that do NOT require multiple devices — they
exercise the slab-building helper and the local H update directly.
"""

import os
# Force 2 virtual devices for the sharded tests in sibling file; also
# harmless here because we don't actually create a mesh in this file.
os.environ.setdefault(
    "XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import numpy as np
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.gpu

from rfx.nonuniform import make_nonuniform_grid
from rfx.core.yee import FDTDState, MaterialArrays, MU_0, update_h_nu
from rfx.runners.distributed_nu import (
    _build_sharded_inv_dx_arrays,
    _update_h_local_nu,
)


def _graded_profile(n_physical, dx0, ratio=1.5):
    """Return a 1-D profile of length n_physical with geometric grading
    of `ratio` on one side, clamping both boundary cells to dx0 so the
    make_nonuniform_grid CPML padding is valid."""
    # Build non-uniform interior, then force both ends back to dx0 so
    # the CPML boundary-value invariant holds.
    prof = dx0 * ratio ** np.linspace(0, 1, n_physical)
    prof[0] = dx0
    prof[-1] = dx0
    return prof


def test_inv_dx_h_slab_boundary_matches_global():
    """The last entry of each device's inv_dx_h slab must equal the
    global inv_dx_h value straddling the slab seam."""
    nz = 8
    ny = 8
    n_physical = 32
    dx0 = 1e-3
    dx_profile = _graded_profile(n_physical, dx0, ratio=1.5)
    dz_profile = np.full(nz, dx0)
    grid = make_nonuniform_grid(
        (n_physical * dx0, ny * dx0), dz_profile, dx0,
        cpml_layers=0,
        dx_profile=dx_profile,
    )

    n_devices = 2
    inv_dx_g, inv_dx_h_g, dx_padded = _build_sharded_inv_dx_arrays(
        grid, n_devices, pad_x=0
    )
    nx = inv_dx_g.shape[0]
    nx_per = nx // n_devices

    # Replay the slab split from distributed_v2 (ghost=1)
    ghost = 1
    nx_local = nx_per + 2 * ghost
    slabs = np.zeros((n_devices, nx_local), dtype=np.float32)
    for d in range(n_devices):
        lo, hi = d * nx_per, (d + 1) * nx_per
        slabs[d, ghost:ghost + nx_per] = inv_dx_h_g[lo:hi]
        if d > 0:
            slabs[d, 0] = inv_dx_h_g[lo - 1]
        if d < n_devices - 1:
            slabs[d, -1] = inv_dx_h_g[hi]

    # For device 0: the rightmost real cell (index ghost+nx_per-1 in slab)
    # must equal inv_dx_h_g[nx_per - 1] — the global mean-spacing
    # straddling the slab seam.
    expected = inv_dx_h_g[nx_per - 1]
    got = slabs[0, ghost + nx_per - 1]
    assert np.isclose(got, expected), (
        f"Device 0 seam-cell inv_dx_h = {got}, expected global "
        f"inv_dx_h[{nx_per - 1}] = {expected}"
    )
    # Verify that this value is specifically 2 / (dx[nx_per-1] + dx[nx_per])
    expected_analytic = 2.0 / (dx_padded[nx_per - 1] + dx_padded[nx_per])
    assert np.isclose(got, expected_analytic, atol=1e-6), (
        f"Seam inv_dx_h = {got}, analytic 2/(dx+dx') = {expected_analytic}"
    )


def test_update_h_nu_local_matches_global_interior():
    """_update_h_local_nu on device-0 slab should match the global
    update_h_nu on the unsharded tensor at interior real cells."""
    nz = 8
    ny = 8
    n_physical = 16
    dx0 = 1e-3
    dx_profile = _graded_profile(n_physical, dx0, ratio=1.2)
    dz_profile = np.full(nz, dx0)
    grid = make_nonuniform_grid(
        (n_physical * dx0, ny * dx0), dz_profile, dx0,
        cpml_layers=0,
        dx_profile=dx_profile,
    )
    nx = grid.nx
    n_devices = 2
    nx_per = nx // n_devices
    ghost = 1
    nx_local = nx_per + 2 * ghost

    # Random-ish E fields
    rng = np.random.default_rng(42)
    ex = jnp.asarray(rng.standard_normal((nx, ny, nz)), dtype=jnp.float32)
    ey = jnp.asarray(rng.standard_normal((nx, ny, nz)), dtype=jnp.float32)
    ez = jnp.asarray(rng.standard_normal((nx, ny, nz)), dtype=jnp.float32)
    zeros_xyz = jnp.zeros((nx, ny, nz), dtype=jnp.float32)
    state = FDTDState(
        ex=ex, ey=ey, ez=ez,
        hx=zeros_xyz, hy=zeros_xyz, hz=zeros_xyz,
        step=jnp.int32(0),
    )
    mats = MaterialArrays(
        eps_r=jnp.ones((nx, ny, nz), dtype=jnp.float32),
        sigma=jnp.zeros((nx, ny, nz), dtype=jnp.float32),
        mu_r=jnp.ones((nx, ny, nz), dtype=jnp.float32),
    )

    # Global H update
    global_h = update_h_nu(
        state, mats, grid.dt,
        grid.inv_dx_h, grid.inv_dy_h, grid.inv_dz_h,
    )

    # Device-0 slab (with ghost). Pick slab build identical to runner.
    inv_dx_g, inv_dx_h_g, _ = _build_sharded_inv_dx_arrays(
        grid, n_devices, pad_x=0)

    def _slab_1d(arr, pad_value):
        slabs = np.zeros((n_devices, nx_local), dtype=arr.dtype)
        for d in range(n_devices):
            lo, hi = d * nx_per, (d + 1) * nx_per
            slabs[d, ghost:ghost + nx_per] = arr[lo:hi]
            if d > 0:
                slabs[d, 0] = arr[lo - 1]
            else:
                slabs[d, 0] = pad_value
            if d < n_devices - 1:
                slabs[d, -1] = arr[hi]
            else:
                slabs[d, -1] = pad_value
        return slabs

    idx_slab = _slab_1d(inv_dx_g, 1.0)
    idxh_slab = _slab_1d(inv_dx_h_g, 0.0)

    def _slab_field(arr):
        out = np.zeros((n_devices, nx_local, ny, nz), dtype=np.float32)
        for d in range(n_devices):
            lo, hi = d * nx_per, (d + 1) * nx_per
            out[d, ghost:ghost + nx_per] = np.asarray(arr)[lo:hi]
            if d > 0:
                out[d, 0] = np.asarray(arr)[lo - 1]
            if d < n_devices - 1:
                out[d, -1] = np.asarray(arr)[hi]
        return out

    ex_sl = _slab_field(ex)
    ey_sl = _slab_field(ey)
    ez_sl = _slab_field(ez)
    z_sl = np.zeros_like(ex_sl)

    d = 0
    slab_state = FDTDState(
        ex=jnp.asarray(ex_sl[d]),
        ey=jnp.asarray(ey_sl[d]),
        ez=jnp.asarray(ez_sl[d]),
        hx=jnp.asarray(z_sl[d]),
        hy=jnp.asarray(z_sl[d]),
        hz=jnp.asarray(z_sl[d]),
        step=jnp.int32(0),
    )
    slab_mats = MaterialArrays(
        eps_r=jnp.ones((nx_local, ny, nz), dtype=jnp.float32),
        sigma=jnp.zeros((nx_local, ny, nz), dtype=jnp.float32),
        mu_r=jnp.ones((nx_local, ny, nz), dtype=jnp.float32),
    )
    slab_h = _update_h_local_nu(
        slab_state, slab_mats, grid.dt,
        jnp.asarray(idx_slab[d]),
        grid.inv_dy, grid.inv_dz,
        jnp.asarray(idxh_slab[d]),
        grid.inv_dy_h, grid.inv_dz_h,
    )

    # Compare interior real cells (exclude ghost + the seam cell, which
    # uses the global mean-spacing; this cell's forward-diff reaches
    # into the ghost and should match the global reference via our
    # inv_dx_h_h pad).
    # Interior real cells in device 0: slab indices [ghost, ghost+nx_per-1).
    # Last real cell (ghost+nx_per-1) uses inv_dx_h straddling the seam,
    # matching global inv_dx_h[nx_per-1], so it should match too.
    glob_slice = np.asarray(global_h.hz)[:nx_per]
    slab_slice = np.asarray(slab_h.hz)[ghost:ghost + nx_per]
    # Exclude last real cell where forward-diff pulls in the ghost ex/ey
    # which are populated from arr[nx_per] (the global interior cell),
    # so this must still match.
    np.testing.assert_allclose(
        slab_slice, glob_slice, atol=1e-5,
        err_msg="device-0 H-z slab should match global interior H-z",
    )


# ---------------------------------------------------------------------------
# Phase 2A: build_sharded_nu_grid metadata tests
# ---------------------------------------------------------------------------

from rfx.runners.distributed_nu import build_sharded_nu_grid, ShardedNUGrid


def _make_test_grid(nx_physical=16, ny_physical=8, nz_physical=8,
                    dx0=1e-3, ratio=1.3, cpml_layers=0):
    """Build a small graded NonUniformGrid for metadata tests."""
    dx_profile = _graded_profile(nx_physical, dx0, ratio=ratio)
    dz_profile = np.full(nz_physical, dx0)
    return make_nonuniform_grid(
        (nx_physical * dx0, ny_physical * dx0), dz_profile, dx0,
        cpml_layers=cpml_layers,
        dx_profile=dx_profile,
    )


def test_build_sharded_nu_grid_metadata_shapes():
    """Local x sizes sum to global nx; y/z unchanged; cpml_layers replicated."""
    grid = _make_test_grid(nx_physical=16, ny_physical=8, nz_physical=6, cpml_layers=0)
    n_devices = 2
    sg = build_sharded_nu_grid(grid, n_devices=n_devices, exchange_interval=1)

    assert isinstance(sg, ShardedNUGrid)

    # x sizes sum correctly
    assert sg.nx == grid.nx
    assert sg.nx_padded % n_devices == 0
    assert sg.nx_per_rank * n_devices == sg.nx_padded
    assert sg.nx_padded >= sg.nx
    assert sg.pad_x == sg.nx_padded - sg.nx

    # nx_local includes ghosts
    assert sg.nx_local == sg.nx_per_rank + 2 * sg.ghost_width
    assert sg.ghost_width == 1

    # y/z unchanged
    assert sg.ny == grid.ny
    assert sg.nz == grid.nz

    # cpml_layers replicated
    assert sg.cpml_layers == grid.cpml_layers

    # inv spacing array shapes
    assert sg.inv_dx_global.shape == (sg.nx_padded,)
    assert sg.inv_dx_h_global.shape == (sg.nx_padded,)
    assert sg.dx_padded.shape == (sg.nx_padded,)
    assert sg.inv_dy.shape == (grid.ny,)
    assert sg.inv_dy_h.shape == (grid.ny,)
    assert sg.inv_dz.shape == (grid.nz,)
    assert sg.inv_dz_h.shape == (grid.nz,)

    # x_starts / x_stops bookkeeping
    assert len(sg.x_starts) == n_devices
    assert len(sg.x_stops) == n_devices
    assert sg.x_starts[0] == 0
    assert sg.x_stops[-1] == sg.nx  # capped at unpadded nx


def test_build_sharded_nu_grid_inv_dx_seam_continuity():
    """inv_dx_h at the slab seam matches the un-sharded global reference."""
    from rfx.runners.distributed_nu import split_1d_with_ghost

    grid = _make_test_grid(nx_physical=32, ny_physical=8, nz_physical=8,
                           dx0=1e-3, ratio=1.5)
    n_devices = 2
    sg = build_sharded_nu_grid(grid, n_devices=n_devices)

    nx_per = sg.nx_per_rank
    ghost = sg.ghost_width
    nx_local = sg.nx_local

    # Build the slabs using the canonical helper
    slabs = split_1d_with_ghost(
        sg.inv_dx_h_global, n_devices, nx_per, nx_local, ghost, pad_value=0.0
    )

    # For device 0: last real cell in slab = global index nx_per - 1
    seam_slab = float(slabs[0, ghost + nx_per - 1])
    seam_global = float(sg.inv_dx_h_global[nx_per - 1])
    assert np.isclose(seam_slab, seam_global, atol=1e-6), (
        f"Seam slab value {seam_slab} != global reference {seam_global}"
    )

    # Cross-check analytically: 2 / (dx[seam-1] + dx[seam])
    dx_arr = sg.dx_padded
    analytic = float(2.0 / (dx_arr[nx_per - 1] + dx_arr[nx_per]))
    assert np.isclose(seam_slab, analytic, atol=1e-5), (
        f"Seam inv_dx_h {seam_slab} != analytic {analytic}"
    )


def test_build_sharded_nu_grid_pad_trim_for_nondivisible_nx():
    """nx=17, n_devices=2 — high-x rank gets the pad; metadata flags are correct."""
    # nx=17 is odd; with n_devices=2 we need pad_x=1 to reach 18
    dx0 = 1e-3
    nx_physical = 17
    ny_physical = 8
    nz_physical = 8
    dx_profile = _graded_profile(nx_physical, dx0, ratio=1.2)
    dz_profile = np.full(nz_physical, dx0)
    grid = make_nonuniform_grid(
        (nx_physical * dx0, ny_physical * dx0), dz_profile, dx0,
        cpml_layers=0,
        dx_profile=dx_profile,
    )
    assert grid.nx == nx_physical  # sanity

    n_devices = 2
    sg = build_sharded_nu_grid(grid, n_devices=n_devices)

    # Padding arithmetic
    assert sg.pad_x == 1, f"Expected pad_x=1, got {sg.pad_x}"
    assert sg.nx_padded == 18
    assert sg.nx_per_rank == 9
    assert sg.nx_trim == 1

    # High-x rank index
    assert sg.rank_has_high_x_pad == n_devices - 1  # rank 1

    # The padded cell in inv_dx_global should equal 1/dx_arr[-1]
    expected_last_inv = float(1.0 / np.asarray(grid.dx_arr)[-1])
    got_last_inv = float(sg.inv_dx_global[-1])
    assert np.isclose(got_last_inv, expected_last_inv, rtol=1e-5), (
        f"Padded inv_dx last cell {got_last_inv} != 1/dx[-1] {expected_last_inv}"
    )


def test_build_sharded_nu_grid_replicates_dt():
    """dt is identical (same Python float) across all conceptual ranks."""
    grid = _make_test_grid(nx_physical=16, cpml_layers=0)
    n_devices = 4
    sg = build_sharded_nu_grid(grid, n_devices=n_devices)

    # dt must equal the grid's dt exactly (no recomputation)
    assert sg.dt == float(grid.dt), (
        f"ShardedNUGrid dt {sg.dt} != grid.dt {float(grid.dt)}"
    )
    # dt is a plain Python float (not a JAX array) so it's trivially
    # identical across all ranks — assert it is not a JAX array
    assert isinstance(sg.dt, float), (
        f"Expected plain float for dt, got {type(sg.dt)}"
    )


def test_build_sharded_nu_grid_position_to_index_deterministic():
    """Known physical coord maps to expected (rank, local_i) deterministically."""
    from rfx.nonuniform import position_to_index

    dx0 = 1e-3
    nx_physical = 16
    ny_physical = 8
    nz_physical = 8
    dx_profile = np.full(nx_physical, dx0)   # uniform so we can predict the index
    dz_profile = np.full(nz_physical, dx0)
    grid = make_nonuniform_grid(
        (nx_physical * dx0, ny_physical * dx0), dz_profile, dx0,
        cpml_layers=0,
        dx_profile=dx_profile,
    )

    n_devices = 2
    sg = build_sharded_nu_grid(grid, n_devices=n_devices)

    # Place a physical position in the middle of the domain
    # For a uniform profile with cpml=0, index = round(pos / dx0)
    # Use a position in the second half so it lands on rank 1.
    # nx_physical=16, nx_per_rank=8; cell 10 → rank 1, local 10-8=2 (+ghost=1 → local_i=3)
    target_global_i = 10
    pos_x = (float(np.asarray(grid.dx_arr[:target_global_i]).sum())
              + 0.5 * float(np.asarray(grid.dx_arr)[target_global_i]))
    pos_y = 0.5 * dx0
    pos_z = 0.5 * dx0
    i_global, j_global, k_global = position_to_index(grid, (pos_x, pos_y, pos_z))
    assert i_global == target_global_i, (
        f"position_to_index returned i={i_global}, expected {target_global_i}"
    )

    # Apply Phase 2A mapping convention
    expected_rank = target_global_i // sg.nx_per_rank          # 10 // 8 = 1
    expected_local_i = (target_global_i % sg.nx_per_rank) + sg.ghost_width  # 2 + 1 = 3

    got_rank = i_global // sg.nx_per_rank
    got_local_i = (i_global % sg.nx_per_rank) + sg.ghost_width

    assert got_rank == expected_rank, (
        f"rank={got_rank}, expected {expected_rank}"
    )
    assert got_local_i == expected_local_i, (
        f"local_i={got_local_i}, expected {expected_local_i}"
    )

    # Calling the mapping a second time must give the same result (deterministic)
    i2, _, _ = position_to_index(grid, (pos_x, pos_y, pos_z))
    assert i2 // sg.nx_per_rank == got_rank
    assert (i2 % sg.nx_per_rank) + sg.ghost_width == got_local_i
