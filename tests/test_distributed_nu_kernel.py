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


# ---------------------------------------------------------------------------
# Phase 2B: hard-PEC + ghost-exchange sharded NU scan body tests
# ---------------------------------------------------------------------------

import jax  # noqa: E402

from rfx.runners.distributed_nu import (  # noqa: E402
    run_nonuniform_distributed_pec,
    shard_pec_mask_x_slab,
)
from rfx.nonuniform import (  # noqa: E402
    run_nonuniform,
    position_to_index as _phase2b_pos_to_idx,
    make_current_source as _phase2b_make_current_source,
)
from rfx.simulation import SourceSpec, ProbeSpec  # noqa: E402
from tests._distributed_nu_tolerances import (  # noqa: E402
    assert_class_b_parity,
)


_PHASE2B_REQUIRES_2DEV = pytest.mark.skipif(
    jax.device_count() < 2,
    reason=(
        "Phase 2B distributed-NU PEC tests need >=2 JAX devices "
        "(set XLA_FLAGS=--xla_force_host_platform_device_count=2)."
    ),
)


def _phase2b_build_test_grid(nx_physical=16, ny=8, nz=8, dx0=1e-3, ratio=1.2):
    """Small NU grid (cpml=0, hard-PEC cavity) for Phase 2B parity tests."""
    dx_profile = _graded_profile(nx_physical, dx0, ratio=ratio)
    dz_profile = np.full(nz, dx0)
    grid = make_nonuniform_grid(
        (nx_physical * dx0, ny * dx0), dz_profile, dx0,
        cpml_layers=0,
        dx_profile=dx_profile,
    )
    return grid


def _phase2b_make_materials(grid):
    """Vacuum (eps=mu=1, sigma=0) materials shaped like ``grid``."""
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    return MaterialArrays(
        eps_r=jnp.ones((nx, ny, nz), dtype=jnp.float32),
        sigma=jnp.zeros((nx, ny, nz), dtype=jnp.float32),
        mu_r=jnp.ones((nx, ny, nz), dtype=jnp.float32),
    )


def _phase2b_gauss_waveform(t, t0=8e-12, tau=2.5e-12):
    return jnp.exp(-((t - t0) ** 2) / (2.0 * tau ** 2))


def _phase2b_shard_mat(materials, sharded_grid):
    """Shard a full-domain MaterialArrays for Phase 2B's runner."""
    n_devices = sharded_grid.n_devices
    nx_local = sharded_grid.nx_local
    ghost = sharded_grid.ghost_width
    pad_x = sharded_grid.pad_x

    devices = jax.devices()[:n_devices]
    from jax.sharding import Mesh, NamedSharding
    mesh = Mesh(np.array(devices), axis_names=("x",))
    shd = NamedSharding(mesh, P("x"))

    if pad_x > 0:
        pad = ((0, pad_x), (0, 0), (0, 0))
        materials = MaterialArrays(
            eps_r=jnp.pad(materials.eps_r, pad, constant_values=1.0),
            sigma=jnp.pad(materials.sigma, pad, constant_values=0.0),
            mu_r=jnp.pad(materials.mu_r, pad, constant_values=1.0),
        )

    from rfx.runners.distributed import _split_materials
    mat_slabs = _split_materials(materials, n_devices, ghost)

    def _shard_stacked(arr):
        n_dev = arr.shape[0]
        rest = arr.shape[1:]
        return jax.device_put(arr.reshape(n_dev * rest[0], *rest[1:]), shd)

    return MaterialArrays(
        eps_r=_shard_stacked(mat_slabs.eps_r),
        sigma=_shard_stacked(mat_slabs.sigma),
        mu_r=_shard_stacked(mat_slabs.mu_r),
    )


from jax.sharding import PartitionSpec as P  # noqa: E402


@_PHASE2B_REQUIRES_2DEV
def test_distributed_pec_only_2device_matches_single_device():
    """Class B forward parity: 2-device distributed run should match the
    single-device ``run_nonuniform`` reference at the final step on a
    small NU PEC cavity.

    Source on rank 0 interior, probe on rank 1 interior, so the signal
    crosses the seam.  Hard PEC at all 6 domain faces (no CPML).
    """
    devices = jax.devices()[:2]
    n_devices = 2
    n_steps = 60

    grid = _phase2b_build_test_grid(nx_physical=16, ny=8, nz=8, ratio=1.2)
    materials = _phase2b_make_materials(grid)

    # Source: global x-index 4 (rank 0), probe: global x-index 12 (rank 1)
    src_idx = (4, 4, 4)
    prb_idx = (12, 4, 4)
    src_si, src_sj, src_sk, src_comp, src_wf = _phase2b_make_current_source(
        grid, src_idx, "ez", _phase2b_gauss_waveform, n_steps, materials,
    )
    src_spec = SourceSpec(
        i=int(src_si), j=int(src_sj), k=int(src_sk),
        component=src_comp, waveform=jnp.asarray(src_wf),
    )
    prb_spec = ProbeSpec(
        i=int(prb_idx[0]), j=int(prb_idx[1]), k=int(prb_idx[2]),
        component="ez",
    )

    # Single-device reference
    single_out = run_nonuniform(
        grid=grid,
        materials=materials,
        n_steps=n_steps,
        sources=[src_spec],
        probes=[prb_spec],
    )
    ts_single = jnp.asarray(single_out["time_series"])[:, 0]

    # 2-device distributed run
    sharded_grid = build_sharded_nu_grid(grid, n_devices=n_devices,
                                         exchange_interval=1)
    sharded_mat = _phase2b_shard_mat(materials, sharded_grid)
    out = run_nonuniform_distributed_pec(
        sharded_grid=sharded_grid,
        sharded_materials=sharded_mat,
        sharded_pec_mask=None,
        n_steps=n_steps,
        sources=[src_spec],
        probes=[prb_spec],
        n_devices=n_devices,
        devices=devices,
    )
    ts_dist = jnp.asarray(out["time_series"])[:, 0]

    assert ts_dist.shape == ts_single.shape, (
        f"shape mismatch: dist={ts_dist.shape}, single={ts_single.shape}"
    )

    assert_class_b_parity(ts_single, ts_dist,
                          label="phase2b_pec_only_2device_parity")


@_PHASE2B_REQUIRES_2DEV
def test_distributed_pec_only_seam_no_double_zeroing():
    """Class D seam isolation: a PEC mask cell exactly at the slab seam
    is zeroed exactly once.

    Place a single PEC cell at global x-index ``nx_per_rank`` (rank 1's
    first real cell).  The Phase 2B mask runner must mask this cell
    once (rank 1's sharded mask has it True at slab-local index
    ``ghost``); rank 0's slab must NOT see it as a real cell (only as
    its right ghost) so apply_pec_mask on rank 0 must not act on it.

    Verification: the distributed run must match the single-device
    reference (which applies the mask exactly once).
    """
    devices = jax.devices()[:2]
    n_devices = 2
    n_steps = 30

    grid = _phase2b_build_test_grid(nx_physical=16, ny=8, nz=8, ratio=1.0)
    materials = _phase2b_make_materials(grid)
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    # Build a PEC mask with a single cell exactly at the slab seam.
    sharded_grid = build_sharded_nu_grid(grid, n_devices=n_devices)
    seam_i = sharded_grid.nx_per_rank  # global x-index of rank 1's first real cell
    seam_j = ny // 2
    seam_k = nz // 2
    pec_mask = jnp.zeros((nx, ny, nz), dtype=jnp.bool_)
    # Mark a 1x1x1 PEC cell + its left neighbour so the tangential mask
    # in apply_pec_mask sees a PEC neighbour (otherwise the thin-sheet
    # rule preserves the field — no double-zeroing risk to detect).
    pec_mask = pec_mask.at[seam_i, seam_j, seam_k].set(True)
    pec_mask = pec_mask.at[seam_i - 1, seam_j, seam_k].set(True)
    pec_mask = pec_mask.at[seam_i + 1, seam_j, seam_k].set(True)

    src_idx = (4, 4, 4)
    src_si, src_sj, src_sk, src_comp, src_wf = _phase2b_make_current_source(
        grid, src_idx, "ez", _phase2b_gauss_waveform, n_steps, materials,
    )
    src_spec = SourceSpec(
        i=int(src_si), j=int(src_sj), k=int(src_sk),
        component=src_comp, waveform=jnp.asarray(src_wf),
    )
    # Probe on rank 1, away from the seam, so the seam-mask physics
    # propagates into the sampled signal.
    prb_spec = ProbeSpec(i=12, j=4, k=4, component="ez")

    # Single-device reference (applies the mask once via apply_pec_mask)
    single_out = run_nonuniform(
        grid=grid,
        materials=materials,
        n_steps=n_steps,
        pec_mask=pec_mask,
        sources=[src_spec],
        probes=[prb_spec],
    )
    ts_single = jnp.asarray(single_out["time_series"])[:, 0]

    # Distributed run with the same PEC mask
    sharded_mat = _phase2b_shard_mat(materials, sharded_grid)
    sharded_mask = shard_pec_mask_x_slab(pec_mask, sharded_grid)
    out = run_nonuniform_distributed_pec(
        sharded_grid=sharded_grid,
        sharded_materials=sharded_mat,
        sharded_pec_mask=sharded_mask,
        n_steps=n_steps,
        sources=[src_spec],
        probes=[prb_spec],
        n_devices=n_devices,
        devices=devices,
    )
    ts_dist = jnp.asarray(out["time_series"])[:, 0]

    # If the seam cell were double-zeroed, the field profile would
    # decay differently — Class B forward parity catches that.
    assert_class_b_parity(ts_single, ts_dist,
                          label="phase2b_seam_no_double_zeroing")


@_PHASE2B_REQUIRES_2DEV
def test_distributed_pec_mask_override_union_semantics():
    """Geometry-defined PEC + override union must match single-device.

    The single-device path takes ``pec_mask = geom_mask | override_mask``
    before passing to the runner.  We replicate that union at the host
    level (mirroring ``Simulation.forward()`` semantics) and verify
    the distributed runner produces the identical probe time-series.
    """
    devices = jax.devices()[:2]
    n_devices = 2
    n_steps = 30

    grid = _phase2b_build_test_grid(nx_physical=16, ny=8, nz=8, ratio=1.0)
    materials = _phase2b_make_materials(grid)
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    # "Geometry" PEC slab: a thin sheet at x=10
    geom_mask = jnp.zeros((nx, ny, nz), dtype=jnp.bool_)
    geom_mask = geom_mask.at[10, 2:6, 2:6].set(True)
    geom_mask = geom_mask.at[11, 2:6, 2:6].set(True)
    # "Override" PEC: another small block at x=5 inside rank 0
    override_mask = jnp.zeros((nx, ny, nz), dtype=jnp.bool_)
    override_mask = override_mask.at[5, 3:5, 3:5].set(True)
    override_mask = override_mask.at[6, 3:5, 3:5].set(True)
    union_mask = geom_mask | override_mask

    src_idx = (2, 4, 4)
    src_si, src_sj, src_sk, src_comp, src_wf = _phase2b_make_current_source(
        grid, src_idx, "ez", _phase2b_gauss_waveform, n_steps, materials,
    )
    src_spec = SourceSpec(
        i=int(src_si), j=int(src_sj), k=int(src_sk),
        component=src_comp, waveform=jnp.asarray(src_wf),
    )
    prb_spec = ProbeSpec(i=13, j=4, k=4, component="ez")

    single_out = run_nonuniform(
        grid=grid, materials=materials, n_steps=n_steps,
        pec_mask=union_mask,
        sources=[src_spec], probes=[prb_spec],
    )
    ts_single = jnp.asarray(single_out["time_series"])[:, 0]

    sharded_grid = build_sharded_nu_grid(grid, n_devices=n_devices)
    sharded_mat = _phase2b_shard_mat(materials, sharded_grid)
    sharded_mask = shard_pec_mask_x_slab(union_mask, sharded_grid)
    out = run_nonuniform_distributed_pec(
        sharded_grid=sharded_grid,
        sharded_materials=sharded_mat,
        sharded_pec_mask=sharded_mask,
        n_steps=n_steps,
        sources=[src_spec],
        probes=[prb_spec],
        n_devices=n_devices,
        devices=devices,
    )
    ts_dist = jnp.asarray(out["time_series"])[:, 0]

    assert_class_b_parity(ts_single, ts_dist,
                          label="phase2b_pec_mask_override_union")


@_PHASE2B_REQUIRES_2DEV
def test_distributed_h_ghost_exchange_recovers_global_field():
    """Class B parity test for H-update + ghost exchange with no PEC mask
    and no source.

    Initialise the E field with a localised non-zero pattern that
    straddles the slab seam, then run a short forward (no source).  The
    distributed scan body's H update + ghost exchange should produce
    H-field evolution that matches the single-device run_nonuniform
    reference at the probe.
    """
    devices = jax.devices()[:2]
    n_devices = 2
    n_steps = 20

    grid = _phase2b_build_test_grid(nx_physical=16, ny=8, nz=8, ratio=1.0)
    materials = _phase2b_make_materials(grid)

    # Inject a soft source so we have a non-trivial time series to compare;
    # this still tests the H ghost exchange because the signal must cross
    # the seam to reach the probe.
    n_steps_run = 40
    src_idx = (3, 4, 4)
    src_si, src_sj, src_sk, src_comp, src_wf = _phase2b_make_current_source(
        grid, src_idx, "ez", _phase2b_gauss_waveform, n_steps_run, materials,
    )
    src_spec = SourceSpec(
        i=int(src_si), j=int(src_sj), k=int(src_sk),
        component=src_comp, waveform=jnp.asarray(src_wf),
    )
    # Probe in rank 1 sampling Hy (so the H ghost exchange is the
    # critical step transporting the signal)
    prb_spec = ProbeSpec(i=12, j=4, k=4, component="hy")

    single_out = run_nonuniform(
        grid=grid, materials=materials, n_steps=n_steps_run,
        sources=[src_spec], probes=[prb_spec],
    )
    ts_single = jnp.asarray(single_out["time_series"])[:, 0]

    sharded_grid = build_sharded_nu_grid(grid, n_devices=n_devices)
    sharded_mat = _phase2b_shard_mat(materials, sharded_grid)
    out = run_nonuniform_distributed_pec(
        sharded_grid=sharded_grid,
        sharded_materials=sharded_mat,
        sharded_pec_mask=None,
        n_steps=n_steps_run,
        sources=[src_spec],
        probes=[prb_spec],
        n_devices=n_devices,
        devices=devices,
    )
    ts_dist = jnp.asarray(out["time_series"])[:, 0]

    assert_class_b_parity(ts_single, ts_dist,
                          label="phase2b_h_ghost_exchange_global_field")


# ---------------------------------------------------------------------------
# Phase 2C: full CPML on x-slabs — V3 plan lines 633-657
# ---------------------------------------------------------------------------

from rfx.runners.distributed_nu import (  # noqa: E402
    init_cpml_for_sharded_nu,
    shard_cpml_state_x_slab,
)
from tests._distributed_nu_tolerances import (  # noqa: E402
    assert_class_c_cpml_seam_noop,
    CPML_X_FACE_PSI_FIELDS,
)


def _phase2c_build_cpml_grid(nx_physical=16, ny=8, nz=8, dx0=1e-3,
                             cpml_layers=4):
    """Small NU grid with CPML padding (uniform xy + uniform dz; the
    NU plumbing exercises the per-axis dx_x/dx_y/dz code paths even on a
    degenerate-uniform profile).
    """
    dx_profile = np.full(nx_physical, dx0)
    dz_profile = np.full(nz, dx0)
    grid = make_nonuniform_grid(
        (nx_physical * dx0, ny * dx0), dz_profile, dx0,
        cpml_layers=cpml_layers,
        dx_profile=dx_profile,
    )
    return grid


def _phase2c_unstack_cpml(cpml_state_sharded, sharded_grid, n_cpml,
                          field_name):
    """Unstack a sharded CPML psi field back into per-rank arrays.

    The runner returns CPMLState psi arrays with leading axis
    ``n_devices * n_cpml``; this helper pulls them to host and reshapes
    to ``(n_devices, n_cpml, d1, d2)`` so individual ranks can be
    inspected.
    """
    arr = np.asarray(getattr(cpml_state_sharded, field_name))
    n_devices = sharded_grid.n_devices
    total = arr.shape[0]
    assert total == n_devices * n_cpml, (
        f"unstack: total leading dim {total} != "
        f"n_devices * n_cpml = {n_devices * n_cpml}"
    )
    return arr.reshape(n_devices, n_cpml, *arr.shape[1:])


@_PHASE2B_REQUIRES_2DEV
def test_distributed_cpml_2device_matches_single_device():
    """Phase 2C Class B forward parity: NU + CPML on all faces, 2 devices.

    The 2-device distributed run with CPML must match the single-device
    ``run_nonuniform`` reference at the final step on a small NU CPML
    domain.  Source on rank 0, probe on rank 1, so the signal crosses
    the seam.  Validates that the per-rank CPMLState evolution + the
    rank-conditional x-face gating preserve forward parity.
    """
    devices = jax.devices()[:2]
    n_devices = 2
    n_steps = 60

    grid = _phase2c_build_cpml_grid(
        nx_physical=16, ny=8, nz=8, dx0=1e-3, cpml_layers=4)
    materials = _phase2b_make_materials(grid)

    # Source / probe in interior; cross the slab seam.
    nx, ny_g, nz_g = grid.nx, grid.ny, grid.nz
    src_idx = (nx // 4, ny_g // 2, nz_g // 2)
    prb_idx = (3 * nx // 4, ny_g // 2, nz_g // 2)
    src_si, src_sj, src_sk, src_comp, src_wf = _phase2b_make_current_source(
        grid, src_idx, "ez", _phase2b_gauss_waveform, n_steps, materials,
    )
    src_spec = SourceSpec(
        i=int(src_si), j=int(src_sj), k=int(src_sk),
        component=src_comp, waveform=jnp.asarray(src_wf),
    )
    prb_spec = ProbeSpec(
        i=int(prb_idx[0]), j=int(prb_idx[1]), k=int(prb_idx[2]),
        component="ez",
    )

    # Single-device reference (uses init_cpml under the hood)
    single_out = run_nonuniform(
        grid=grid, materials=materials, n_steps=n_steps,
        sources=[src_spec], probes=[prb_spec],
    )
    ts_single = jnp.asarray(single_out["time_series"])[:, 0]

    # Distributed run with CPML
    sharded_grid = build_sharded_nu_grid(grid, n_devices=n_devices)
    sharded_mat = _phase2b_shard_mat(materials, sharded_grid)
    cpml_params, cpml_state_stacked = init_cpml_for_sharded_nu(
        sharded_grid, n_devices=n_devices,
    )
    from jax.sharding import Mesh
    mesh = Mesh(np.array(devices), axis_names=("x",))
    cpml_state_sharded = shard_cpml_state_x_slab(
        cpml_state_stacked, sharded_grid, mesh,
    )
    out = run_nonuniform_distributed_pec(
        sharded_grid=sharded_grid,
        sharded_materials=sharded_mat,
        sharded_pec_mask=None,
        n_steps=n_steps,
        sources=[src_spec],
        probes=[prb_spec],
        n_devices=n_devices,
        devices=devices,
        cpml_params=cpml_params,
        cpml_state=cpml_state_sharded,
    )
    ts_dist = jnp.asarray(out["time_series"])[:, 0]

    assert ts_dist.shape == ts_single.shape, (
        f"shape mismatch: dist={ts_dist.shape}, single={ts_single.shape}"
    )
    assert_class_b_parity(ts_single, ts_dist,
                          label="phase2c_cpml_2device_parity")


@_PHASE2B_REQUIRES_2DEV
def test_distributed_cpml_internal_seam_is_noop():
    """Phase 2C Class C: interior-rank x-face CPML state is exactly zero.

    Run a 2-device distributed CPML scan for 50 steps with a source that
    drives a non-trivial field through the seam.  After the run, every
    x-face CPML psi array on an interior rank must be exactly zero —
    the seam is NOT a CPML boundary, only the outer x_lo / x_hi faces are.

    With ``n_devices=3`` rank 1 is the interior rank; with ``n_devices=2``
    there is no interior rank, so we pick a 3-device run when at least
    3 virtual devices are available; otherwise we run the 2-device case
    and assert that rank 0 owns x_lo psi but the *upper* slab of rank 0's
    x-hi psi is unmodified (rank 0 is not rank N-1, so its x-hi psi must
    stay zero).
    """
    devices_all = jax.devices()
    n_devices = 3 if len(devices_all) >= 3 else 2
    devices = devices_all[:n_devices]
    n_steps = 50

    grid = _phase2c_build_cpml_grid(
        nx_physical=24 if n_devices == 3 else 16, ny=8, nz=8,
        dx0=1e-3, cpml_layers=4,
    )
    materials = _phase2b_make_materials(grid)
    nx = grid.nx

    # Source in the centre so the field reaches every rank
    src_idx = (nx // 2, grid.ny // 2, grid.nz // 2)
    src_si, src_sj, src_sk, src_comp, src_wf = _phase2b_make_current_source(
        grid, src_idx, "ez", _phase2b_gauss_waveform, n_steps, materials,
    )
    src_spec = SourceSpec(
        i=int(src_si), j=int(src_sj), k=int(src_sk),
        component=src_comp, waveform=jnp.asarray(src_wf),
    )
    prb_spec = ProbeSpec(i=nx // 2, j=grid.ny // 2, k=grid.nz // 2,
                         component="ez")

    sharded_grid = build_sharded_nu_grid(grid, n_devices=n_devices)
    sharded_mat = _phase2b_shard_mat(materials, sharded_grid)
    cpml_params, cpml_state_stacked = init_cpml_for_sharded_nu(
        sharded_grid, n_devices=n_devices,
    )
    from jax.sharding import Mesh
    mesh = Mesh(np.array(devices), axis_names=("x",))
    cpml_state_sharded = shard_cpml_state_x_slab(
        cpml_state_stacked, sharded_grid, mesh,
    )
    out = run_nonuniform_distributed_pec(
        sharded_grid=sharded_grid,
        sharded_materials=sharded_mat,
        sharded_pec_mask=None,
        n_steps=n_steps,
        sources=[src_spec],
        probes=[prb_spec],
        n_devices=n_devices,
        devices=devices,
        cpml_params=cpml_params,
        cpml_state=cpml_state_sharded,
    )
    final_cpml = out["cpml_state_sharded"]
    assert final_cpml is not None, (
        "Phase 2C runner must return cpml_state_sharded when CPML is enabled"
    )

    n_cpml = sharded_grid.cpml_layers

    # Rank-by-rank inspection of every x-face psi field
    if n_devices >= 3:
        # Rank 1 is interior — every x-face psi must be exactly zero
        from rfx.boundaries.cpml import CPMLState
        rank1_psi = {}
        for fname in CPML_X_FACE_PSI_FIELDS:
            stacked = _phase2c_unstack_cpml(
                final_cpml, sharded_grid, n_cpml, fname,
            )
            rank1_psi[fname] = jnp.asarray(stacked[1])  # rank 1
        # Build a fake CPMLState-like NamedTuple subset for the helper
        rank1_state = CPMLState(
            psi_ex_ylo=jnp.zeros(1), psi_ex_yhi=jnp.zeros(1),
            psi_ex_zlo=jnp.zeros(1), psi_ex_zhi=jnp.zeros(1),
            psi_ey_xlo=rank1_psi["psi_ey_xlo"],
            psi_ey_xhi=rank1_psi["psi_ey_xhi"],
            psi_ey_zlo=jnp.zeros(1), psi_ey_zhi=jnp.zeros(1),
            psi_ez_xlo=rank1_psi["psi_ez_xlo"],
            psi_ez_xhi=rank1_psi["psi_ez_xhi"],
            psi_ez_ylo=jnp.zeros(1), psi_ez_yhi=jnp.zeros(1),
            psi_hx_ylo=jnp.zeros(1), psi_hx_yhi=jnp.zeros(1),
            psi_hx_zlo=jnp.zeros(1), psi_hx_zhi=jnp.zeros(1),
            psi_hy_xlo=rank1_psi["psi_hy_xlo"],
            psi_hy_xhi=rank1_psi["psi_hy_xhi"],
            psi_hy_zlo=jnp.zeros(1), psi_hy_zhi=jnp.zeros(1),
            psi_hz_xlo=rank1_psi["psi_hz_xlo"],
            psi_hz_xhi=rank1_psi["psi_hz_xhi"],
            psi_hz_ylo=jnp.zeros(1), psi_hz_yhi=jnp.zeros(1),
        )
        assert_class_c_cpml_seam_noop(rank1_state, label="phase2c_seam_noop_rank1")
    else:
        # 2-device fallback: rank 0 owns x_lo only, so x_hi psi must be 0;
        # rank 1 owns x_hi only, so x_lo psi must be 0.
        for fname in CPML_X_FACE_PSI_FIELDS:
            stacked = _phase2c_unstack_cpml(
                final_cpml, sharded_grid, n_cpml, fname,
            )
            if fname.endswith("xlo"):
                # Rank 1 (not rank 0) must NOT touch x_lo
                assert np.all(stacked[1] == 0), (
                    f"Phase 2C: rank 1 (interior side) modified x_lo psi "
                    f"field '{fname}' — internal seam treated as boundary."
                )
            elif fname.endswith("xhi"):
                # Rank 0 (not rank N-1) must NOT touch x_hi
                assert np.all(stacked[0] == 0), (
                    f"Phase 2C: rank 0 (interior side) modified x_hi psi "
                    f"field '{fname}' — internal seam treated as boundary."
                )


@_PHASE2B_REQUIRES_2DEV
def test_distributed_cpml_outer_face_active():
    """Phase 2C Class C+D: outer x-face CPML on rank 0 / rank N-1 is active.

    Drive a pulse from the centre of the domain.  After enough time
    steps for the wavefront to reach both x-faces, rank 0's x_lo psi
    arrays and rank (N-1)'s x_hi psi arrays must be non-zero — the
    outer face IS a physical CPML boundary.

    This is the complement of ``test_distributed_cpml_internal_seam_is_noop``:
    seam-noop says interior ranks don't activate; outer-face-active
    says boundary ranks DO activate.
    """
    devices = jax.devices()[:2]
    n_devices = 2
    n_steps = 80  # long enough for the pulse to reach both x-faces

    grid = _phase2c_build_cpml_grid(
        nx_physical=16, ny=8, nz=8, dx0=1e-3, cpml_layers=4)
    materials = _phase2b_make_materials(grid)
    nx = grid.nx

    # Source in the centre (cell ~nx/2)
    src_idx = (nx // 2, grid.ny // 2, grid.nz // 2)
    src_si, src_sj, src_sk, src_comp, src_wf = _phase2b_make_current_source(
        grid, src_idx, "ez", _phase2b_gauss_waveform, n_steps, materials,
    )
    src_spec = SourceSpec(
        i=int(src_si), j=int(src_sj), k=int(src_sk),
        component=src_comp, waveform=jnp.asarray(src_wf),
    )
    prb_spec = ProbeSpec(i=nx // 2, j=grid.ny // 2, k=grid.nz // 2,
                         component="ez")

    sharded_grid = build_sharded_nu_grid(grid, n_devices=n_devices)
    sharded_mat = _phase2b_shard_mat(materials, sharded_grid)
    cpml_params, cpml_state_stacked = init_cpml_for_sharded_nu(
        sharded_grid, n_devices=n_devices,
    )
    from jax.sharding import Mesh
    mesh = Mesh(np.array(devices), axis_names=("x",))
    cpml_state_sharded = shard_cpml_state_x_slab(
        cpml_state_stacked, sharded_grid, mesh,
    )
    out = run_nonuniform_distributed_pec(
        sharded_grid=sharded_grid,
        sharded_materials=sharded_mat,
        sharded_pec_mask=None,
        n_steps=n_steps,
        sources=[src_spec],
        probes=[prb_spec],
        n_devices=n_devices,
        devices=devices,
        cpml_params=cpml_params,
        cpml_state=cpml_state_sharded,
    )
    final_cpml = out["cpml_state_sharded"]
    assert final_cpml is not None
    n_cpml = sharded_grid.cpml_layers

    # Rank 0 must have non-zero x_lo psi after the wave hits the boundary
    rank0_psi_ey_xlo = _phase2c_unstack_cpml(
        final_cpml, sharded_grid, n_cpml, "psi_ey_xlo",
    )[0]
    assert np.max(np.abs(rank0_psi_ey_xlo)) > 0, (
        "Phase 2C: rank 0's psi_ey_xlo must be non-zero after the "
        "wavefront reaches the x_lo boundary — outer x-face CPML inactive."
    )

    # Rank N-1 must have non-zero x_hi psi
    rankN_psi_ey_xhi = _phase2c_unstack_cpml(
        final_cpml, sharded_grid, n_cpml, "psi_ey_xhi",
    )[n_devices - 1]
    assert np.max(np.abs(rankN_psi_ey_xhi)) > 0, (
        "Phase 2C: rank N-1's psi_ey_xhi must be non-zero after the "
        "wavefront reaches the x_hi boundary — outer x-face CPML inactive."
    )
