"""Phase-1 3D SBP-SAT z-slab tests."""

import inspect
import numpy as np
import jax.numpy as jnp
import pytest

from rfx.subgridding import jit_runner
from rfx.subgridding.sbp_sat_3d import (
    _active_corners,
    _active_edges,
    _active_faces,
    compute_energy_3d,
    init_subgrid_3d,
    step_subgrid_3d,
)


def test_zslab_energy_pec_1000_steps():
    config, state = init_subgrid_3d(
        shape_c=(8, 8, 10),
        dx_c=0.004,
        fine_region=(0, 8, 0, 8, 3, 7),
        ratio=2,
        tau=0.5,
    )
    state = state._replace(ez_c=state.ez_c.at[3, 3, 2].set(1.0))
    initial_energy = compute_energy_3d(state, config)

    max_energy = initial_energy
    for i in range(1000):
        state = step_subgrid_3d(state, config)
        if (i + 1) % 100 == 0:
            energy = compute_energy_3d(state, config)
            assert energy <= initial_energy * 1.02, (
                f"Energy exceeded transient bound at step {i+1}: "
                f"{energy/initial_energy:.4f}"
            )
            max_energy = max(max_energy, energy)

    final_energy = compute_energy_3d(state, config)
    assert np.isfinite(final_energy)
    assert final_energy <= initial_energy
    assert max_energy <= initial_energy * 1.02


def test_init_subgrid_3d_accepts_arbitrary_box_region():
    config, _ = init_subgrid_3d(
        shape_c=(8, 8, 10),
        dx_c=0.004,
        fine_region=(1, 7, 1, 6, 3, 7),
        ratio=2,
    )
    assert (config.fi_lo, config.fi_hi) == (1, 7)
    assert (config.fj_lo, config.fj_hi) == (1, 6)
    assert (config.fk_lo, config.fk_hi) == (3, 7)


def test_init_subgrid_3d_default_region_is_full_span_xy():
    config, _ = init_subgrid_3d(shape_c=(8, 8, 10), dx_c=0.004, ratio=2)
    assert (config.fi_lo, config.fi_hi) == (0, 8)
    assert (config.fj_lo, config.fj_hi) == (0, 8)


def test_zslab_fine_grid_receives_signal():
    config, state = init_subgrid_3d(
        shape_c=(8, 8, 10),
        dx_c=0.004,
        fine_region=(0, 8, 0, 8, 3, 7),
        ratio=2,
    )
    state = state._replace(ez_c=state.ez_c.at[4, 4, 2].set(1.0))
    for _ in range(200):
        state = step_subgrid_3d(state, config)

    max_f = float(jnp.max(jnp.abs(state.ez_f)))
    assert np.isfinite(max_f)
    assert max_f > 1e-8


def test_box_fine_grid_receives_signal():
    config, state = init_subgrid_3d(
        shape_c=(8, 8, 10),
        dx_c=0.004,
        fine_region=(2, 6, 2, 6, 3, 7),
        ratio=2,
    )
    state = state._replace(ez_c=state.ez_c.at[1, 4, 5].set(1.0))
    for _ in range(250):
        state = step_subgrid_3d(state, config)

    max_f = float(jnp.max(jnp.abs(state.ez_f)))
    assert np.isfinite(max_f)
    assert max_f > 1e-8


def test_box_energy_pec_1000_steps():
    config, state = init_subgrid_3d(
        shape_c=(8, 8, 10),
        dx_c=0.004,
        fine_region=(2, 6, 2, 6, 3, 7),
        ratio=2,
        tau=0.5,
    )
    state = state._replace(ez_c=state.ez_c.at[1, 4, 5].set(1.0))
    initial_energy = compute_energy_3d(state, config)

    max_energy = initial_energy
    for i in range(1000):
        state = step_subgrid_3d(state, config)
        if (i + 1) % 100 == 0:
            energy = compute_energy_3d(state, config)
            assert energy <= initial_energy * 1.05, (
                f"Box energy exceeded transient bound at step {i+1}: "
                f"{energy/initial_energy:.4f}"
            )
            max_energy = max(max_energy, energy)

    final_energy = compute_energy_3d(state, config)
    assert np.isfinite(final_energy)
    assert final_energy <= initial_energy * 1.05
    assert max_energy <= initial_energy * 1.05


def test_box_edge_region_receives_signal():
    config, state = init_subgrid_3d(
        shape_c=(8, 8, 10),
        dx_c=0.004,
        fine_region=(2, 6, 2, 6, 3, 7),
        ratio=2,
    )
    state = state._replace(ez_c=state.ez_c.at[1, 1, 5].set(1.0))
    for _ in range(250):
        state = step_subgrid_3d(state, config)

    edge_mag = float(jnp.max(jnp.abs(state.ez_f[0:2, 0:2, :])))
    assert np.isfinite(edge_mag)
    assert edge_mag > 1e-8


def test_box_corner_region_receives_signal():
    config, state = init_subgrid_3d(
        shape_c=(8, 8, 10),
        dx_c=0.004,
        fine_region=(2, 6, 2, 6, 3, 7),
        ratio=2,
    )
    state = state._replace(ez_c=state.ez_c.at[1, 1, 1].set(1.0))
    for _ in range(250):
        state = step_subgrid_3d(state, config)

    corner_mag = float(jnp.max(jnp.abs(state.ez_f[0:2, 0:2, 0:2])))
    assert np.isfinite(corner_mag)
    assert corner_mag > 1e-8


def test_phase1_uses_single_canonical_stepper():
    source = inspect.getsource(jit_runner.run_subgridded_jit)
    assert "step_subgrid_3d" in source
    assert "_shared_node_coupling_3d" not in source
    assert "_shared_node_coupling_h_3d" not in source


def test_active_interfaces_for_full_span_zslab():
    config, _ = init_subgrid_3d(
        shape_c=(8, 8, 10),
        dx_c=0.004,
        fine_region=(0, 8, 0, 8, 3, 7),
        ratio=2,
    )
    assert _active_faces(config) == ("z_lo", "z_hi")
    assert _active_edges(config) == ()
    assert _active_corners(config) == ()


def test_active_interfaces_for_interior_box():
    config, _ = init_subgrid_3d(
        shape_c=(8, 8, 10),
        dx_c=0.004,
        fine_region=(2, 6, 2, 6, 3, 7),
        ratio=2,
    )
    assert set(_active_faces(config)) == {"x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi"}
    assert len(_active_edges(config)) == 12
    assert len(_active_corners(config)) == 8
