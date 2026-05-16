"""Fine-shape parity between the two subgrid region builders.

``build_subgrid_region`` (overlap z-slab) and ``build_stage2_disjoint_region``
(Stage-2 disjoint) must agree on the fine-grid extent for a shared
``(z_range, ratio)`` fixture. The runner path (``rfx.runners.subgridded`` and
the ``SubgridConfig3D`` builder in ``rfx.subgridding.sbp_sat_3d``) consumes the
region with the cell-extent convention ``(hi - lo) * ratio``; both builders
must use that one convention so validation and the runner stay consistent.
"""

from __future__ import annotations

import pytest

from rfx.api import Simulation
from rfx.subgridding.validation import (
    build_stage2_disjoint_region,
    build_subgrid_region,
)


@pytest.fixture
def shared_region_fixture():
    """Return ``(sim, grid, z_range, ratio)`` shared by both builders."""
    z_range = (0.004, 0.012)
    ratio = 4
    sim = Simulation(
        freq_max=6e9,
        domain=(0.04, 0.04, 0.02),
        boundary="pec",
        cpml_layers=0,
        dx=0.002,
    )
    sim.add_refinement(z_range, ratio=ratio)
    grid = sim._build_grid()
    return sim, grid, z_range, ratio


def test_both_builders_agree_on_fine_shape(shared_region_fixture):
    """Both region builders yield an identical fine-grid shape."""
    sim, grid, _z_range, _ratio = shared_region_fixture

    overlap_region = build_subgrid_region(sim, grid)
    disjoint_region = build_stage2_disjoint_region(sim, grid)

    assert overlap_region is not None
    overlap_shape = (overlap_region.nx_f, overlap_region.ny_f, overlap_region.nz_f)
    disjoint_shape = (disjoint_region.nx_f, disjoint_region.ny_f, disjoint_region.nz_f)
    assert overlap_shape == disjoint_shape


def test_fine_shape_uses_cell_extent_convention(shared_region_fixture):
    """The fine shape follows the runner's ``(hi - lo) * ratio`` convention."""
    sim, grid, _z_range, ratio = shared_region_fixture

    region = build_subgrid_region(sim, grid)
    assert region is not None
    assert region.nx_f == (region.fi_hi - region.fi_lo) * ratio
    assert region.ny_f == (region.fj_hi - region.fj_lo) * ratio
    assert region.nz_f == (region.fk_hi - region.fk_lo) * ratio
