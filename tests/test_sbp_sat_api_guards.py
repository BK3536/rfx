"""API guard tests for the canonical Phase-1 z-slab lane."""

import numpy as np
import pytest

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec


def test_subgrid_touching_cpml_fails():
    sim = Simulation(freq_max=5e9, domain=(0.04, 0.04, 0.04), boundary="cpml", dx=2e-3)
    with pytest.raises(ValueError, match="does not yet support absorbing BoundarySpec faces|boundary='pec' only"):
        sim.add_refinement(z_range=(0.01, 0.03), ratio=3)


def test_subgrid_accepts_all_pec_boundaryspec():
    sim = Simulation(
        freq_max=5e9,
        domain=(0.04, 0.04, 0.04),
        boundary=BoundarySpec.uniform("pec"),
        dx=2e-3,
    )
    sim.add_refinement(z_range=(0.01, 0.03), ratio=3)

    assert sim._refinement["ratio"] == 3


def test_subgrid_accepts_reflector_only_pmc_boundaryspec():
    sim = Simulation(
        freq_max=5e9,
        domain=(0.04, 0.04, 0.04),
        boundary=BoundarySpec(
            x="pec",
            y="pec",
            z=Boundary(lo="pmc", hi="pec"),
        ),
        dx=2e-3,
    )
    sim.add_refinement(z_range=(0.0, 0.028), ratio=2)
    sim.add_source(position=(0.020, 0.020, 0.012), component="ez")
    sim.add_probe(position=(0.020, 0.020, 0.020), component="ez")
    result = sim.run(n_steps=80)

    assert result.time_series.shape == (80, 1)
    assert float(np.max(np.abs(np.asarray(result.time_series)))) > 0.0


def test_subgrid_accepts_periodic_axis_when_box_spans_it():
    sim = Simulation(
        freq_max=5e9,
        domain=(0.04, 0.04, 0.04),
        boundary=BoundarySpec(x="periodic", y="pec", z="pec"),
        dx=2e-3,
    )
    sim.add_refinement(
        x_range=(0.0, 0.04),
        y_range=(0.012, 0.028),
        z_range=(0.012, 0.028),
        ratio=2,
    )
    sim.add_source(position=(0.020, 0.020, 0.020), component="ez")
    sim.add_probe(position=(0.020, 0.020, 0.020), component="ez")
    result = sim.run(n_steps=80)

    assert result.time_series.shape == (80, 1)
    assert float(np.max(np.abs(np.asarray(result.time_series)))) > 0.0


def test_subgrid_accepts_periodic_axis_when_box_is_interior():
    sim = Simulation(
        freq_max=5e9,
        domain=(0.04, 0.04, 0.04),
        boundary=BoundarySpec(x="periodic", y="pec", z="pec"),
        dx=2e-3,
    )
    sim.add_refinement(
        x_range=(0.012, 0.028),
        y_range=(0.012, 0.028),
        z_range=(0.012, 0.028),
        ratio=2,
    )
    sim.add_source(position=(0.020, 0.020, 0.020), component="ez")
    sim.add_probe(position=(0.020, 0.020, 0.020), component="ez")
    result = sim.run(n_steps=80)

    assert result.time_series.shape == (80, 1)
    assert float(np.max(np.abs(np.asarray(result.time_series)))) > 0.0


def test_subgrid_rejects_periodic_axis_touched_on_one_side_only():
    sim = Simulation(
        freq_max=5e9,
        domain=(0.04, 0.04, 0.04),
        boundary=BoundarySpec(x="periodic", y="pec", z="pec"),
        dx=2e-3,
    )
    sim.add_refinement(
        x_range=(0.0, 0.028),
        y_range=(0.012, 0.028),
        z_range=(0.012, 0.028),
        ratio=2,
    )

    with pytest.raises(ValueError, match="periodic axis.*one side"):
        sim.run(n_steps=5)


def test_subgrid_rejects_mixed_pmc_periodic_boundaryspec():
    sim = Simulation(
        freq_max=5e9,
        domain=(0.04, 0.04, 0.04),
        boundary=BoundarySpec(
            x="periodic",
            y="pec",
            z=Boundary(lo="pmc", hi="pec"),
        ),
        dx=2e-3,
    )

    with pytest.raises(ValueError, match="mixed PMC \\+ periodic"):
        sim.add_refinement(z_range=(0.01, 0.03), ratio=2)


def test_all_pec_box_refinement_runs():
    sim = Simulation(
        freq_max=5e9,
        domain=(0.04, 0.04, 0.04),
        boundary="pec",
        dx=2e-3,
    )
    sim.add_refinement(
        z_range=(0.012, 0.028),
        x_range=(0.010, 0.028),
        y_range=(0.010, 0.028),
        ratio=2,
    )
    sim.add_source(position=(0.012, 0.020, 0.020), component="ez")
    sim.add_probe(position=(0.026, 0.020, 0.020), component="ez")
    result = sim.run(n_steps=80)

    assert result.time_series.shape == (80, 1)
    assert float(np.max(np.abs(np.asarray(result.time_series)))) > 0.0


@pytest.mark.parametrize(
    "boundary",
    [
        BoundarySpec(x="pec", y="pec", z=Boundary(lo="pec", hi="cpml")),
        BoundarySpec(x=Boundary(lo="pmc", hi="cpml"), y="pec", z="pec"),
        BoundarySpec(
            x="pec",
            y="pec",
            z=Boundary(lo="cpml", hi="pec", lo_thickness=4),
        ),
    ],
)
def test_subgrid_rejects_non_pec_boundaryspec(boundary):
    sim = Simulation(
        freq_max=5e9,
        domain=(0.04, 0.04, 0.04),
        boundary=boundary,
        dx=2e-3,
    )
    with pytest.raises(ValueError, match="does not yet support absorbing BoundarySpec faces"):
        sim.add_refinement(z_range=(0.01, 0.03), ratio=3)


def test_subgrid_accepts_late_periodic_axes_on_run_when_box_spans_axis():
    sim = Simulation(freq_max=5e9, domain=(0.04, 0.04, 0.04), boundary="pec", dx=2e-3)
    sim.add_refinement(z_range=(0.012, 0.028), ratio=2)
    sim.add_source(position=(0.02, 0.02, 0.02), component="ez")
    sim.add_probe(position=(0.02, 0.02, 0.02), component="ez")
    with pytest.warns(DeprecationWarning):
        sim.set_periodic_axes("x")

    result = sim.run(n_steps=10)
    assert result.time_series.shape == (10, 1)


def test_partial_xy_refinement_fails():
    sim = Simulation(freq_max=5e9, domain=(0.04, 0.04, 0.04), boundary="pec", dx=2e-3)
    with pytest.raises(ValueError, match="does not support xy_margin"):
        sim.add_refinement(z_range=(0.01, 0.03), ratio=3, xy_margin=1e-3)


def test_source_outside_zslab_fails():
    sim = Simulation(freq_max=5e9, domain=(0.04, 0.04, 0.04), boundary="pec", dx=2e-3)
    sim.add_refinement(
        z_range=(0.012, 0.020),
        x_range=(0.012, 0.028),
        y_range=(0.012, 0.028),
        ratio=2,
    )
    sim.add_source(position=(0.02, 0.02, 0.032), component="ez")
    sim.add_probe(position=(0.02, 0.02, 0.032), component="ez")
    with pytest.raises(ValueError, match="outside .*fine grid|Adjust x_range/y_range/z_range"):
        sim.run(n_steps=10)


@pytest.mark.parametrize(
    ("attach_feature", "message"),
    [
        (
            lambda sim: sim.add_ntff_box((0.008, 0.008, 0.008), (0.032, 0.032, 0.032)),
            "does not support NTFF",
        ),
        (
            lambda sim: sim.add_dft_plane_probe(axis="z", coordinate=0.018, component="ez"),
            "does not support DFT plane probes",
        ),
        (
            lambda sim: sim.add_flux_monitor(axis="z", coordinate=0.018, n_freqs=4),
            "does not support flux monitors",
        ),
        (
            lambda sim: sim._waveguide_ports.append(object()),
            "does not support waveguide ports",
        ),
        (
            lambda sim: setattr(sim, "_tfsf", object()),
            "does not support TFSF sources",
        ),
        (
            lambda sim: sim.add_lumped_rlc((0.02, 0.02, 0.02), component="ez", R=50.0),
            "does not support lumped RLC",
        ),
        (
            lambda sim: sim.add_coaxial_port((0.02, 0.02, 0.04), face="top"),
            "does not support coaxial ports",
        ),
        (
            lambda sim: sim.add_floquet_port(position=0.01, axis="z"),
            "does not support Floquet ports",
        ),
    ],
)
def test_unsupported_phase1_features_fail_fast(attach_feature, message):
    sim = Simulation(freq_max=5e9, domain=(0.04, 0.04, 0.04), boundary="pec", dx=2e-3)
    sim.add_refinement(z_range=(0.012, 0.028), ratio=2)
    sim.add_source(position=(0.02, 0.02, 0.02), component="ez")
    sim.add_probe(position=(0.02, 0.02, 0.02), component="ez")
    attach_feature(sim)
    with pytest.raises(ValueError, match=message):
        sim.run(n_steps=10)


@pytest.mark.parametrize(
    ("attach_port", "message"),
    [
        (
            lambda sim: sim.add_port(
                position=(0.02, 0.02, 0.02),
                component="ez",
                impedance=50.0,
            ),
            "soft point sources only|impedance point ports",
        ),
        (
            lambda sim: sim.add_port(
                position=(0.02, 0.02, 0.018),
                component="ez",
                impedance=50.0,
                extent=0.004,
            ),
            "soft point sources only|wire/extent ports",
        ),
    ],
)
def test_subgrid_rejects_impedance_ports(attach_port, message):
    sim = Simulation(freq_max=5e9, domain=(0.04, 0.04, 0.04), boundary="pec", dx=2e-3)
    sim.add_refinement(z_range=(0.012, 0.028), ratio=2)
    sim.add_probe(position=(0.02, 0.02, 0.02), component="ez")
    attach_port(sim)
    with pytest.raises(ValueError, match=message):
        sim.run(n_steps=10)
