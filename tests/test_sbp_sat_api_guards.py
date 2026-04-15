"""API guard tests for the canonical Phase-1 z-slab lane."""

import pytest

from rfx import Simulation


def test_subgrid_touching_cpml_fails():
    sim = Simulation(freq_max=5e9, domain=(0.04, 0.04, 0.04), boundary="cpml", dx=2e-3)
    with pytest.raises(ValueError, match="boundary='pec' only|CPML/UPML coexistence"):
        sim.add_refinement(z_range=(0.01, 0.03), ratio=3)


def test_partial_xy_refinement_fails():
    sim = Simulation(freq_max=5e9, domain=(0.04, 0.04, 0.04), boundary="pec", dx=2e-3)
    with pytest.raises(ValueError, match="does not support xy_margin"):
        sim.add_refinement(z_range=(0.01, 0.03), ratio=3, xy_margin=1e-3)


def test_source_outside_zslab_fails():
    sim = Simulation(freq_max=5e9, domain=(0.04, 0.04, 0.04), boundary="pec", dx=2e-3)
    sim.add_refinement(z_range=(0.012, 0.020), ratio=2)
    sim.add_source(position=(0.02, 0.02, 0.032), component="ez")
    sim.add_probe(position=(0.02, 0.02, 0.032), component="ez")
    with pytest.raises(ValueError, match="outside .*z-slab fine grid|Widen z_range"):
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
            lambda sim: sim.add_lumped_rlc((0.02, 0.02, 0.02), component="ez", R=50.0),
            "does not support lumped RLC",
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
