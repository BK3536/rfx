"""Principled coverage for the coaxial-port abstraction.

These tests exercise the current public/high-level builder plus the low-level
material/source helpers that already define the coaxial-port behavior.
"""

from __future__ import annotations

import pytest

from rfx import CoaxialPort, GaussianPulse, Simulation
from rfx.core.yee import EPS_0, init_materials, init_state
from rfx.grid import Grid
from rfx.sources.coaxial_port import (
    PEC_SIGMA,
    PTFE_EPS_R,
    SMA_OUTER_RADIUS,
    SMA_PIN_RADIUS,
    make_coaxial_port_source,
    setup_coaxial_port,
)


class ConstantWaveform:
    def __init__(self, amplitude: float):
        self.amplitude = amplitude

    def __call__(self, t: float) -> float:
        return self.amplitude


FACE_CASES = [
    pytest.param("top", (0.010, 0.010, 0.015), "ez", id="top"),
    pytest.param("bottom", (0.010, 0.010, 0.005), "ez", id="bottom"),
    pytest.param("front", (0.010, 0.005, 0.010), "ey", id="front"),
    pytest.param("back", (0.010, 0.015, 0.010), "ey", id="back"),
    pytest.param("left", (0.005, 0.010, 0.010), "ex", id="left"),
    pytest.param("right", (0.015, 0.010, 0.010), "ex", id="right"),
]


def _make_grid() -> Grid:
    return Grid(
        freq_max=10e9,
        domain=(0.020, 0.020, 0.020),
        dx=0.5e-3,
        cpml_layers=0,
    )


def test_add_coaxial_port_records_defaults_on_simulation():
    sim = Simulation(freq_max=8e9, domain=(0.020, 0.020, 0.020), boundary="cpml")

    returned = sim.add_coaxial_port((0.010, 0.010, 0.015))

    assert returned is sim
    assert len(sim._coaxial_ports) == 1

    port = sim._coaxial_ports[0]
    assert isinstance(port, CoaxialPort)
    assert port.position == (0.010, 0.010, 0.015)
    assert port.face == "top"
    assert port.pin_length == pytest.approx(5e-3)
    assert port.pin_radius == pytest.approx(SMA_PIN_RADIUS)
    assert port.outer_radius == pytest.approx(SMA_OUTER_RADIUS)
    assert port.impedance == pytest.approx(50.0)
    assert isinstance(port.excitation, GaussianPulse)
    assert port.excitation.f0 == pytest.approx(4e9)
    assert port.excitation.bandwidth == pytest.approx(0.8)


def test_add_coaxial_port_preserves_explicit_parameters_and_waveform():
    waveform = ConstantWaveform(1.25)
    sim = Simulation(freq_max=6e9, domain=(0.020, 0.020, 0.020), boundary="pec")

    sim.add_coaxial_port(
        (0.005, 0.010, 0.010),
        face="left",
        pin_length=3.0e-3,
        pin_radius=0.4e-3,
        outer_radius=1.4e-3,
        impedance=75.0,
        waveform=waveform,
    )

    port = sim._coaxial_ports[0]
    assert port.face == "left"
    assert port.pin_length == pytest.approx(3.0e-3)
    assert port.pin_radius == pytest.approx(0.4e-3)
    assert port.outer_radius == pytest.approx(1.4e-3)
    assert port.impedance == pytest.approx(75.0)
    assert port.excitation is waveform


def test_setup_coaxial_port_stamps_pin_dielectric_and_gap_conductivity():
    grid = _make_grid()
    base_materials = init_materials(grid.shape)
    port = CoaxialPort(
        position=(0.010, 0.010, 0.015),
        face="top",
        pin_length=5e-3,
        pin_radius=SMA_PIN_RADIUS,
        outer_radius=SMA_OUTER_RADIUS,
        impedance=50.0,
        excitation=ConstantWaveform(0.0),
    )

    stamped = setup_coaxial_port(grid, port, base_materials)

    pin_idx = grid.position_to_index((0.010, 0.010, 0.0125))
    ptfe_idx = grid.position_to_index((0.0110, 0.010, 0.0125))
    free_idx = grid.position_to_index((0.0140, 0.010, 0.0125))
    gap_idx = grid.position_to_index(port.position)

    assert float(stamped.eps_r[pin_idx]) == pytest.approx(1.0)
    assert float(stamped.sigma[pin_idx]) >= PEC_SIGMA

    assert float(stamped.eps_r[ptfe_idx]) == pytest.approx(PTFE_EPS_R)
    assert float(stamped.sigma[ptfe_idx]) == pytest.approx(0.0)

    assert float(stamped.eps_r[free_idx]) == pytest.approx(1.0)
    assert float(stamped.sigma[free_idx]) == pytest.approx(0.0)

    sigma_port = 1.0 / (port.impedance * grid.dx)
    assert float(stamped.sigma[gap_idx]) >= sigma_port


@pytest.mark.parametrize(("face", "position", "component"), FACE_CASES)
def test_make_coaxial_port_source_injects_expected_e_field_component(face, position, component):
    grid = _make_grid()
    waveform = ConstantWaveform(2.5)
    port = CoaxialPort(
        position=position,
        face=face,
        pin_length=5e-3,
        pin_radius=SMA_PIN_RADIUS,
        outer_radius=SMA_OUTER_RADIUS,
        impedance=50.0,
        excitation=waveform,
    )
    materials = setup_coaxial_port(grid, port, init_materials(grid.shape))
    source = make_coaxial_port_source(grid, port, materials, n_steps=8)
    state = init_state(grid.shape)

    updated = source(state, 0.0)
    gap_idx = grid.position_to_index(position)

    eps = float(materials.eps_r[gap_idx]) * EPS_0
    sigma = float(materials.sigma[gap_idx])
    loss = sigma * grid.dt / (2.0 * eps)
    cb = (grid.dt / eps) / (1.0 + loss)
    expected_delta = cb * waveform.amplitude / grid.dx

    for field_name in ("ex", "ey", "ez"):
        field = getattr(updated, field_name)
        observed = float(field[gap_idx])
        if field_name == component:
            assert observed == pytest.approx(expected_delta)
        else:
            assert observed == pytest.approx(0.0)
