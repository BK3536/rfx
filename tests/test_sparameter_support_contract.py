"""Contract tests for port-family-specific S-parameter support.

These tests lock the public API boundary: unsupported port/source families must
fail loudly instead of returning ``None`` after an explicit S-parameter request.
They do not run FDTD.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rfx import MSLSMatrixResult, Simulation


MATRIX_PATH = Path("docs/guides/sparameter_support_matrix.json")


def test_sparameter_support_matrix_lists_every_public_port_surface():
    data = json.loads(MATRIX_PATH.read_text())
    primitives = {entry["primitive"] for entry in data["port_families"]}

    expected = {
        "add_port(extent=None)",
        "add_port(extent=...)",
        "add_msl_port(...)",
        "add_waveguide_port(...)",
        "add_coaxial_port(...)",
        "add_floquet_port(...)",
        "add_source(...) / add_polarized_source(...)",
        "add_tfsf_source(...)",
        "add_probe(...) / add_dft_plane_probe(...) / add_flux_monitor(...)",
    }
    assert expected <= primitives
    assert data["result_convention"]["full_s_matrix_shape"] == "(n_ports, n_ports, n_freqs)"
    assert data["result_convention"]["indexing"].startswith("S[receiver_port")


def test_msl_result_is_publicly_importable():
    assert MSLSMatrixResult.__name__ == "MSLSMatrixResult"


def test_run_sparams_rejects_source_only_request():
    sim = Simulation(freq_max=5e9, domain=(0.01, 0.01, 0.01), boundary="pec")
    sim.add_source((0.005, 0.005, 0.005), "ez")

    with pytest.raises(ValueError, match="add_source.*cannot populate"):
        sim.run(n_steps=1, compute_s_params=True)


def test_run_sparams_rejects_msl_port_family_with_actionable_api():
    sim = Simulation(freq_max=10e9, domain=(0.02, 0.006, 0.002), dx=0.5e-3, boundary="pec")
    sim.add_msl_port(
        position=(0.004, 0.003, 0.0),
        width=0.5e-3,
        height=0.5e-3,
        direction="+x",
    )

    with pytest.raises(ValueError, match="compute_msl_s_matrix"):
        sim.run(n_steps=1, compute_s_params=True)


def test_preflight_sparameters_routes_msl_before_running_fdtd():
    sim = Simulation(freq_max=10e9, domain=(0.02, 0.006, 0.002), dx=0.5e-3, boundary="pec")
    sim.add_msl_port(
        position=(0.004, 0.003, 0.0),
        width=0.5e-3,
        height=0.5e-3,
        direction="+x",
    )

    issues = sim.preflight_sparameters(calculator="run")

    assert any("compute_msl_s_matrix" in issue for issue in issues)
    assert sim.preflight_sparameters(calculator="msl") == []


def test_run_sparams_rejects_waveguide_port_family_with_actionable_api():
    sim = Simulation(
        freq_max=12e9,
        domain=(0.10, 0.023, 0.010),
        dx=1e-3,
        boundary="cpml",
        cpml_layers=4,
    )
    sim.add_waveguide_port(
        x_position=0.010,
        y_range=(0.0, 0.023),
        z_range=(0.0, 0.010),
        direction="+x",
        f0=10e9,
        name="wg",
    )

    with pytest.raises(ValueError, match="compute_waveguide_s_matrix"):
        sim.run(n_steps=1, compute_s_params=True)


def test_preflight_sparameters_catches_waveguide_cardinality():
    sim = Simulation(
        freq_max=12e9,
        domain=(0.10, 0.023, 0.010),
        dx=1e-3,
        boundary="cpml",
        cpml_layers=4,
    )
    sim.add_waveguide_port(
        x_position=0.010,
        y_range=(0.0, 0.023),
        z_range=(0.0, 0.010),
        direction="+x",
        f0=10e9,
        name="wg",
    )

    issues = sim.preflight_sparameters(calculator="waveguide")

    assert any("at least two waveguide ports" in issue for issue in issues)


def test_run_sparams_rejects_floquet_experimental_family():
    sim = Simulation(freq_max=10e9, domain=(0.015, 0.015, 0.030), dx=1e-3, boundary="cpml")
    sim.add_floquet_port(0.005, axis="z")

    with pytest.raises(ValueError, match="floquet"):
        sim.run(n_steps=1, compute_s_params=True)


def test_run_rejects_unwired_coaxial_port():
    sim = Simulation(freq_max=8e9, domain=(0.020, 0.020, 0.020), dx=1e-3, boundary="pec")
    sim.add_coaxial_port((0.010, 0.010, 0.015))

    with pytest.raises(NotImplementedError, match="add_coaxial_port"):
        sim.run(n_steps=1, compute_s_params=False)


def test_forward_s11_rejects_non_port_source_only_request():
    sim = Simulation(freq_max=5e9, domain=(0.01, 0.01, 0.01), dx=1e-3, boundary="pec")
    sim.add_source((0.005, 0.005, 0.005), "ez")

    with pytest.raises(ValueError, match="add_source.*not an impedance port"):
        sim.forward(n_steps=1, port_s11_freqs=np.array([1.0e9]), skip_preflight=True)


def test_preflight_sparameters_routes_forward_specialized_families():
    sim = Simulation(
        freq_max=12e9,
        domain=(0.10, 0.023, 0.010),
        dx=1e-3,
        boundary="cpml",
        cpml_layers=4,
    )
    sim.add_waveguide_port(
        x_position=0.010,
        y_range=(0.0, 0.023),
        z_range=(0.0, 0.010),
        direction="+x",
        f0=10e9,
        name="wg",
    )

    issues = sim.preflight_sparameters(calculator="forward")

    assert any("waveguide ports use compute_waveguide_s_matrix" in issue for issue in issues)


def test_preflight_sparameters_rejects_unknown_calculator():
    sim = Simulation(freq_max=5e9, domain=(0.01, 0.01, 0.01), dx=1e-3, boundary="pec")

    with pytest.raises(ValueError, match="Unknown S-parameter calculator"):
        sim.preflight_sparameters(calculator="not-a-calculator")


def test_compute_msl_s_matrix_rejects_nonuniform_profiles():
    sim = Simulation(
        freq_max=10e9,
        domain=(0.02, 0.006, 0.002),
        dx=0.5e-3,
        dz_profile=np.full(4, 0.5e-3),
        boundary="pec",
    )
    sim.add_msl_port(
        position=(0.004, 0.003, 0.0),
        width=0.5e-3,
        height=0.5e-3,
        direction="+x",
    )

    with pytest.raises(NotImplementedError, match="uniform Yee lane only"):
        sim.compute_msl_s_matrix(n_steps=1)
