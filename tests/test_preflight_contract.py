from __future__ import annotations

import inspect

import numpy as np
import pytest

from rfx import (
    Box,
    DesignRegion,
    GaussianPulse,
    Simulation,
    TopologyDesignRegion,
    maximize_directivity,
    maximize_transmitted_energy,
    minimize_reflected_energy,
    optimize,
    topology_optimize,
)
import rfx.preflight as preflight_mod
from rfx.preflight import PreflightIssue, PreflightReport


F0 = 5e9


def _small_region() -> DesignRegion:
    return DesignRegion(corner_lo=(0.010, 0.0, 0.0), corner_hi=(0.014, 0.01, 0.01), eps_range=(1.0, 4.4))


def _build_ported_sim(*, boundary: str = "pec", domain=(0.03, 0.01, 0.01), dx: float = 0.001, port_x: float = 0.008, probe_x: float = 0.022) -> Simulation:
    kwargs = {"freq_max": F0, "domain": domain, "boundary": boundary, "dx": dx}
    if boundary == "cpml":
        kwargs["cpml_layers"] = 8
    sim = Simulation(**kwargs)
    sim.add_port((port_x, 0.005, 0.005), "ez", impedance=50.0, waveform=GaussianPulse(f0=F0, bandwidth=0.5))
    sim.add_probe((probe_x, 0.005, 0.005), "ez")
    return sim


def test_public_preflight_api_exists_and_signature_surface_is_present():
    sim = _build_ported_sim()
    assert hasattr(sim, "preflight_optimize")
    assert hasattr(sim, "preflight_topology_optimize")

    optimize_sig = inspect.signature(optimize)
    assert "preflight_mode" in optimize_sig.parameters
    assert "memory_budget_mb" in optimize_sig.parameters

    topology_sig = inspect.signature(topology_optimize)
    assert "n_steps" in topology_sig.parameters
    assert "preflight_mode" in topology_sig.parameters
    assert "memory_budget_mb" in topology_sig.parameters


def test_preflight_report_strict_semantics():
    report = PreflightReport(
        optimizer_name="optimize",
        objective_name="demo",
        n_steps=10,
        n_steps_auto=False,
        grid_shape=(8, 8, 8),
        cells=512,
        trace_cost=5120,
        issues=[PreflightIssue(code="WARN", severity="warning", message="warn")],
    )
    assert report.ok
    assert not report.strict_ok
    with pytest.raises(ValueError):
        report.enforce("strict")


def test_cpml_boundary_clearance_warning_is_reported():
    sim = _build_ported_sim(boundary="cpml", dx=0.002, port_x=0.006, probe_x=0.020)
    report = sim.preflight_optimize(_small_region(), maximize_transmitted_energy(output_probe_idx=0), n_steps=20)
    codes = {issue.code for issue in report.issues}
    assert "PORT_NEAR_CPML_BOUNDARY" in codes


def test_zero_thickness_pec_on_nonuniform_mesh_is_rejected():
    sim = Simulation(
        freq_max=F0,
        domain=(0.03, 0.02, 0.004),
        boundary="pec",
        dx=0.001,
        dz_profile=np.array([0.0005] * 8, dtype=float),
    )
    sim.add(Box((0.005, 0.005, 0.001), (0.020, 0.015, 0.001)), material="pec")
    sim.add_port((0.008, 0.010, 0.002), "ez", impedance=50.0, waveform=GaussianPulse(f0=F0, bandwidth=0.5))
    sim.add_probe((0.022, 0.010, 0.002), "ez")
    report = sim.preflight_optimize(_small_region(), maximize_transmitted_energy(output_probe_idx=0), n_steps=20)
    codes = {issue.code for issue in report.issues}
    assert "ZERO_THICKNESS_PEC_NONUNIFORM_UNSUPPORTED" in codes
    assert not report.ok


def test_under_resolved_dielectric_layer_warns():
    sim = _build_ported_sim(dx=0.001)
    sim.add(Box((0.000, 0.000, 0.0005), (0.030, 0.010, 0.0020)), material="fr4")
    report = sim.preflight_optimize(_small_region(), maximize_transmitted_energy(output_probe_idx=0), n_steps=20)
    codes = {issue.code for issue in report.issues}
    assert "CRITICAL_LAYER_POORLY_RESOLVED" in codes


def test_soft_source_warning_for_port_like_proxy_objective():
    sim = Simulation(freq_max=F0, domain=(0.03, 0.01, 0.01), boundary="pec", dx=0.001)
    sim.add_source((0.008, 0.005, 0.005), "ez", waveform=GaussianPulse(f0=F0, bandwidth=0.5))
    sim.add_probe((0.008, 0.005, 0.005), "ez")
    report = sim.preflight_optimize(_small_region(), minimize_reflected_energy(port_probe_idx=0), n_steps=20)
    codes = {issue.code for issue in report.issues}
    assert "SOFT_SOURCE_OBJECTIVE_MISMATCH" in codes


def test_pec_boundary_ntff_warning_is_reported():
    sim = _build_ported_sim(boundary="pec")
    sim.add_ntff_box((0.006, 0.002, 0.002), (0.024, 0.008, 0.008))
    report = sim.preflight_optimize(_small_region(), maximize_directivity(theta_target=0.0, phi_target=0.0), n_steps=20)
    codes = {issue.code for issue in report.issues}
    assert "PEC_BOUNDARY_WITH_NTFF_RISK" in codes


def test_topology_preflight_api_returns_report():
    sim = _build_ported_sim(boundary="pec")
    region = TopologyDesignRegion(
        corner_lo=(0.010, 0.0, 0.0),
        corner_hi=(0.014, 0.01, 0.01),
        material_bg="air",
        material_fg="fr4",
    )
    report = sim.preflight_topology_optimize(region, maximize_transmitted_energy(output_probe_idx=0), n_steps=20)
    assert isinstance(report, PreflightReport)


def test_memory_gate_is_truthful_for_untagged_custom_objective():
    sim = _build_ported_sim(boundary="pec")

    def custom_objective(result):
        return result.time_series[:, 0].sum()

    report = sim.preflight_optimize(_small_region(), custom_objective, n_steps=20, memory_budget_mb=1.0)
    codes = {issue.code for issue in report.issues}
    assert "COMPILED_MEMORY_CHECK_UNAVAILABLE" in codes


def test_memory_gate_budget_exceeded_is_reported(monkeypatch):
    sim = _build_ported_sim(boundary="pec")

    def fake_gate(*args, **kwargs):
        return {"argument_mb": 1.0, "output_mb": 1.0, "temp_mb": 9.0, "total_mb": 11.0}, None

    monkeypatch.setattr(preflight_mod, "_compile_memory_gate_optimize", fake_gate)
    report = sim.preflight_optimize(
        _small_region(),
        maximize_transmitted_energy(output_probe_idx=0),
        n_steps=20,
        memory_budget_mb=5.0,
    )
    codes = {issue.code for issue in report.issues}
    assert "COMPILED_MEMORY_BUDGET_EXCEEDED" in codes
