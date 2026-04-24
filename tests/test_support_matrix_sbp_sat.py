"""Regression locks for SBP-SAT benchmark claim level."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUPPORT_MATRIX = ROOT / "docs/guides/support_matrix.json"
TRUE_RT_SPEC = ROOT / "docs/guides/sbp_sat_zslab_true_rt_benchmark_spec.md"


def _sbp_lane() -> dict:
    data = json.loads(SUPPORT_MATRIX.read_text())
    return data["lanes"]["sbp_sat_subgridding"]


def test_sbp_sat_support_matrix_records_experimental_proxy_evidence():
    lane = _sbp_lane()

    assert lane["status"] == "experimental"
    assert lane["supported_subset"]["boundary"] == "all_pec_only"
    assert lane["supported_subset"]["geometry"] == "all_pec_arbitrary_box_only"
    assert lane["supported_subset"]["sources"] == ["soft_point_source"]
    assert lane["supported_subset"]["observables"] == ["point_probe"]

    proxy = lane["benchmark_evidence"]["proxy_crossval"]
    assert proxy["status"] == "implemented"
    assert proxy["test_file"] == "tests/test_subgrid_crossval.py"
    assert proxy["claim_level"] == "proxy_only_not_physical_rt"
    assert proxy["tolerance"] == {
        "relative_amplitude_error_max": 0.05,
        "phase_error_deg_max": 5.0,
    }

    box_proxy = lane["benchmark_evidence"]["box_proxy_crossval"]
    assert box_proxy["status"] == "implemented"
    assert box_proxy["test_file"] == "tests/test_sbp_sat_box_crossval.py"
    assert box_proxy["fixtures"] == [
        "x_face_proxy",
        "y_face_proxy",
        "edge_proxy",
        "corner_proxy",
    ]


def test_sbp_sat_true_rt_benchmark_is_explicitly_deferred():
    true_rt = _sbp_lane()["benchmark_evidence"]["true_reflection_transmission"]

    assert true_rt["status"] == "deferred"
    assert true_rt["spec"] == "docs/guides/sbp_sat_zslab_true_rt_benchmark_spec.md"
    assert true_rt["deferred_issue"].endswith("#deferred-issue-record")
    assert true_rt["public_claim_allowed"] is False
    assert TRUE_RT_SPEC.exists()

    spec_text = TRUE_RT_SPEC.read_text()
    assert "proxy numerical-equivalence benchmark" in spec_text
    assert "does **not** establish calibrated physical reflection" in spec_text
    assert "true R/T benchmark: deferred" in spec_text
    assert "## Deferred issue record" in spec_text


def test_sbp_sat_unsupported_surface_remains_hard_fail_in_support_matrix():
    unsupported = _sbp_lane()["unsupported_combinations"]

    for key in (
        "cpml_upml_boundary",
        "pmc_or_periodic_boundaryspec",
        "xy_margin_geometry_autobox",
        "ntff",
        "dft_plane",
        "flux_monitor",
        "tfsf",
        "waveguide_port",
        "floquet_port",
        "lumped_rlc",
        "coaxial_port",
        "impedance_or_wire_port",
    ):
        assert unsupported[key] == "hard_fail"
