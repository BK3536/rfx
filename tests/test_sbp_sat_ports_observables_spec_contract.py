"""Regression locks for the Milestone 7 ports/observables RFC."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RFC = ROOT / "docs/guides/sbp_sat_ports_observables_rfc.md"
FULL_SPEC = ROOT / "docs/guides/sbp_sat_zslab_phase1_full_spec.md"


def _text(path: Path) -> str:
    return path.read_text()


def test_ports_observables_rfc_has_required_sections():
    text = _text(RFC)

    for heading in (
        "# SBP-SAT ports and observables RFC",
        "## Status",
        "## Current baseline",
        "## Port and observable classes",
        "## Source normalization contract",
        "## Placement contract for future support",
        "## DFT / flux / NTFF contract",
        "## Unsupported-combination hard-fail matrix",
        "## Benchmark families",
        "## Implementation plan",
        "## Implementation gate",
    ):
        assert heading in text


def test_ports_observables_rfc_locks_current_supported_surface():
    text = _text(RFC)

    for token in (
        "soft point sources",
        "point probes",
        "maps to a valid **fine-grid cell index** inside the refined box",
        "hard-fail with a placement",
        "error rather than silently sampling or exciting the coarse grid",
        "state = result.state_f",
        "grid = fine_grid",
        "fine-grid-centric",
    ):
        assert token in text


def test_ports_observables_rfc_locks_normalization_contract():
    text = _text(RFC)

    for token in (
        "raw field add",
        "make_j_source(...)",
        "Cb = dt / (eps * (1 + sigma*dt/(2*eps)))",
        "make_port_source(...)",
        "make_wire_port_sources(...)",
        "waveform_port = (Cb / d_parallel) * excitation(t)",
        "multi-cell wire sources divide the excitation by `N_cells`",
        "raw field amplitude",
        "total current",
        "total voltage drop",
        "delivered power",
    ):
        assert token in text


def test_ports_observables_rfc_locks_port_observable_classes_and_placements():
    text = _text(RFC)

    for token in (
        "impedance point port",
        "wire / extent port",
        "coaxial port",
        "waveguide port",
        "Floquet port",
        "DFT plane probe",
        "flux monitor",
        "NTFF / Huygens box",
        "fine_cell",
        "fine_face",
        "fine_edge_or_corner",
        "coarse_exterior",
        "cross_interface_surface",
        "outer_boundary_attached",
        "one positive placement test",
        "unsupported-placement failure test",
    ):
        assert token in text


def test_ports_observables_rfc_locks_hard_fail_matrix_and_benchmarks():
    text = _text(RFC)

    for token in (
        "soft point source outside refined box",
        "point probe outside refined box",
        "impedance point port in refined box",
        "wire / extent port in refined box",
        "coaxial port with subgrid",
        "waveguide port with subgrid",
        "Floquet port with subgrid",
        "DFT plane with subgrid",
        "flux monitor with subgrid",
        "NTFF with subgrid",
        "any interface-crossing port/observable",
        "### PO-1: impedance point-port benchmark",
        "### PO-2: wire / extent port benchmark",
        "### PO-3: coaxial feed benchmark",
        "### PO-4: waveguide modal benchmark",
        "### PO-5: Floquet / periodic benchmark",
        "### PO-6: planar / far-field observable benchmark",
    ):
        assert token in text


def test_ports_observables_rfc_blocks_runtime_enablement():
    text = _text(RFC)

    for token in (
        "It does **not** widen shipped runtime support",
        "does **not** mean additional ports or observables are implemented",
        "the item has an explicit normalization contract",
        "every supported placement class has a positive test",
        "at least one nearby unsupported placement has a negative test",
        "the support matrix and public docs are updated only after those gates pass",
    ):
        assert token in text


def test_full_spec_references_milestone7_artifact_and_contract_test():
    text = _text(FULL_SPEC)

    assert "docs/guides/sbp_sat_ports_observables_rfc.md" in text
    assert "tests/test_sbp_sat_ports_observables_spec_contract.py" in text
