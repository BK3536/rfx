"""Regression locks for the Milestone 6 boundary coexistence RFC."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RFC = ROOT / "docs/guides/sbp_sat_boundary_coexistence_rfc.md"
FULL_SPEC = ROOT / "docs/guides/sbp_sat_zslab_phase1_full_spec.md"


def _text(path: Path) -> str:
    return path.read_text()


def test_boundary_coexistence_rfc_has_required_sections():
    text = _text(RFC)

    for heading in (
        "# SBP-SAT boundary coexistence RFC",
        "## Status",
        "## Current baseline",
        "## Boundary coexistence classes",
        "## Coexistence invariants",
        "## Per-face CPML / UPML layer and padding contract",
        "## Periodic-axis coexistence contract",
        "## PMC coexistence contract",
        "## Open-boundary benchmark definitions",
        "## Unsupported-combination hard-fail matrix",
        "## Implementation plan",
        "## Implementation gate",
    ):
        assert heading in text


def test_boundary_coexistence_rfc_locks_core_invariants_and_padding_contract():
    text = _text(RFC)

    for token in (
        "Single canonical boundary source of truth",
        "Single absorber family per simulation",
        "Reflector / periodic faces consume zero absorber layers",
        "Refinement interface is distinct from outer boundary policy",
        "L_f = Boundary.resolved_lo_thickness(cpml_layers)",
        "L_f = Boundary.resolved_hi_thickness(cpml_layers)",
        "If face `f` is `pec`, `pmc`, or `periodic`, then `L_f = 0`",
        "pad_f = L_f * dx_c",
        "box_lo_a >= pad_lo_a + g * dx_c",
        "box_hi_a <= domain_a - pad_hi_a - g * dx_c",
        "Using `max(L_lo, L_hi)` as a single symmetric shortcut is not acceptable",
    ):
        assert token in text


def test_boundary_coexistence_rfc_locks_boundary_classes_and_hard_fail_matrix():
    text = _text(RFC)

    for row_token in (
        "all-PEC",
        "reflector-only with PMC faces",
        "periodic axes with reflector faces",
        "all-absorbing CPML outer boundary",
        "all-absorbing UPML outer boundary",
        "mixed absorber + reflector faces",
        "per-face absorber thickness overrides",
        "mixed absorber families (`cpml` and `upml`)",
        "scalar `boundary='cpml'` + subgrid",
        "scalar `boundary='upml'` + subgrid",
        "any PMC face in `BoundarySpec`",
        "any periodic axis in `BoundarySpec`",
        "per-face absorber thickness override on any absorbing face",
        "`set_periodic_axes(...)` after refinement",
        "mixed absorber families (`cpml` + `upml`)",
    ):
        assert row_token in text


def test_boundary_coexistence_rfc_locks_open_boundary_benchmark_families():
    text = _text(RFC)

    for token in (
        "### OB-1: normal-incidence slab benchmark",
        "vacuum -> dielectric slab -> vacuum",
        "incident, reflected, and transmitted measurements",
        "analytic transfer matrix plus uniform-fine numerical reference",
        "### OB-2: oblique face-orientation benchmark",
        "separate x-face and y-face cases, not only z-face regression",
        "### OB-3: periodic unit-cell benchmark",
        "periodic on one or two transverse axes, absorbing on the remaining",
        "These benchmarks are implementation gates, not public claims",
    ):
        assert token in text


def test_boundary_coexistence_rfc_locks_periodic_axis_rules():
    text = _text(RFC)

    for token in (
        "Periodic coexistence is defined only for full-axis periodicity on an axis",
        "Boundary(lo='periodic', hi='periodic')",
        "the periodic axis contributes zero absorber padding",
        "preserve phase consistency across the wrapped",
        "domain direction",
        "unit-cell benchmarks include at least one periodic axis and at least one",
        "Until then, any periodic axis with subgridding remains a hard-fail condition",
    ):
        assert token in text


def test_boundary_coexistence_rfc_locks_pmc_rules():
    text = _text(RFC)

    for token in (
        "PMC coexistence is defined as a reflector coexistence problem",
        "face-orientation contract for tangential `H` vs tangential `E` treatment",
        "not apply contradictory updates in the same half step",
        "reflector benchmarks separate from the open-boundary benchmark family",
        "Until then, any PMC face with subgridding remains a hard-fail condition",
    ):
        assert token in text


def test_boundary_coexistence_rfc_blocks_runtime_enablement():
    text = _text(RFC)

    for token in (
        "It does **not** widen shipped runtime support",
        "must keep hard-failing every non-PEC",
        "boundary combination until the gates in this RFC are satisfied",
        "Milestone 6 completes when this RFC exists and is regression-locked",
        "It does **not** mean non-PEC boundary coexistence is implemented",
        "explicit `BoundarySpec` coexistence tests exist",
        "the support matrix is updated from `all_pec_only` only after",
    ):
        assert token in text


def test_full_spec_references_milestone6_artifact_and_contract_test():
    text = _text(FULL_SPEC)

    assert "docs/guides/sbp_sat_boundary_coexistence_rfc.md" in text
    assert "tests/test_sbp_sat_boundary_coexistence_spec_contract.py" in text
