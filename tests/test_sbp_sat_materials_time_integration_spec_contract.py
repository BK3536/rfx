"""Regression locks for the Milestone 8 materials/time RFC."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RFC = ROOT / "docs/guides/sbp_sat_materials_time_integration_rfc.md"
FULL_SPEC = ROOT / "docs/guides/sbp_sat_zslab_phase1_full_spec.md"


def _text(path: Path) -> str:
    return path.read_text()


def test_materials_time_rfc_has_required_sections():
    text = _text(RFC)

    for heading in (
        "# SBP-SAT materials, dispersion, and time-integration RFC",
        "## Status",
        "## Current baseline",
        "## Material classes",
        "## Bulk-update vs interface-update contract",
        "## SAT penalty policy",
        "## CFL and sub-stepping decision record",
        "## Discrete energy estimate contract",
        "## Benchmark ladder",
        "## Implementation plan",
        "## Implementation gate",
    ):
        assert heading in text


def test_materials_time_rfc_locks_current_baseline():
    text = _text(RFC)

    for token in (
        "MaterialArrays",
        "eps_r`, `sigma`, and `mu_r`",
        "current coefficients depend only on `ratio` and `tau`",
        "dt = phase1_3d_dt(dx_f)",
        "no sub-stepping machinery",
        "no subgridded Debye/Lorentz/Drude",
        "anisotropic, or nonlinear state evolution",
        "weights fields by",
        "vacuum `EPS_0` / `MU_0`",
    ):
        assert token in text


def test_materials_time_rfc_locks_material_classes_and_unaware_penalty():
    text = _text(RFC)

    for token in (
        "vacuum / isotropic linear dielectric",
        "current internal baseline",
        "isotropic conductive material",
        "blocked for support claims until benchmarked",
        "isotropic magnetic material",
        "Debye dispersion",
        "Lorentz / Drude-style dispersion",
        "anisotropic permittivity",
        "nonlinear / Kerr material",
        "reduced-dt or sub-stepped coarse/fine integration",
        "alpha_c = tau / (ratio + 1)",
        "alpha_f = tau * ratio / (ratio + 1)",
        "material-unaware",
        "no dependence on `eps_r`",
        "no dependence on `sigma`",
        "no dependence on `mu_r`",
        "no dependence on dispersion pole state",
        "no dependence on anisotropic tensor orientation",
        "electric impedance, magnetic impedance, wave",
        "speed, energy norm, or another derived quantity",
        "whether the coarse and fine sides use symmetric or asymmetric scaling",
        "whether electric and magnetic SAT penalties share the same scaling",
        "whether lossy or dispersive state enters the penalty directly",
        "whether the scaling is evaluated cellwise, facewise averaged, or benchmark-fit",
    ):
        assert token in text


def test_materials_time_rfc_locks_substepping_and_energy_contract():
    text = _text(RFC)

    for token in (
        "one shared timestep",
        "coarse and fine updates both use that same `dt`",
        "which grid advances multiple times per outer step",
        "where SAT coupling is applied in the multi-rate schedule",
        "source waveforms and observables are time-aligned",
        "overlap-counted proxy metric",
        "no material-dependent electric/magnetic energy density is included",
        "no dispersive stored-energy term is included",
        "no nonlinear stored-energy term is included",
        "eps_r`-scaled electric energy",
        "mu_r`-scaled magnetic energy",
        "conductive dissipation accounting",
        "Debye / Lorentz / Drude auxiliary stored energy",
        "sub-step/interface exchange work terms",
    ):
        assert token in text


def test_materials_time_rfc_locks_benchmark_ladder_and_gate():
    text = _text(RFC)

    for token in (
        "### MT-1: vacuum / isotropic dielectric interface baseline",
        "### MT-2: conductive-material bulk-vs-interface isolation",
        "### MT-3: magnetic-material interface isolation",
        "### MT-4: dispersive-material gate",
        "### MT-5: anisotropic / nonlinear gate",
        "### MT-6: time-integration gate",
        "material-scaled SAT or sub-stepping is implemented",
        "the accepted material/time class has an explicit interface policy",
        "the energy metric for that accepted class is stated explicitly",
        "the relevant benchmark family above is implemented and passing",
    ):
        assert token in text


def test_full_spec_references_milestone8_artifact_and_contract_test():
    text = _text(FULL_SPEC)

    assert "docs/guides/sbp_sat_materials_time_integration_rfc.md" in text
    assert "tests/test_sbp_sat_materials_time_integration_spec_contract.py" in text
