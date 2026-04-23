"""Regression locks for the public SBP-SAT documentation boundary."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
PUBLIC_INDEX = ROOT / "docs/public/index.mdx"
GUIDE_INDEX = ROOT / "docs/public/guide/index.md"
SUBGRID_GUIDE = ROOT / "docs/public/guide/subgridding.mdx"
SUPPORT_BOUNDARIES = ROOT / "docs/public/api/support-boundaries.mdx"


def _text(path: Path) -> str:
    return path.read_text()


def test_readme_marks_subgridding_as_non_reference_lane():
    text = _text(README)
    assert "SBP-SAT subgridding" in text
    assert "docs/guides/support_matrix.md" in text
    assert "SBP-SAT Subgridding guide" in text
    assert "experimental all-PEC z-slab research lane; proxy benchmark only" in text


def test_public_subgridding_guide_is_scoped_and_experimental():
    text = _text(SUBGRID_GUIDE)
    assert "Experimental research lane" in text
    assert "all-PEC, full-span x/y, z-slab" in text
    assert "soft point sources" in text
    assert "point probes" in text
    assert "proxy benchmark" in text
    assert "If you need a practical thin-substrate workflow today" in text
    assert "local mesh refinement with JIT performance" not in text


def test_public_indexes_do_not_overclaim_subgridding():
    for path in (PUBLIC_INDEX, GUIDE_INDEX):
        text = _text(path)
        assert "experimental all-PEC z-slab" in text
        assert "proxy benchmark only" in text
        assert "local mesh refinement with JIT performance" not in text


def test_support_boundaries_publish_experimental_sbp_sat_subset():
    text = _text(SUPPORT_BOUNDARIES)
    assert "proxy numerical-equivalence evidence only" in text
    assert "all-PEC boundary only" in text
    assert "one full-span x/y refined z slab only" in text
    assert "CPML/UPML + SBP-SAT subgridding" in text
    assert "DFT planes / flux monitors / NTFF + SBP-SAT subgridding" in text
