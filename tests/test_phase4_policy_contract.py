"""Phase IV Stage 1 public policy contract tests."""

from pathlib import Path

from rfx.optimize import optimize
from rfx.topology import topology_optimize


ROOT = Path(__file__).resolve().parents[1]


def _normalize(text: str) -> str:
    return " ".join(text.replace("**", "").split())


def test_phase4_public_optimizer_docstrings_describe_stage1_policy():
    optimize_doc = _normalize(optimize.__doc__ or "")
    topology_doc = _normalize(topology_optimize.__doc__ or "")

    for doc in (optimize_doc, topology_doc):
        assert "pure_ad" in doc
        assert "hybrid" in doc
        assert "auto" in doc
        assert "default behavior" in doc or "public default behavior" in doc
        assert "falls back to the current pure-AD path" in doc


def test_phase4_policy_guide_fences_strategy_b_out_of_stage1_matrix():
    text = _normalize((ROOT / "docs/guides/hybrid_optimizer_mode_policy.md").read_text())

    assert "Stage 1 public policy" in text
    assert "Strategy B is not part of the Stage 1 public optimizer policy matrix." in text
    assert "no default status" in text
    assert "no recommended status" in text
    assert "fallback to `pure_ad`" in text


def test_phase4_support_matrix_keeps_strategy_b_outside_stage1_table():
    text = _normalize((ROOT / "docs/guides/support_matrix.md").read_text())

    assert "| `pure_ad` | supported default |" in text
    assert "| `auto` | experimental-supported / bounded opt-in |" in text
    assert "| `hybrid` | experimental-supported / strict opt-in |" in text
    assert "| Strategy B |" not in text
    assert "Strategy B outside this public optimizer policy matrix" in text
