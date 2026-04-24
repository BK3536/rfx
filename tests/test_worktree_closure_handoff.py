"""Post-Phase-IV worktree closure handoff pins."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _normalize(text: str) -> str:
    return " ".join(text.replace("**", "").split())


def test_worktree_closure_handoff_summarizes_current_branch_state():
    text = _normalize((ROOT / "docs/guides/hybrid_adjoint_worktree_closure_handoff.md").read_text())

    assert "Phase II — Practical Workflow Coverage" in text
    assert "Phase III — Strategy B / Scale-up" in text
    assert "Phase IV — Productization / Default Policy" in text
    assert 'adjoint_mode="pure_ad"' in text
    assert 'adjoint_mode="auto"' in text
    assert 'adjoint_mode="hybrid"' in text
    assert "Strategy B remains outside the public optimizer policy matrix" in text
    assert "There is currently no defined Phase V in the roadmap." in text
    assert "external/pipeline-owned" in text
    assert "python scripts/check_public_docs_sync.py --format text" in text
    assert "gitops deploy snapshot has not yet been re-exported and rebuilt" in text
