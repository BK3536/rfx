"""T7-E Phase 1: PMC spec acceptance + Simulation-level runtime guard.

BoundarySpec accepts 'pmc' on any face (so users can construct and
inspect specs), but Simulation construction rejects the spec with a
clear NotImplementedError. The Phase 2 follow-up (v1.7.1 / v1.8) wires
apply_pmc_faces into the Yee update path with cross-validation tests.
"""

from __future__ import annotations

import pytest

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec


# ---------------------------------------------------------------------------
# Spec-only construction still works (users can build and inspect)
# ---------------------------------------------------------------------------

class TestSpecOnly:
    def test_spec_accepts_pmc_on_single_face(self):
        spec = BoundarySpec(
            x="cpml", y="cpml",
            z=Boundary(lo="pmc", hi="cpml"),
        )
        assert spec.pmc_faces() == {"z_lo"}

    def test_spec_accepts_pmc_on_both_faces_same_axis(self):
        spec = BoundarySpec(x="cpml", y="cpml", z="pmc")
        assert spec.pmc_faces() == {"z_lo", "z_hi"}

    def test_spec_accepts_pmc_mixed_with_pec(self):
        spec = BoundarySpec(
            x="cpml", y="cpml",
            z=Boundary(lo="pmc", hi="pec"),
        )
        assert spec.pmc_faces() == {"z_lo"}
        assert spec.pec_faces() == {"z_hi"}


# ---------------------------------------------------------------------------
# Simulation construction must raise NotImplementedError
# ---------------------------------------------------------------------------

class TestSimulationRaisesOnPMC:
    def test_single_pmc_face_raises(self):
        spec = BoundarySpec(
            x="cpml", y="cpml",
            z=Boundary(lo="pmc", hi="cpml"),
        )
        with pytest.raises(NotImplementedError, match="PMC runtime not yet wired"):
            Simulation(
                freq_max=10e9, domain=(0.01, 0.01, 0.005), dx=0.5e-3,
                boundary=spec,
            )

    def test_both_pmc_faces_raise(self):
        spec = BoundarySpec(x="cpml", y="cpml", z="pmc")
        with pytest.raises(NotImplementedError, match="PMC runtime not yet wired"):
            Simulation(
                freq_max=10e9, domain=(0.01, 0.01, 0.005), dx=0.5e-3,
                boundary=spec,
            )

    def test_pmc_plus_pec_raises(self):
        spec = BoundarySpec(
            x="cpml", y="cpml",
            z=Boundary(lo="pmc", hi="pec"),
        )
        with pytest.raises(NotImplementedError, match="PMC runtime not yet wired"):
            Simulation(
                freq_max=10e9, domain=(0.01, 0.01, 0.005), dx=0.5e-3,
                boundary=spec,
            )

    def test_error_message_is_actionable(self):
        spec = BoundarySpec(x="cpml", y="cpml", z="pmc")
        with pytest.raises(NotImplementedError) as excinfo:
            Simulation(
                freq_max=10e9, domain=(0.01, 0.01, 0.005), dx=0.5e-3,
                boundary=spec,
            )
        msg = str(excinfo.value)
        assert "v1.8" in msg or "v1.7.1" in msg  # names the follow-up
        assert "pec" in msg.lower()  # suggests a workaround
