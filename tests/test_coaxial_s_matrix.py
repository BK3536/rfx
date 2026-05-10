"""Tests for the experimental ``compute_coaxial_s_matrix`` API.

These tests exercise the M67 distributed TEM plane-source scaffold once
promoted into ``Simulation.compute_coaxial_s_matrix``. The scaffold is
documented as **experimental**: per-frequency |S11| numerics are not yet
calibrated against analytic PEC-cavity reflection or external openEMS
fixtures. The tests focus on:

- API/schema contract (shape, types, status field)
- Family-routing rejection (mixing port families)
- Closed-form ``Z_TEM`` consistency between the helper and the result
- Plane-source amplitude scale invariance (same S-matrix at two field_scale
  values)
- Validation contract on the returned ``CoaxialSMatrixResult``

Calibrated PEC-short / matched-load / open / load gates against an external
solver are tracked separately as the next promotion step (M71-> calibrated
fixture). They are intentionally **not** wired into pytest because they
would otherwise mark the experimental API as physics-validated by
test-pass-alone, which the physics-validation evidence rule forbids.
"""

from __future__ import annotations

import numpy as np
import pytest

from rfx import (
    CoaxialSMatrixResult,
    Simulation,
    coaxial_tem_characteristic_impedance,
    validate_port_smatrix,
)
from rfx.sources.coaxial_port import (
    PTFE_EPS_R,
    SMA_OUTER_RADIUS,
    SMA_PIN_RADIUS,
)


def _make_one_port_sim() -> Simulation:
    sim = Simulation(
        freq_max=10.0e9,
        domain=(0.020, 0.020, 0.020),
        boundary="pec",
    )
    sim.add_coaxial_port((0.010, 0.010, 0.015), face="top")
    return sim


def test_compute_coaxial_s_matrix_returns_canonical_result_shape():
    sim = _make_one_port_sim()
    res = sim.compute_coaxial_s_matrix(n_steps=200, n_freqs=3)
    assert isinstance(res, CoaxialSMatrixResult)
    assert res.s_params.shape == (1, 1, 3)
    assert res.freqs.shape == (3,)
    assert res.port_names == ("coax_0",)
    assert res.port_faces == ("top",)
    assert res.reference_planes.shape == (1,)
    assert res.z_tem_ohm.shape == (1, 3)
    assert res.voltages.shape == (1, 1, 3)
    assert res.currents.shape == (1, 1, 3)
    assert res.status in {"passed", "degraded"}


def test_z_tem_matches_closed_form_helper():
    sim = _make_one_port_sim()
    res = sim.compute_coaxial_s_matrix(n_steps=200, n_freqs=3)
    z_expected = coaxial_tem_characteristic_impedance(
        SMA_PIN_RADIUS, SMA_OUTER_RADIUS, PTFE_EPS_R
    )
    np.testing.assert_allclose(
        res.z_tem_ohm,
        np.full_like(res.z_tem_ohm, z_expected, dtype=np.complex128),
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_field_scale_does_not_change_s_matrix_amplitudes():
    sim = _make_one_port_sim()
    res_lo = sim.compute_coaxial_s_matrix(n_steps=200, n_freqs=3, field_scale=1.0e3)
    sim2 = _make_one_port_sim()
    res_hi = sim2.compute_coaxial_s_matrix(n_steps=200, n_freqs=3, field_scale=1.0e5)
    # The plane-source amplitude is a pure linear scale on both V and I, so
    # the power-wave ratio b/a is invariant under field_scale up to the
    # float32 DFT-accumulator round-off floor.
    np.testing.assert_allclose(
        res_lo.s_params, res_hi.s_params, rtol=1.0e-4, atol=1.0e-7
    )


def test_returns_finite_complex_smatrix_with_reasonable_magnitude():
    sim = _make_one_port_sim()
    res = sim.compute_coaxial_s_matrix(n_steps=200, n_freqs=3)
    assert np.all(np.isfinite(res.s_params))
    # PEC cavity feed cannot leak more than the source amplitude. We do not
    # require |S11| ≈ 1 here because the plane source is forward-only and
    # the M67 scaffold is documented as not yet calibrated.
    assert float(np.max(np.abs(res.s_params))) < 5.0


def test_validation_helper_normalizes_coaxial_result():
    sim = _make_one_port_sim()
    res = sim.compute_coaxial_s_matrix(n_steps=200, n_freqs=3)
    report = validate_port_smatrix(
        res,
        port_names=res.port_names,
        check_passivity=False,
        check_reciprocity=False,
    )
    assert report.n_ports == 1
    assert report.n_freqs == 3
    assert "max_column_power" not in report.metrics or np.isfinite(
        report.metrics.get("max_column_power", 0.0)
    )


def test_compute_coaxial_rejects_no_coaxial_ports():
    sim = Simulation(
        freq_max=10.0e9,
        domain=(0.020, 0.020, 0.020),
        boundary="pec",
    )
    with pytest.raises(ValueError, match="No coaxial ports"):
        sim.compute_coaxial_s_matrix(n_steps=200, n_freqs=3)


def test_compute_coaxial_rejects_mixed_port_families():
    sim = Simulation(
        freq_max=10.0e9,
        domain=(0.020, 0.020, 0.020),
        boundary="pec",
    )
    sim.add_coaxial_port((0.010, 0.010, 0.015), face="top")
    sim.add_msl_port(
        (0.005, 0.010, 0.001),
        width=0.6e-3,
        height=0.254e-3,
        direction="+x",
    )
    with pytest.raises(NotImplementedError, match="add_coaxial_port"):
        sim.compute_coaxial_s_matrix(n_steps=200, n_freqs=3)


def test_preflight_sparameters_routes_coaxial_calculator():
    sim = _make_one_port_sim()
    issues = sim.preflight_sparameters(calculator="coaxial")
    assert issues == []
    issues_alias = sim.preflight_sparameters(calculator="compute_coaxial_s_matrix")
    assert issues_alias == []


def test_preflight_sparameters_rejects_coaxial_when_no_ports():
    sim = Simulation(
        freq_max=10.0e9,
        domain=(0.020, 0.020, 0.020),
        boundary="pec",
    )
    issues = sim.preflight_sparameters(calculator="coaxial")
    assert any("No coaxial ports" in issue for issue in issues)


# ---------------------------------------------------------------------------
# Calibration target tests (xfail)
# ---------------------------------------------------------------------------
#
# These tests state the expected calibrated behaviour the experimental
# compute_coaxial_s_matrix API should produce once the underlying source-
# plane bidirectional-injection limitation is resolved (TFSF-style one-side
# injection or explicit incident-wave subtraction). They are marked xfail
# while the M72 prototype-promoted source remains bidirectional. Removing
# the xfail marker is the gate for promoting the API beyond experimental.


def _make_pec_short_sim() -> Simulation:
    """Coaxial port whose centre pin reaches the cavity floor (PEC short).

    ``freq_max=20 GHz`` gives ``dx ≈ 0.75 mm`` so the SMA PTFE annulus
    (0.71 mm wide between the 0.635 mm pin and the 1.345 mm shell-inner
    edge) is resolved by ~1 cell; at the previous ``freq_max=10 GHz``
    setting the PTFE region was 0.47 cells wide and could not support a
    discrete coax TEM mode regardless of source quality (see the
    20260510_coaxial_tfsf_session_progress handover doc).
    """
    from rfx.sources.sources import GaussianPulse

    sim = Simulation(
        freq_max=20.0e9,
        domain=(0.020, 0.020, 0.020),
        boundary="pec",
    )
    sim.add_coaxial_port(
        position=(0.010, 0.010, 0.015),
        face="top",
        pin_length=15.0e-3,  # extends from gap (z=15mm) to cavity floor (z=0)
        waveform=GaussianPulse(f0=5.0e9, bandwidth=0.8),
    )
    return sim


def test_pec_short_yields_full_reflection_calibration_target():
    sim = _make_pec_short_sim()
    res = sim.compute_coaxial_s_matrix(n_steps=400, n_freqs=5)
    s11 = np.abs(res.s_params[0, 0, :])
    assert float(np.min(s11)) >= 0.9, (
        f"PEC-short |S11| should be near 1, got {s11.tolist()}"
    )


def test_pec_short_smoke_returns_finite_values():
    """Mechanical smoke check that the PEC-short geometry runs end-to-end.

    Pairs with the xfail calibration-target test above: while the API is
    experimental, this test simply asserts |S11| is finite, bounded, and
    nonzero so regressions on plumbing (DFT-plane scope, V/I extraction,
    power-wave decomposition) are still caught.
    """
    sim = _make_pec_short_sim()
    res = sim.compute_coaxial_s_matrix(n_steps=400, n_freqs=5)
    s11 = res.s_params[0, 0, :]
    assert np.all(np.isfinite(s11))
    mag = np.abs(s11)
    assert float(np.max(mag)) < 5.0
    assert float(np.max(mag)) > 0.0
