"""Phase XVI native Strategy B full RF-port S-matrix tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.phase16_strategy_b_native_smatrix_validation import run_validation
from rfx import GaussianPulse, Simulation
from rfx.probes.probes import extract_s_matrix
from rfx.sources.sources import LumpedPort


def _make_two_port_sim(
    *,
    boundary: str = "pec",
    passive_first: bool = False,
    explicit_direction: bool = False,
) -> tuple[Simulation, GaussianPulse, list[tuple[float, float, float]]]:
    pulse = GaussianPulse(f0=3e9, bandwidth=0.8)
    sim = Simulation(
        freq_max=5e9,
        domain=(0.010, 0.009, 0.009),
        dx=0.001,
        boundary=boundary,
        cpml_layers=0 if boundary == "pec" else 2,
    )
    active = (0.003, 0.004, 0.004)
    passive = (0.004, 0.004, 0.004)
    direction = "+x" if explicit_direction else None

    def add_active():
        sim.add_port(
            active,
            "ez",
            impedance=50.0,
            waveform=pulse,
            direction=direction,
        )

    def add_passive():
        sim.add_port(
            passive,
            "ez",
            impedance=50.0,
            excite=False,
            direction=direction,
        )

    if passive_first:
        add_passive()
        add_active()
        ordered = [passive, active]
    else:
        add_active()
        add_passive()
        ordered = [active, passive]
    sim.add_probe((0.006, 0.004, 0.004), "ez")
    return sim, pulse, ordered


def _standard_reference(
    sim: Simulation,
    pulse: GaussianPulse,
    ordered_positions: list[tuple[float, float, float]],
    freqs: jnp.ndarray,
    *,
    n_steps: int,
):
    grid = sim._build_grid()
    base_materials = sim._assemble_materials(grid)[0]
    ports = [
        LumpedPort(position, "ez", 50.0, pulse)
        for position in ordered_positions
    ]
    return extract_s_matrix(
        grid,
        base_materials,
        ports,
        freqs,
        n_steps=n_steps,
        boundary=sim._boundary,
    )


@pytest.mark.parametrize("boundary", ["pec", "cpml"])
def test_phase16_two_port_native_smatrix_matches_standard_reference(boundary):
    sim, pulse, ordered = _make_two_port_sim(boundary=boundary)
    freqs = jnp.array([2.5e9, 3.0e9, 3.5e9], dtype=jnp.float32)
    n_steps = 64
    inputs = sim.build_hybrid_phase1_inputs(n_steps=n_steps, s_param_freqs=freqs)
    report = sim.inspect_hybrid_strategy_b_phase6_from_inputs(inputs, checkpoint_every=16)
    assert report.supported, report.reason_text

    native = sim.forward_hybrid_phase1_from_inputs(inputs, strategy="b", checkpoint_every=16)
    reference = _standard_reference(sim, pulse, ordered, freqs, n_steps=n_steps)

    native_s = np.asarray(native.s_params)
    reference_s = np.asarray(reference)
    assert native_s.shape == (2, 2, len(freqs))
    np.testing.assert_allclose(np.asarray(native.freqs), np.asarray(freqs), rtol=0.0, atol=0.0)
    assert np.isfinite(native_s).all()

    abs_err = np.abs(native_s - reference_s)
    trusted = np.abs(reference_s) > 1e-4
    assert float(np.max(abs_err)) <= 5e-3
    assert float(np.max(abs_err[trusted] / np.abs(reference_s[trusted]))) <= 5e-2
    assert max(np.max(np.abs(native_s[1, 0, :])), np.max(np.abs(native_s[0, 1, :]))) > 1e-4

    reciprocity = np.mean(np.abs(native_s[0, 1, :] - native_s[1, 0, :])) / (
        np.mean((np.abs(native_s[0, 1, :]) + np.abs(native_s[1, 0, :])) / 2.0) + 1e-30
    )
    assert float(reciprocity) <= 5e-2
    column_power = np.sum(np.abs(native_s) ** 2, axis=0)
    assert float(np.max(column_power)) <= 1.5


def test_phase16_native_smatrix_does_not_use_standard_extractors(monkeypatch):
    sim, _, _ = _make_two_port_sim()
    freqs = jnp.array([3.0e9], dtype=jnp.float32)

    def _fail_standard(*_args, **_kwargs):
        raise AssertionError("native Strategy B S-matrix must not use standard S-parameter extraction")

    import rfx.probes.probes as probes_mod

    monkeypatch.setattr(probes_mod, "extract_s_matrix", _fail_standard)
    monkeypatch.setattr(probes_mod, "extract_s_matrix_wire", _fail_standard)
    monkeypatch.setattr(Simulation, "run", _fail_standard)

    inputs = sim.build_hybrid_phase1_inputs(n_steps=24, s_param_freqs=freqs)
    result = sim.forward_hybrid_phase1_from_inputs(inputs, strategy="b", checkpoint_every=8)

    assert result.s_params is not None
    assert result.s_params.shape == (2, 2, 1)
    assert np.isfinite(np.asarray(result.s_params)).all()


def test_phase16_passive_first_active_second_preserves_public_port_order():
    sim, pulse, ordered = _make_two_port_sim(passive_first=True)
    freqs = jnp.array([3.0e9], dtype=jnp.float32)
    n_steps = 64
    inputs = sim.build_hybrid_phase1_inputs(n_steps=n_steps, s_param_freqs=freqs)
    request = inputs.s_param_request
    assert request is not None
    assert [tuple(port.cell) for port in request.ports] == [
        inputs.grid.position_to_index(position) for position in ordered
    ]
    assert [port.excite_in_main_run for port in request.ports] == [False, True]

    native = sim.forward_hybrid_phase1_from_inputs(inputs, strategy="b", checkpoint_every=16)
    reference = _standard_reference(sim, pulse, ordered, freqs, n_steps=n_steps)

    np.testing.assert_allclose(
        np.asarray(native.s_params),
        np.asarray(reference),
        rtol=5e-2,
        atol=5e-3,
    )


def test_phase16_explicit_lumped_port_direction_fails_closed():
    sim, _, _ = _make_two_port_sim(explicit_direction=True)
    inputs = sim.build_hybrid_phase1_inputs(n_steps=16, s_param_freqs=jnp.array([3.0e9]))

    report = sim.inspect_hybrid_strategy_b_phase6_from_inputs(inputs, checkpoint_every=8)

    assert not report.supported
    assert "does not support explicit port direction yet" in report.reason_text
    with pytest.raises(ValueError, match="explicit port direction"):
        sim.forward_hybrid_phase1_from_inputs(inputs, strategy="b", checkpoint_every=8)


def test_phase16_sparams_remain_stop_gradient_sidecar_for_two_port():
    sim, _, _ = _make_two_port_sim()
    freqs = jnp.array([3.0e9], dtype=jnp.float32)
    no_sp_inputs = sim.build_hybrid_phase1_inputs(n_steps=12)
    sp_inputs = sim.build_hybrid_phase1_inputs(n_steps=12, s_param_freqs=freqs)
    grid = sp_inputs.grid
    assert grid is not None
    assert sp_inputs.materials is not None

    def eps_with_alpha(inputs, alpha):
        i, j, k = grid.position_to_index((0.006, 0.006, 0.006))
        return inputs.materials.eps_r.at[i, j, k].add(alpha)

    def time_loss(inputs, alpha):
        result = sim.forward_hybrid_phase1_from_inputs(
            inputs,
            eps_override=eps_with_alpha(inputs, alpha),
            strategy="b",
            checkpoint_every=6,
        )
        return jnp.sum(result.time_series**2)

    alpha0 = jnp.float32(0.05)
    grad_without = jax.grad(lambda a: time_loss(no_sp_inputs, a))(alpha0)
    grad_with = jax.grad(lambda a: time_loss(sp_inputs, a))(alpha0)
    np.testing.assert_allclose(np.asarray(grad_with), np.asarray(grad_without), rtol=1e-5, atol=1e-8)

    def sidecar_loss(alpha):
        result = sim.forward_hybrid_phase1_from_inputs(
            sp_inputs,
            eps_override=eps_with_alpha(sp_inputs, alpha),
            strategy="b",
            checkpoint_every=6,
        )
        return jnp.real(jnp.sum(jnp.abs(result.s_params) ** 2))

    sidecar_grad = jax.grad(sidecar_loss)(alpha0)
    np.testing.assert_allclose(np.asarray(sidecar_grad), np.asarray(0.0, dtype=np.float32), atol=0.0, rtol=0.0)


def test_phase16_validation_artifact_gates_pass():
    artifact = run_validation(n_steps=64)

    assert artifact["overall_status"] == "phase16_native_full_smatrix_validated"
    assert all(artifact["gates"].values())
    assert artifact["strategy_b_seam"]["native_s_params_supported"] is True
    assert artifact["strategy_b_seam"]["observable_source"] == "strategy_b_native_sparams"
    assert artifact["strategy_b_seam"]["phase1_forward_result_s_params_shape"] == [2, 2, 3]
    assert artifact["unsupported_scope"]["fail_closed_valid"] is True
