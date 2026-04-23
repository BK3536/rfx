"""Phase III Strategy B source/probe prototype tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from rfx import GaussianPulse, Simulation


def _make_source_probe_sim(*, boundary: str = "pec") -> Simulation:
    sim = Simulation(
        freq_max=5e9,
        domain=(0.015, 0.015, 0.015),
        boundary=boundary,
    )
    sim.add_source(
        (0.005, 0.0075, 0.0075),
        "ez",
        waveform=GaussianPulse(f0=3e9, bandwidth=0.5),
    )
    sim.add_probe((0.010, 0.0075, 0.0075), "ez")
    return sim


def _single_cell_eps(grid, base_eps: jnp.ndarray, alpha: jnp.ndarray) -> jnp.ndarray:
    i, j, k = grid.position_to_index((0.0075, 0.0075, 0.0075))
    return base_eps.at[i, j, k].add(alpha)


def test_phase3_strategy_b_support_inspection_accepts_pec_source_probe_only():
    sim = _make_source_probe_sim()

    report = sim.inspect_hybrid_strategy_b_phase3(n_steps=8, checkpoint_every=3)

    assert report.supported
    assert report.source_count == 1
    assert report.probe_count == 1
    assert report.boundary == "pec"
    assert report.inventory is not None


def test_phase3_strategy_b_support_inspection_accepts_cpml_source_probe():
    cpml = _make_source_probe_sim(boundary="cpml")
    cpml_report = cpml.inspect_hybrid_strategy_b_phase3(n_steps=8, checkpoint_every=3)
    assert cpml_report.supported
    assert cpml_report.boundary == "cpml"


def test_phase3_strategy_b_support_inspection_rejects_missing_checkpoint():
    pec = _make_source_probe_sim()
    checkpoint_report = pec.inspect_hybrid_strategy_b_phase3(n_steps=8)
    assert not checkpoint_report.supported
    assert "checkpoint_every is required" in checkpoint_report.reason_text


def test_phase3_strategy_b_support_inspection_rejects_ntff_and_port_workflows():
    ntff = _make_source_probe_sim()
    ntff.add_ntff_box(
        corner_lo=(0.003, 0.003, 0.003),
        corner_hi=(0.012, 0.012, 0.012),
        n_freqs=4,
    )
    ntff_report = ntff.inspect_hybrid_strategy_b_phase3(n_steps=8, checkpoint_every=3)
    assert not ntff_report.supported
    assert "does not support NTFF" in ntff_report.reason_text

    port = Simulation(freq_max=5e9, domain=(0.015, 0.015, 0.015), boundary="pec")
    port.add_port(
        (0.005, 0.0075, 0.0075),
        "ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=3e9, bandwidth=0.5),
    )
    port.add_probe((0.010, 0.0075, 0.0075), "ez")
    port_report = port.inspect_hybrid_strategy_b_phase3(n_steps=8, checkpoint_every=3)
    assert not port_report.supported
    assert "supports only add_source()/probe workflows" in port_report.reason_text


def test_phase3_strategy_b_forward_matches_strategy_a_source_probe():
    sim = _make_source_probe_sim()

    strategy_a = sim.forward_hybrid_phase1(n_steps=8, fallback="raise")
    strategy_b = sim.forward_hybrid_phase1(
        n_steps=8,
        fallback="raise",
        strategy="b",
        checkpoint_every=3,
    )

    np.testing.assert_allclose(
        np.asarray(strategy_b.time_series),
        np.asarray(strategy_a.time_series),
        rtol=1e-6,
        atol=1e-12,
    )


def test_phase3_strategy_b_forward_matches_strategy_a_cpml_source_probe():
    sim = _make_source_probe_sim(boundary="cpml")

    strategy_a = sim.forward_hybrid_phase1(n_steps=8, fallback="raise")
    strategy_b = sim.forward_hybrid_phase1(
        n_steps=8,
        fallback="raise",
        strategy="b",
        checkpoint_every=3,
    )

    np.testing.assert_allclose(
        np.asarray(strategy_b.time_series),
        np.asarray(strategy_a.time_series),
        rtol=1e-6,
        atol=1e-12,
    )


def test_phase3_strategy_b_gradient_matches_pure_ad_source_probe():
    sim = _make_source_probe_sim()
    grid = sim._build_grid()
    materials, *_ = sim._assemble_materials(grid)

    def pure_loss(alpha):
        eps = _single_cell_eps(grid, materials.eps_r, alpha)
        result = sim.forward(eps_override=eps, n_steps=8, checkpoint=True)
        return jnp.sum(result.time_series ** 2)

    def strategy_b_loss(alpha):
        eps = _single_cell_eps(grid, materials.eps_r, alpha)
        result = sim.forward_hybrid_phase1(
            eps_override=eps,
            n_steps=8,
            fallback="raise",
            strategy="b",
            checkpoint_every=3,
        )
        return jnp.sum(result.time_series ** 2)

    alpha0 = jnp.float32(0.1)
    grad_pure = jax.grad(pure_loss)(alpha0)
    grad_strategy_b = jax.grad(strategy_b_loss)(alpha0)
    rel_err = float(
        jnp.abs(grad_strategy_b - grad_pure)
        / jnp.maximum(jnp.abs(grad_pure), 1e-12)
    )

    assert np.isfinite(float(grad_pure))
    assert np.isfinite(float(grad_strategy_b))
    assert rel_err <= 1e-4, (
        f"Strategy B gradient drifted from pure AD: pure={float(grad_pure):.6e}, "
        f"strategy_b={float(grad_strategy_b):.6e}, rel_err={rel_err:.6e}"
    )


def test_phase3_strategy_b_gradient_matches_pure_ad_cpml_source_probe():
    sim = _make_source_probe_sim(boundary="cpml")
    grid = sim._build_grid()
    materials, *_ = sim._assemble_materials(grid)

    def pure_loss(alpha):
        eps = _single_cell_eps(grid, materials.eps_r, alpha)
        result = sim.forward(eps_override=eps, n_steps=8, checkpoint=True)
        return jnp.sum(result.time_series ** 2)

    def strategy_b_loss(alpha):
        eps = _single_cell_eps(grid, materials.eps_r, alpha)
        result = sim.forward_hybrid_phase1(
            eps_override=eps,
            n_steps=8,
            fallback="raise",
            strategy="b",
            checkpoint_every=3,
        )
        return jnp.sum(result.time_series ** 2)

    alpha0 = jnp.float32(0.1)
    grad_pure = jax.grad(pure_loss)(alpha0)
    grad_strategy_b = jax.grad(strategy_b_loss)(alpha0)
    rel_err = float(
        jnp.abs(grad_strategy_b - grad_pure)
        / jnp.maximum(jnp.abs(grad_pure), 1e-12)
    )

    assert np.isfinite(float(grad_pure))
    assert np.isfinite(float(grad_strategy_b))
    assert rel_err <= 1e-4, (
        f"Strategy B CPML gradient drifted from pure AD: pure={float(grad_pure):.6e}, "
        f"strategy_b={float(grad_strategy_b):.6e}, rel_err={rel_err:.6e}"
    )
