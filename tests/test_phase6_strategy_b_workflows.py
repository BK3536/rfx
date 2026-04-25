"""Phase VI Strategy B practical workflow expansion tests."""

from __future__ import annotations

import importlib.util
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import GaussianPulse, Simulation
from rfx.optimize import DesignRegion, optimize
from rfx.topology import (
    TopologyDesignRegion,
    _inspect_topology_hybrid_support,
    topology_optimize,
)

pytestmark = pytest.mark.gpu


def _make_source_probe_sim(*, boundary: str = "pec") -> Simulation:
    sim = Simulation(freq_max=5e9, domain=(0.015, 0.015, 0.015), boundary=boundary)
    sim.add_source(
        (0.005, 0.0075, 0.0075),
        "ez",
        waveform=GaussianPulse(f0=3e9, bandwidth=0.5),
    )
    sim.add_probe((0.010, 0.0075, 0.0075), "ez")
    return sim


def _make_topology_case(*, boundary: str = "cpml") -> tuple[Simulation, TopologyDesignRegion]:
    sim = _make_source_probe_sim(boundary=boundary)
    sim.add_material("phase6_diel", eps_r=4.0, sigma=0.0)
    region = TopologyDesignRegion(
        corner_lo=(0.009, 0.003, 0.003),
        corner_hi=(0.012, 0.006, 0.006),
        material_bg="air",
        material_fg="phase6_diel",
        beta_projection=1.0,
    )
    return sim, region


def _topology_probe_energy_objective(result):
    return -jnp.sum(result.time_series ** 2)


def _make_passive_port_optimize_sim() -> Simulation:
    sim = Simulation(freq_max=5e9, domain=(0.015, 0.015, 0.015), boundary="pec")
    sim.add_port(
        (0.005, 0.0075, 0.0075),
        "ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=3e9, bandwidth=0.5),
    )
    sim.add_port((0.010, 0.0075, 0.0075), "ez", impedance=50.0, excite=False)
    sim.add_probe((0.012, 0.0075, 0.0075), "ez")
    return sim


def _make_port_design_region() -> DesignRegion:
    return DesignRegion(
        corner_lo=(0.009, 0.003, 0.003),
        corner_hi=(0.012, 0.006, 0.006),
        eps_range=(1.0, 4.4),
    )


def _make_overlapping_passive_port_design_region() -> DesignRegion:
    return DesignRegion(
        corner_lo=(0.009, 0.006, 0.006),
        corner_hi=(0.011, 0.009, 0.009),
        eps_range=(1.0, 4.4),
    )


def _probe_energy_objective(result):
    return -jnp.sum(result.time_series ** 2)


def _single_cell_eps(grid, base_eps: jnp.ndarray, alpha: jnp.ndarray) -> jnp.ndarray:
    i, j, k = grid.position_to_index((0.0075, 0.0075, 0.0075))
    return base_eps.at[i, j, k].add(alpha)


def test_phase6_strategy_b_inputs_helper_matches_source_probe_strategy_a():
    sim = _make_source_probe_sim(boundary="cpml")
    inputs = sim.build_hybrid_phase1_inputs(n_steps=8)

    strategy_a = sim.forward_hybrid_phase1_from_inputs(inputs)
    strategy_b = sim.forward_hybrid_phase1_from_inputs(
        inputs,
        strategy="b",
        checkpoint_every=3,
    )

    np.testing.assert_allclose(
        np.asarray(strategy_b.time_series),
        np.asarray(strategy_a.time_series),
        rtol=1e-6,
        atol=1e-12,
    )


def test_phase6_strategy_b_inputs_helper_rejects_missing_checkpoint():
    sim = _make_source_probe_sim()
    inputs = sim.build_hybrid_phase1_inputs(n_steps=8)

    report = sim.inspect_hybrid_strategy_b_phase6_from_inputs(inputs)

    assert not report.supported
    assert "checkpoint_every is required" in report.reason_text
    with pytest.raises(ValueError, match="checkpoint_every is required"):
        sim.forward_hybrid_phase1_from_inputs(inputs, strategy="b")


@pytest.mark.skipif(
    not importlib.util.find_spec("optax"),
    reason="optax not installed",
)
def test_phase6_cpml_topology_strategy_b_route_uses_inputs_seam(monkeypatch):
    sim, region = _make_topology_case(boundary="cpml")
    calls = {"inputs": 0}
    original = sim.forward_hybrid_phase1_from_inputs

    def _wrapped_inputs(inputs, *, eps_override=None, strategy="a", checkpoint_every=None):
        calls["inputs"] += 1
        assert strategy == "b"
        assert checkpoint_every == 3
        return original(
            inputs,
            eps_override=eps_override,
            strategy=strategy,
            checkpoint_every=checkpoint_every,
        )

    def _fail_pure_ad(*args, **kwargs):
        raise AssertionError("Strategy B topology unexpectedly used the pure-AD material path")

    monkeypatch.setattr(sim, "forward_hybrid_phase1_from_inputs", _wrapped_inputs)
    monkeypatch.setattr(sim, "_forward_from_materials", _fail_pure_ad)

    result = topology_optimize(
        sim,
        region,
        _topology_probe_energy_objective,
        n_iterations=1,
        learning_rate=0.05,
        n_steps=8,
        beta_schedule=[(0, 1.0)],
        verbose=False,
        adjoint_mode="hybrid",
        strategy="b",
        checkpoint_every=3,
    )

    assert len(result.history) == 1
    assert calls["inputs"] > 0


@pytest.mark.skipif(
    not importlib.util.find_spec("optax"),
    reason="optax not installed",
)
def test_phase6_cpml_topology_strategy_b_matches_pure_ad_one_step():
    sim, region = _make_topology_case(boundary="cpml")

    pure = topology_optimize(
        sim,
        region,
        _topology_probe_energy_objective,
        n_iterations=1,
        learning_rate=0.05,
        n_steps=8,
        beta_schedule=[(0, 1.0)],
        verbose=False,
        adjoint_mode="pure_ad",
    )
    strategy_b = topology_optimize(
        sim,
        region,
        _topology_probe_energy_objective,
        n_iterations=1,
        learning_rate=0.05,
        n_steps=8,
        beta_schedule=[(0, 1.0)],
        verbose=False,
        adjoint_mode="hybrid",
        strategy="b",
        checkpoint_every=3,
    )

    np.testing.assert_allclose(np.asarray(strategy_b.history), np.asarray(pure.history), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(np.asarray(strategy_b.density), np.asarray(pure.density), rtol=1e-4, atol=1e-6)


@pytest.mark.skipif(
    not importlib.util.find_spec("optax"),
    reason="optax not installed",
)
def test_phase6_pec_topology_strategy_b_matches_pure_ad_one_step():
    sim, region = _make_topology_case(boundary="pec")

    pure = topology_optimize(
        sim,
        region,
        _topology_probe_energy_objective,
        n_iterations=1,
        learning_rate=0.05,
        n_steps=8,
        beta_schedule=[(0, 1.0)],
        verbose=False,
        adjoint_mode="pure_ad",
    )
    strategy_b = topology_optimize(
        sim,
        region,
        _topology_probe_energy_objective,
        n_iterations=1,
        learning_rate=0.05,
        n_steps=8,
        beta_schedule=[(0, 1.0)],
        verbose=False,
        adjoint_mode="hybrid",
        strategy="b",
        checkpoint_every=3,
    )

    np.testing.assert_allclose(np.asarray(strategy_b.history), np.asarray(pure.history), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(np.asarray(strategy_b.density), np.asarray(pure.density), rtol=1e-4, atol=1e-6)


def test_phase6_cpml_topology_strategy_b_gradient_matches_pure_ad():
    sim, region = _make_topology_case(boundary="cpml")
    inputs, report, grid, *_ = _inspect_topology_hybrid_support(
        sim,
        region,
        n_steps=8,
    )
    assert report.supported
    assert inputs.materials is not None

    def pure_loss(alpha):
        eps = _single_cell_eps(grid, inputs.materials.eps_r, alpha)
        result = sim.forward(eps_override=eps, n_steps=8, checkpoint=True)
        return jnp.sum(result.time_series ** 2)

    def strategy_b_loss(alpha):
        eps = _single_cell_eps(grid, inputs.materials.eps_r, alpha)
        result = sim.forward_hybrid_phase1_from_inputs(
            inputs,
            eps_override=eps,
            strategy="b",
            checkpoint_every=3,
        )
        return jnp.sum(result.time_series ** 2)

    alpha0 = jnp.float32(0.1)
    grad_pure = jax.grad(pure_loss)(alpha0)
    grad_strategy_b = jax.grad(strategy_b_loss)(alpha0)

    np.testing.assert_allclose(np.asarray(grad_strategy_b), np.asarray(grad_pure), rtol=1e-4, atol=1e-7)


def test_phase6_port_proxy_strategy_b_route_uses_inputs_seam(monkeypatch):
    sim = _make_passive_port_optimize_sim()
    calls = {"inputs": 0}
    original = sim.forward_hybrid_phase1_from_inputs

    def _wrapped_inputs(inputs, *, eps_override=None, strategy="a", checkpoint_every=None):
        calls["inputs"] += 1
        assert strategy == "b"
        assert checkpoint_every == 3
        return original(
            inputs,
            eps_override=eps_override,
            strategy=strategy,
            checkpoint_every=checkpoint_every,
        )

    def _fail_pure_ad(*args, **kwargs):
        raise AssertionError("Strategy B port proxy unexpectedly used the pure-AD material path")

    monkeypatch.setattr(sim, "forward_hybrid_phase1_from_inputs", _wrapped_inputs)
    monkeypatch.setattr(sim, "_forward_from_materials", _fail_pure_ad)

    result = optimize(
        sim,
        _make_port_design_region(),
        _probe_energy_objective,
        n_iters=1,
        lr=0.01,
        n_steps=8,
        verbose=False,
        adjoint_mode="hybrid",
        strategy="b",
        checkpoint_every=3,
    )

    assert len(result.loss_history) == 1
    assert calls["inputs"] > 0


def test_phase6_port_proxy_strategy_b_matches_pure_ad_one_step():
    sim = _make_passive_port_optimize_sim()

    pure = optimize(
        sim,
        _make_port_design_region(),
        _probe_energy_objective,
        n_iters=1,
        lr=0.01,
        n_steps=8,
        verbose=False,
        adjoint_mode="pure_ad",
    )
    strategy_b = optimize(
        sim,
        _make_port_design_region(),
        _probe_energy_objective,
        n_iters=1,
        lr=0.01,
        n_steps=8,
        verbose=False,
        adjoint_mode="hybrid",
        strategy="b",
        checkpoint_every=3,
    )

    np.testing.assert_allclose(
        np.asarray(strategy_b.latent),
        np.asarray(pure.latent),
        rtol=1e-4,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(strategy_b.loss_history),
        np.asarray(pure.loss_history),
        rtol=1e-5,
        atol=1e-7,
    )


def test_phase6_port_proxy_strategy_b_rejects_passive_port_overlap():
    sim = _make_passive_port_optimize_sim()

    with pytest.raises(ValueError, match=re.escape("design region overlaps a passive lumped-port cell")):
        optimize(
            sim,
            _make_overlapping_passive_port_design_region(),
            _probe_energy_objective,
            n_iters=1,
            lr=0.01,
            n_steps=8,
            verbose=False,
            adjoint_mode="hybrid",
            strategy="b",
            checkpoint_every=3,
        )


def test_phase6_port_proxy_strategy_b_auto_raises_on_explicit_unsupported_request():
    sim = _make_passive_port_optimize_sim()

    with pytest.raises(ValueError, match=re.escape("design region overlaps a passive lumped-port cell")):
        optimize(
            sim,
            _make_overlapping_passive_port_design_region(),
            _probe_energy_objective,
            n_iters=1,
            lr=0.01,
            n_steps=8,
            verbose=False,
            adjoint_mode="auto",
            strategy="b",
            checkpoint_every=3,
        )


@pytest.mark.skipif(
    not importlib.util.find_spec("optax"),
    reason="optax not installed",
)
def test_phase6_topology_strategy_b_auto_raises_on_explicit_unsupported_request():
    sim, region = _make_topology_case(boundary="cpml")
    sim.add_material("phase6_lossy", eps_r=4.0, sigma=0.02)
    lossy_region = TopologyDesignRegion(
        corner_lo=region.corner_lo,
        corner_hi=region.corner_hi,
        material_bg=region.material_bg,
        material_fg="phase6_lossy",
        beta_projection=region.beta_projection,
    )

    with pytest.raises(ValueError, match="zero sigma"):
        topology_optimize(
            sim,
            lossy_region,
            _topology_probe_energy_objective,
            n_iterations=1,
            learning_rate=0.05,
            n_steps=8,
            beta_schedule=[(0, 1.0)],
            verbose=False,
            adjoint_mode="auto",
            strategy="b",
            checkpoint_every=3,
        )


def test_phase6_port_proxy_strategy_b_gradient_matches_pure_ad():
    sim = _make_passive_port_optimize_sim()
    grid = sim._build_grid()
    materials, *_ = sim._assemble_materials(grid)
    inputs = sim.build_hybrid_phase1_inputs(n_steps=8)

    def pure_loss(alpha):
        eps = _single_cell_eps(grid, materials.eps_r, alpha)
        result = sim.forward(eps_override=eps, n_steps=8, checkpoint=True)
        return jnp.sum(result.time_series ** 2)

    def strategy_b_loss(alpha):
        eps = _single_cell_eps(grid, materials.eps_r, alpha)
        result = sim.forward_hybrid_phase1_from_inputs(
            inputs,
            eps_override=eps,
            strategy="b",
            checkpoint_every=3,
        )
        return jnp.sum(result.time_series ** 2)

    alpha0 = jnp.float32(0.1)
    grad_pure = jax.grad(pure_loss)(alpha0)
    grad_strategy_b = jax.grad(strategy_b_loss)(alpha0)

    np.testing.assert_allclose(np.asarray(grad_strategy_b), np.asarray(grad_pure), rtol=1e-4, atol=1e-7)
