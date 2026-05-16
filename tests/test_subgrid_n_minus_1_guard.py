"""N-1 guard and introspection tests for the guarded subgrid core.

This branch carries ``add_refinement`` and ``validate_subgrid`` but not the
SBP-SAT subgrid runner. ``run()`` on a refined ``Simulation`` must fail loudly
with ``NotImplementedError`` rather than silently dropping the refinement or
crashing on unmerged runner code.
"""

from __future__ import annotations

import inspect

import pytest

from rfx import GaussianPulse, Simulation


def _refined_simulation() -> Simulation:
    """Return a small PEC ``Simulation`` with one z-slab refinement."""
    sim = Simulation(
        freq_max=6e9,
        domain=(0.04, 0.04, 0.02),
        boundary="pec",
        cpml_layers=0,
        dx=0.002,
    )
    sim.add_refinement((0.004, 0.012), ratio=4)
    sim.add_source(
        (0.02, 0.02, 0.008),
        "ez",
        waveform=GaussianPulse(f0=3e9, bandwidth=0.5),
    )
    sim.add_probe((0.02, 0.02, 0.01), "ez")
    return sim


def test_run_on_refined_sim_raises_not_implemented():
    """run() on a refined Simulation raises NotImplementedError (N-1 guard)."""
    sim = _refined_simulation()
    with pytest.raises(NotImplementedError, match="subgrid runner"):
        sim.run(n_steps=20, compute_s_params=False)


def test_add_refinement_signature_matches_documented_kwargs():
    """add_refinement exposes exactly the documented parameter set."""
    params = set(inspect.signature(Simulation.add_refinement).parameters)
    params.discard("self")
    assert params == {"z_range", "ratio", "xy_margin", "tau", "validation", "topology"}
