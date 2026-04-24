"""Internal proxy benchmarks for the bounded CPML subgrid subset.

These tests are intentionally proxy-level evidence. They prove that the first
absorbing coexistence path is finite and dissipative for an interior box outside
the CPML guard band; they are not reflection/transmission or S-parameter claims.
"""

from __future__ import annotations

import numpy as np
import pytest


pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def _run_subgrid_case(*, boundary: str, cpml_layers: int, n_steps: int = 500):
    from rfx import Simulation

    sim = Simulation(
        freq_max=6e9,
        domain=(0.08, 0.06, 0.05),
        boundary=boundary,
        cpml_layers=cpml_layers,
        dx=2e-3,
    )
    sim.add_refinement(
        x_range=(0.030, 0.050),
        y_range=(0.022, 0.038),
        z_range=(0.018, 0.038),
        ratio=2,
    )
    sim.add_source(position=(0.040, 0.030, 0.024), component="ez")
    sim.add_probe(position=(0.040, 0.030, 0.032), component="ez")
    return sim.run(n_steps=n_steps)


def _peak_and_tail(result) -> tuple[float, float]:
    signal = np.asarray(result.time_series[:, 0])
    peak = float(np.max(np.abs(signal)))
    tail = float(np.max(np.abs(signal[-100:])))
    return peak, tail


class TestAbsorbingCoexistenceProxyBenchmarks:
    def test_cpml_subgrid_interior_box_is_finite_and_decays(self):
        result = _run_subgrid_case(boundary="cpml", cpml_layers=4)
        signal = np.asarray(result.time_series[:, 0])
        peak, tail = _peak_and_tail(result)

        assert np.all(np.isfinite(signal))
        assert peak > 1e-6
        assert tail / peak <= 0.01

    def test_cpml_subgrid_late_tail_is_below_pec_cavity_tail(self):
        result_cpml = _run_subgrid_case(boundary="cpml", cpml_layers=4)
        result_pec = _run_subgrid_case(boundary="pec", cpml_layers=0)
        _, tail_cpml = _peak_and_tail(result_cpml)
        _, tail_pec = _peak_and_tail(result_pec)

        assert tail_cpml <= 0.5 * tail_pec
