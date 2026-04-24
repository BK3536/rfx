"""Internal proxy benchmarks for implemented boundary coexistence subsets."""

from __future__ import annotations

import numpy as np
import pytest

from rfx.boundaries.spec import Boundary, BoundarySpec


pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def _dft_amplitude_phase(signal: np.ndarray, dt: float, freq_hz: float) -> tuple[float, float]:
    t = np.arange(len(signal)) * dt
    coeff = np.sum(signal.astype(np.float64) * np.exp(-1j * 2.0 * np.pi * freq_hz * t)) * dt
    return abs(coeff), float(np.angle(coeff, deg=True))


def _phase_error_deg(phase_a: float, phase_b: float) -> float:
    return abs((phase_a - phase_b + 180.0) % 360.0 - 180.0)


def _benchmark_errors(result_ref, result_sub, freq_hz: float) -> tuple[float, float]:
    amp_ref, phase_ref = _dft_amplitude_phase(np.asarray(result_ref.time_series[:, 0]), float(result_ref.dt), freq_hz)
    amp_sub, phase_sub = _dft_amplitude_phase(np.asarray(result_sub.time_series[:, 0]), float(result_sub.dt), freq_hz)
    return abs(amp_sub - amp_ref) / max(amp_ref, 1e-30), _phase_error_deg(phase_sub, phase_ref)


def _run_case(*, boundary, dx, refinement=None, source, probe, n_steps=300, freq_max=8e9):
    from rfx import Simulation

    sim = Simulation(freq_max=freq_max, domain=(0.04, 0.04, 0.04), boundary=boundary, dx=dx)
    if refinement is not None:
        sim.add_refinement(**refinement)
    sim.add_source(position=source, component="ez")
    sim.add_probe(position=probe, component="ez")
    return sim.run(n_steps=n_steps)


class TestBoundaryCoexistenceProxyBenchmarks:
    def test_reflector_only_pmc_proxy_vs_uniform_fine(self):
        boundary = BoundarySpec(x="pec", y="pec", z=Boundary(lo="pmc", hi="pec"))
        refinement = {"z_range": (0.0, 0.028), "ratio": 2}
        source = (0.020, 0.020, 0.012)
        probe = (0.020, 0.020, 0.020)
        result_ref = _run_case(boundary=boundary, dx=1e-3, source=source, probe=probe)
        result_sub = _run_case(boundary=boundary, dx=2e-3, refinement=refinement, source=source, probe=probe)
        amp_error, phase_error = _benchmark_errors(result_ref, result_sub, 1.0e9)
        assert amp_error <= 0.05
        assert phase_error <= 1.0

    @pytest.mark.parametrize(
        "boundary, refinement",
        [
            (
                BoundarySpec(x="periodic", y="pec", z="pec"),
                {"x_range": (0.0, 0.04), "y_range": (0.012, 0.028), "z_range": (0.012, 0.028), "ratio": 2},
            ),
            (
                BoundarySpec(x="periodic", y="pec", z="pec"),
                {"x_range": (0.012, 0.028), "y_range": (0.012, 0.028), "z_range": (0.012, 0.028), "ratio": 2},
            ),
            (
                BoundarySpec(x="pec", y="periodic", z="pec"),
                {"x_range": (0.012, 0.028), "y_range": (0.0, 0.04), "z_range": (0.012, 0.028), "ratio": 2},
            ),
        ],
    )
    def test_periodic_full_axis_proxy_vs_uniform_fine(self, boundary, refinement):
        source = (0.020, 0.020, 0.020)
        probe = (0.020, 0.020, 0.020)
        result_ref = _run_case(boundary=boundary, dx=1e-3, source=source, probe=probe)
        result_sub = _run_case(boundary=boundary, dx=2e-3, refinement=refinement, source=source, probe=probe)
        amp_error, phase_error = _benchmark_errors(result_ref, result_sub, 1.0e9)
        assert amp_error <= 0.01
        assert phase_error <= 0.1
