"""Internal proxy comparisons for arbitrary all-PEC box refinement."""

from __future__ import annotations

import numpy as np
import pytest


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


def _run_case(*, dx, refinement=None, source, probe, domain=(0.04, 0.04, 0.04), n_steps=500, freq_max=10e9):
    from rfx import Simulation

    sim = Simulation(freq_max=freq_max, domain=domain, boundary="pec", dx=dx)
    if refinement is not None:
        sim.add_refinement(**refinement)
    sim.add_source(position=source, component="ez")
    sim.add_probe(position=probe, component="ez")
    return sim.run(n_steps=n_steps)


class TestAllPecBoxProxyBenchmarks:
    freq_eval = 1.5e9
    uniform_dx = 1e-3
    coarse_dx = 2e-3

    @staticmethod
    def _refinement():
        return {
            "x_range": (0.010, 0.028),
            "y_range": (0.010, 0.028),
            "z_range": (0.010, 0.028),
            "ratio": 2,
        }

    def test_x_face_proxy_vs_uniform_fine(self):
        refinement = self._refinement()
        source = (0.016, 0.020, 0.020)
        probe = (0.022, 0.020, 0.020)
        result_ref = _run_case(dx=self.uniform_dx, source=source, probe=probe)
        result_sub = _run_case(dx=self.coarse_dx, refinement=refinement, source=source, probe=probe)
        amp_error, phase_error = _benchmark_errors(result_ref, result_sub, self.freq_eval)
        assert amp_error <= 0.10
        assert phase_error <= 5.0

    def test_y_face_proxy_vs_uniform_fine(self):
        refinement = self._refinement()
        source = (0.020, 0.016, 0.020)
        probe = (0.020, 0.022, 0.020)
        result_ref = _run_case(dx=self.uniform_dx, source=source, probe=probe)
        result_sub = _run_case(dx=self.coarse_dx, refinement=refinement, source=source, probe=probe)
        amp_error, phase_error = _benchmark_errors(result_ref, result_sub, self.freq_eval)
        assert amp_error <= 0.10
        assert phase_error <= 5.0

    def test_edge_proxy_vs_uniform_fine(self):
        refinement = self._refinement()
        source = (0.014, 0.014, 0.020)
        probe = (0.022, 0.022, 0.020)
        freq_eval = 0.8e9
        result_ref = _run_case(dx=self.uniform_dx, source=source, probe=probe, n_steps=400)
        result_sub = _run_case(dx=self.coarse_dx, refinement=refinement, source=source, probe=probe, n_steps=400)
        amp_error, phase_error = _benchmark_errors(result_ref, result_sub, freq_eval)
        assert amp_error <= 0.25
        assert phase_error <= 15.0

    def test_corner_proxy_vs_uniform_fine(self):
        refinement = self._refinement()
        source = (0.014, 0.014, 0.014)
        probe = (0.022, 0.022, 0.022)
        freq_eval = 1.0e9
        result_ref = _run_case(dx=self.uniform_dx, source=source, probe=probe, n_steps=400)
        result_sub = _run_case(dx=self.coarse_dx, refinement=refinement, source=source, probe=probe, n_steps=400)
        amp_error, phase_error = _benchmark_errors(result_ref, result_sub, freq_eval)
        assert amp_error <= 0.15
        assert phase_error <= 5.0
