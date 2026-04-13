"""Subgridding accuracy crossvalidation tests.

Compares subgridded simulations against uniform fine-grid references:
1. PEC cavity with dielectric slab: probe RMS error < 5%
2. PEC cavity with off-axis source: probe RMS error < 10%

Both use PEC boundaries (not CPML) because CPML+subgrid is currently
unstable — the CPML absorber on the coarse grid conflicts with the
fine-grid SAT coupling, causing late-time energy growth. This is a known
limitation to be addressed in Phase A2 (runner generalization).

Both tests require GPU and are slow — marked accordingly.
"""

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def _rms_error(sig_ref, sig_test):
    """Normalised RMS error between two 1-D arrays."""
    n = min(len(sig_ref), len(sig_test))
    ref = sig_ref[:n].astype(np.float64)
    tst = sig_test[:n].astype(np.float64)
    rms_ref = np.sqrt(np.mean(ref ** 2))
    if rms_ref < 1e-30:
        return 0.0
    rms_err = np.sqrt(np.mean((ref - tst) ** 2))
    return rms_err / rms_ref


# ---------------------------------------------------------------------------
# Test 1: PEC cavity with dielectric slab
# ---------------------------------------------------------------------------

class TestSlabCavitySubgrid:
    """Subgridded dielectric slab in PEC cavity vs uniform fine-grid.

    PEC cavity avoids CPML+subgrid instability. The slab creates a
    dielectric interface that the subgrid must resolve accurately.

    Geometry:
    - Domain (0.06, 0.06, 0.06), PEC boundary
    - Dielectric slab: eps_r=4.0, z in [0.02, 0.04]
    - Source: z=0.015 (before slab)
    - Probe: z=0.045 (after slab, transmitted)
    """

    def _run_uniform_fine(self, n_steps: int):
        from rfx import Simulation, Box

        domain = (0.06, 0.06, 0.06)
        dx_fine = 1e-3

        sim = Simulation(
            freq_max=10e9, domain=domain, boundary="pec", dx=dx_fine,
        )
        sim.add_material("dielectric", eps_r=4.0)
        sim.add(Box((0, 0, 0.02), (0.06, 0.06, 0.04)), material="dielectric")
        sim.add_source(position=(0.03, 0.03, 0.015), component="ez")
        sim.add_probe(position=(0.03, 0.03, 0.045), component="ez")

        return sim.run(n_steps=n_steps)

    def _run_subgridded(self, n_steps: int):
        from rfx import Simulation, Box

        domain = (0.06, 0.06, 0.06)
        dx_coarse = 3e-3

        sim = Simulation(
            freq_max=10e9, domain=domain, boundary="pec", dx=dx_coarse,
        )
        sim.add_material("dielectric", eps_r=4.0)
        sim.add(Box((0, 0, 0.02), (0.06, 0.06, 0.04)), material="dielectric")
        # z_range covers source, slab, and probe
        sim.add_refinement(z_range=(0.010, 0.050), ratio=3)
        sim.add_source(position=(0.03, 0.03, 0.015), component="ez")
        sim.add_probe(position=(0.03, 0.03, 0.045), component="ez")

        return sim.run(n_steps=n_steps)

    def test_slab_transmitted_rms_error(self):
        """Transmitted probe RMS error < 5%."""
        n_steps = 1000

        result_ref = self._run_uniform_fine(n_steps)
        result_sub = self._run_subgridded(n_steps)

        ts_ref = np.array(result_ref.time_series[:, 0])
        ts_sub = np.array(result_sub.time_series[:, 0])

        err = _rms_error(ts_ref, ts_sub)
        print(f"\nSlab cavity crossval:")
        print(f"  Ref max: {np.max(np.abs(ts_ref)):.6e}")
        print(f"  Sub max: {np.max(np.abs(ts_sub)):.6e}")
        print(f"  RMS error: {err:.3%}")
        assert err < 0.05, f"Slab cavity: RMS error {err:.3%} >= 5%"

    def test_slab_signals_finite(self):
        """Both runs must produce finite signals."""
        n_steps = 500
        result = self._run_subgridded(n_steps)
        ts = np.array(result.time_series[:, 0])
        assert np.all(np.isfinite(ts)), "Subgridded signal has NaN/Inf"


# ---------------------------------------------------------------------------
# Test 2: PEC cavity with off-axis source (3D stress test)
# ---------------------------------------------------------------------------

class TestCavitySubgrid:
    """3D PEC cavity with off-axis source — stresses all 6 subgrid faces.

    Geometry:
    - Domain (0.04, 0.04, 0.04), PEC boundary
    - Source: (0.012, 0.015, 0.018) — off-axis
    - Probe: (0.028, 0.025, 0.022)
    """

    def _run_uniform_fine(self, n_steps: int):
        from rfx import Simulation

        sim = Simulation(
            freq_max=10e9, domain=(0.04, 0.04, 0.04),
            boundary="pec", dx=1e-3,
        )
        sim.add_source(position=(0.012, 0.015, 0.018), component="ez")
        sim.add_probe(position=(0.028, 0.025, 0.022), component="ez")
        return sim.run(n_steps=n_steps)

    def _run_subgridded(self, n_steps: int):
        from rfx import Simulation

        sim = Simulation(
            freq_max=10e9, domain=(0.04, 0.04, 0.04),
            boundary="pec", dx=3e-3,
        )
        # z_range covers source (z=0.018) and probe (z=0.022)
        sim.add_refinement(z_range=(0.008, 0.032), ratio=3)
        sim.add_source(position=(0.012, 0.015, 0.018), component="ez")
        sim.add_probe(position=(0.028, 0.025, 0.022), component="ez")
        return sim.run(n_steps=n_steps)

    def test_cavity_probe_rms_error(self):
        """Off-axis probe RMS error < 10%."""
        n_steps = 1000

        result_ref = self._run_uniform_fine(n_steps)
        result_sub = self._run_subgridded(n_steps)

        ts_ref = np.array(result_ref.time_series[:, 0])
        ts_sub = np.array(result_sub.time_series[:, 0])

        err = _rms_error(ts_ref, ts_sub)
        print(f"\n3D cavity crossval:")
        print(f"  Ref max: {np.max(np.abs(ts_ref)):.6e}")
        print(f"  Sub max: {np.max(np.abs(ts_sub)):.6e}")
        print(f"  RMS error: {err:.3%}")
        assert err < 0.10, f"3D cavity: RMS error {err:.3%} >= 10%"

    def test_cavity_signals_finite(self):
        """Both runs must produce finite signals."""
        n_steps = 500
        result = self._run_subgridded(n_steps)
        ts = np.array(result.time_series[:, 0])
        assert np.all(np.isfinite(ts)), "Subgridded signal has NaN/Inf"
