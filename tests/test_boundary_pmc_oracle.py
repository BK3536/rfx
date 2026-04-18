"""T7-E Phase 2 — PMC λ/4 cavity mode-ladder oracle (v1.7.2).

Closes critic blocker #5: the shipped PMC tests (tangential-H=0,
dual-boundary Hx sample) prove the PMC hook fires and is a distinct
code path from PEC, but they do not prove PMC is a magnetic wall
(reflection sign convention, quarter-wave resonance).

## Analytic basis

In a 1D-like cavity of length L with PEC at z = L (``E_tan = 0``) and
PMC at z = 0 (``H_tan = 0``, which forces ``∂E_tan/∂z = 0`` at the
wall), the transverse-electric standing-wave modes satisfy

    E_x(z, t) = cos(k_n z) · cos(ω_n t)
    boundary conditions: cos(k_n · L) = 0  =>  k_n · L = (2n + 1) π / 2

giving resonance frequencies

    f_n = c · (2n + 1) / (4 L),     n = 0, 1, 2, ...

So the PEC-PMC ladder is f_0 = c/(4L), f_1 = 3c/(4L), f_2 = 5c/(4L),
spaced at ratio 1 : 3 : 5 : 7. By contrast a PEC-PEC cavity supports

    f_n^{PEC-PEC} = n · c / (2 L),  ladder 1 : 2 : 3 : 4.

The spacing ratio f_1 / f_0 is **3 for PEC-PMC** and **2 for PEC-PEC**;
a pure 2% frequency check on either peak alone cannot distinguish the
two, but the SPACING RATIO does. This test asserts the ladder AND
includes a negative PEC-PEC control.

## Test configuration

- Narrow transverse xy (periodic) to suppress non-axial modes.
- Cavity z-length L locked so f_0 = c/(4L) lies well below the
  source frequency cap and f_1 = 3 f_0 is still covered.
- Gaussian impulse E_x source near the centre of the interior.
- Probe E_x at z = 0.7 L (clear of the n=0 and n=1 nodes).
- Ez DFT over a long run (2048 steps) to resolve the ladder.
- Peak detection finds the two strongest DFT peaks in the
  [0.5 f_0, 4 f_0] band.

## What this oracle pins

1. **Spacing ratio f_1 / f_0 ∈ [2.5, 3.5]**: distinguishes PEC-PMC
   (analytic 3.0) from PEC-PEC (analytic 2.0). Tolerance is wide
   enough to absorb the finite-cavity + discretization frequency
   drift but tight enough to separate the two ladders.
2. **f_0 within 10 % of analytic c/(4L)**: confirms the quarter-wave
   resonance is where PEC-PMC physics predicts.
3. **Negative control**: the same cavity with PMC swapped for PEC
   on z_lo fails the 3.0 spacing ratio (lands at 2.0, PEC-PEC ladder).
"""

from __future__ import annotations

import numpy as np

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec


_C0 = 299_792_458.0
_L_CAVITY = 0.02              # 20 mm z-axis interior
_DX = 0.5e-3                  # 0.5 mm cells → 40 interior cells
_N_STEPS = 2048
_F0_ANALYTIC = _C0 / (4.0 * _L_CAVITY)  # quarter-wave for PEC-PMC


def _run_cavity(z_lo_token: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (freqs, |E(f)|, dt) for the probe spectrum of a
    z-axis cavity with ``z_hi='pec'`` and ``z_lo=z_lo_token``.
    """
    # cpml_layers=0 produces a closed grid — the PMC / PEC walls sit
    # at the grid edge and no CPML is allocated on any axis.
    spec = BoundarySpec(
        x="periodic", y="periodic",
        z=Boundary(lo=z_lo_token, hi="pec"),
    )
    sim = Simulation(
        freq_max=40e9,                          # covers through f_2 = 5 f_0
        domain=(0.002, 0.002, _L_CAVITY),
        dx=_DX, boundary=spec, cpml_layers=0,
    )
    # Source and probe on the z-axis, clear of analytic nodes.
    sim.add_source((0.001, 0.001, 0.3 * _L_CAVITY), "ex")
    sim.add_probe((0.001, 0.001, 0.7 * _L_CAVITY), "ex")
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")  # skip preflight advisories on this tiny cavity
        res = sim.run(n_steps=_N_STEPS)
    ts = np.asarray(res.time_series)[:, 0]
    dt = float(res.dt)
    # Hann window to reduce sidelobes while the cavity rings.
    window = np.hanning(_N_STEPS)
    spec_mag = np.abs(np.fft.rfft(ts * window))
    freqs = np.fft.rfftfreq(_N_STEPS, dt)
    return freqs, spec_mag, dt


def _two_strongest_peaks(freqs: np.ndarray, spec: np.ndarray,
                         band: tuple[float, float]) -> tuple[float, float]:
    """Find the two strongest spectral peaks inside ``band``."""
    mask = (freqs >= band[0]) & (freqs <= band[1])
    band_freqs = freqs[mask]
    band_spec = spec[mask]
    # Sort by amplitude descending; pick the top two peaks with >= 2 bins
    # separation to reject neighbouring-bin spillover.
    order = np.argsort(-band_spec)
    chosen = []
    for idx in order:
        if all(abs(idx - c) >= 2 for c in chosen):
            chosen.append(int(idx))
        if len(chosen) == 2:
            break
    f_sorted = sorted(band_freqs[c] for c in chosen)
    return float(f_sorted[0]), float(f_sorted[1])


def test_pmc_lambda_quarter_two_peak_ladder():
    """PEC-PMC cavity produces a 1:3 spacing ratio (quarter-wave ladder).
    First mode within 10 % of analytic c/(4L)."""
    freqs, spec, _ = _run_cavity("pmc")
    band = (0.5 * _F0_ANALYTIC, 4.0 * _F0_ANALYTIC)
    f0, f1 = _two_strongest_peaks(freqs, spec, band)
    assert abs(f0 - _F0_ANALYTIC) / _F0_ANALYTIC < 0.10, (
        f"PMC-PEC f_0 must land within 10 % of analytic c/(4L) = "
        f"{_F0_ANALYTIC:.3e} Hz; got {f0:.3e} Hz"
    )
    ratio = f1 / f0
    assert 2.5 < ratio < 3.5, (
        f"PEC-PMC cavity ladder spacing f_1/f_0 must be near 3.0 "
        f"(quarter-wave); got f_0={f0:.3e}, f_1={f1:.3e}, "
        f"ratio={ratio:.3f}. PEC-PEC half-wave gives ratio 2.0."
    )


def test_pec_cavity_fails_pmc_ladder_negative_control():
    """Negative control: a PEC-PEC cavity has half-wave ladder
    f_1/f_0 = 2.0, which MUST fail the PMC quarter-wave check."""
    freqs, spec, _ = _run_cavity("pec")
    # PEC-PEC f_0 = c/(2L) = 2 · _F0_ANALYTIC (the quarter-wave analytic).
    band = (0.5 * _F0_ANALYTIC, 4.0 * _F0_ANALYTIC)
    f0, f1 = _two_strongest_peaks(freqs, spec, band)
    ratio = f1 / f0
    assert ratio <= 2.5 or ratio >= 3.5, (
        f"PEC-PEC control must NOT satisfy the PMC quarter-wave spacing "
        f"check (test would otherwise pass on any reflector). Got "
        f"f_0={f0:.3e}, f_1={f1:.3e}, ratio={ratio:.3f}; need outside "
        f"[2.5, 3.5]."
    )
