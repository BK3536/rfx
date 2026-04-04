"""GPU Accuracy Validation: Free-Space Energy Decay (CPML)

Validates that the FDTD simulator with CPML boundaries correctly
reproduces the 1/r^2 energy decay (1/r field decay) of a radiating
source in free space.

Physics:
  A point source in free space radiates spherically.  The power density
  decays as 1/r^2 (Poynting vector), and the E-field amplitude decays
  as 1/r.  Probes at increasing distances from the source should show
  energy proportional to 1/r^2.

  In dB: energy_dB ~ -20*log10(r/r0)

Validation criteria:
  - Fit energy vs distance to power law: E ~ r^(-n)
  - Fitted exponent n should be in range [1.0, 3.0]
    (ideal = 2.0 for 3D free space; near-field contributions at
    short distances push toward 3.0, grid dispersion can shift it)
  - Energy should decrease monotonically with distance

Reference:
  Balanis, "Antenna Theory", 4th ed., Ch 2 (Fundamental Parameters)
  Free-space power density: S = P_rad / (4*pi*r^2)

Exit 0 on PASS, 1 on FAIL.
"""

import sys
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rfx import Simulation, GaussianPulse
from rfx.grid import C0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = SCRIPT_DIR

# Valid range for energy decay exponent (ideal = 2.0 in 3D)
# Near-field contributions and grid dispersion can push the fitted
# exponent toward 3.0 (reactive near-field ~ 1/r^3) or below 2.0.
# A range of [1.0, 3.0] is physically reasonable for the mix of
# near-field and far-field distances used here.
EXPONENT_MIN = 1.0
EXPONENT_MAX = 3.0


def main():
    t_start = time.time()

    f0 = 2.4e9
    lam = C0 / f0  # ~125 mm
    dx = 2.0e-3    # 2 mm -> lambda/62

    # Domain: large enough to place probes at several distances
    # Need >lambda/4 margin from probes to CPML boundary
    cpml_margin = lam * 0.35  # ~44 mm, generous CPML clearance
    max_probe_dist = lam * 1.2  # furthest probe distance
    dom_half = max_probe_dist + cpml_margin
    dom = 2 * dom_half  # symmetric domain

    print("=" * 60)
    print("GPU VALIDATION: Free-Space Energy Decay (CPML)")
    print("=" * 60)
    print(f"Frequency   : {f0/1e9:.1f} GHz (lambda = {lam*1e3:.0f} mm)")
    print(f"Resolution  : dx = {dx*1e3:.1f} mm (lambda/{lam/dx:.0f})")
    print(f"Domain      : {dom*1e3:.0f} mm cube")
    print(f"CPML margin : {cpml_margin*1e3:.0f} mm (lambda*{cpml_margin/lam:.2f})")
    print()

    # --- Probe distances from source ---
    n_probes = 8
    probe_distances = np.linspace(0.2 * lam, max_probe_dist, n_probes)

    # Source at the centre of the domain
    src_x = dom / 2
    src_y = dom / 2
    src_z = dom / 2

    print(f"Probe distances: {probe_distances[0]*1e3:.0f} to {probe_distances[-1]*1e3:.0f} mm "
          f"({probe_distances[0]/lam:.2f} to {probe_distances[-1]/lam:.2f} lambda, {n_probes} points)")

    # --- Build simulation with CPML boundary ---
    sim = Simulation(
        freq_max=f0 * 2,
        domain=(dom, dom, dom),
        boundary="cpml",
        cpml_layers=12,
        dx=dx,
    )

    # Excitation source
    sim.add_port(
        (src_x, src_y, src_z),
        component="ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=f0, bandwidth=0.6),
    )

    # Probes along the +x axis at increasing distances
    for d in probe_distances:
        sim.add_probe((src_x + d, src_y, src_z), component="ez")

    # --- Run simulation ---
    n_steps = 3000
    print(f"\nRunning FDTD ({n_steps} steps, CPML boundary) ...")
    result = sim.run(n_steps=n_steps, compute_s_params=False)

    ts = np.asarray(result.time_series)
    dt = float(result.dt)

    print(f"Time series shape: {ts.shape}")

    # --- Compute energy at each probe ---
    energies = []
    for i in range(n_probes):
        e = float(np.sum(ts[:, i] ** 2))
        energies.append(e)

    energies = np.array(energies)
    energy_db = 10 * np.log10(np.maximum(energies / energies[0], 1e-30))

    print(f"\nEnergy at each probe (relative to closest):")
    for i, (d, e_db) in enumerate(zip(probe_distances, energy_db)):
        print(f"  r = {d*1e3:6.0f} mm ({d/lam:.2f} lam) : {e_db:7.2f} dB")

    # --- Fit power-law: energy ~ r^(-n) ---
    # In dB: energy_dB = C - n * 10*log10(r)
    log_r = np.log10(probe_distances)

    # Use all probes for fit
    coeffs = np.polyfit(log_r, energy_db, 1)
    fitted_exponent = -coeffs[0] / 10.0  # energy_dB = C - n*10*log10(r)
    fitted_curve_db = np.polyval(coeffs, log_r)

    # Theoretical 1/r^2 reference (20 dB/decade)
    r_ratio = probe_distances / probe_distances[0]
    theory_db = -20 * np.log10(r_ratio)

    # --- Check monotonicity ---
    diffs = np.diff(energy_db)
    monotonic = np.all(diffs <= 0.5)  # allow minor numerical fluctuation

    # --- Validation ---
    elapsed = time.time() - t_start

    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Fitted decay exponent  : {fitted_exponent:.3f}")
    print(f"Expected (3D ideal)    : 2.000")
    print(f"Allowed range          : [{EXPONENT_MIN}, {EXPONENT_MAX}]")
    print(f"Monotonic decrease     : {'Yes' if monotonic else 'No'}")
    print(f"Elapsed time           : {elapsed:.1f}s")

    exponent_valid = EXPONENT_MIN <= fitted_exponent <= EXPONENT_MAX
    print(f"\nCriteria:")
    print(f"  Exponent in [{EXPONENT_MIN}, {EXPONENT_MAX}]  : {'PASS' if exponent_valid else 'FAIL'} ({fitted_exponent:.3f})")
    print(f"  Monotonic decrease       : {'PASS' if monotonic else 'FAIL'}")

    # --- Figures ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("GPU Validation: Free-Space Energy Decay (Balanis Ch 2)", fontsize=14, fontweight="bold")

    # Panel 1: Energy vs distance with fit
    ax = axes[0, 0]
    d_lam = probe_distances / lam
    ax.plot(d_lam, energy_db, "bo-", markersize=7, lw=1.5, label="FDTD")
    ax.plot(d_lam, fitted_curve_db, "r--", lw=1.5,
            label=f"Fit: n = {fitted_exponent:.2f}")
    ax.plot(d_lam, theory_db, "g:", lw=1.5, alpha=0.6, label="Ideal 1/r^2")
    ax.set_xlabel("Distance (wavelengths)")
    ax.set_ylabel("Relative energy (dB)")
    ax.set_title("Energy Decay vs Distance")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Time-domain signals at near and far probes
    ax = axes[0, 1]
    t_ns = np.arange(ts.shape[0]) * dt * 1e9
    for idx, label_color in [(0, ("b", f"r={probe_distances[0]*1e3:.0f} mm (near)")),
                              (n_probes // 2, ("g", f"r={probe_distances[n_probes//2]*1e3:.0f} mm (mid)")),
                              (-1, ("r", f"r={probe_distances[-1]*1e3:.0f} mm (far)"))]:
        color, label = label_color
        ax.plot(t_ns, ts[:, idx], color=color, lw=0.6, alpha=0.7, label=label)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Ez (V/m)")
    ax.set_title("Time-Domain Probe Signals")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Log-log energy vs distance
    ax = axes[1, 0]
    ax.loglog(probe_distances * 1e3, energies / energies[0], "bo-", markersize=7, lw=1.5, label="FDTD")
    r_fine = np.linspace(probe_distances[0], probe_distances[-1], 100)
    ax.loglog(r_fine * 1e3, (r_fine / probe_distances[0]) ** (-2 * fitted_exponent),
              "r--", lw=1.5, label=f"Fit: r^(-{2*fitted_exponent:.2f})")
    ax.loglog(r_fine * 1e3, (r_fine / probe_distances[0]) ** (-2),
              "g:", lw=1.5, alpha=0.6, label="Ideal: r^(-2)")
    ax.set_xlabel("Distance (mm)")
    ax.set_ylabel("Relative energy")
    ax.set_title("Log-Log Energy Decay")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    # Panel 4: Summary
    ax = axes[1, 1]
    ax.axis("off")
    verdict = "PASS" if (exponent_valid and monotonic) else "FAIL"
    lines = [
        "Free-Space Energy Decay Validation",
        "-" * 38,
        f"f0 = {f0/1e9:.1f} GHz, lambda = {lam*1e3:.0f} mm",
        f"dx = {dx*1e3:.1f} mm (lambda/{lam/dx:.0f})",
        f"Domain = {dom*1e3:.0f} mm cube, CPML 12 layers",
        "",
        f"Fitted exponent: {fitted_exponent:.3f}",
        f"Ideal (3D)     : 2.000",
        f"Allowed range  : [{EXPONENT_MIN}, {EXPONENT_MAX}]",
        f"Monotonic      : {'Yes' if monotonic else 'No'}",
        "",
        "Energy at probes:",
    ]
    for d, e_db in zip(probe_distances[::2], energy_db[::2]):
        lines.append(f"  r={d*1e3:5.0f} mm ({d/lam:.2f}l): {e_db:7.2f} dB")
    lines.extend([
        "",
        f"Verdict: {verdict}",
        f"Time: {elapsed:.1f}s",
    ])
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes, va="top",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "04_coupling_validation.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved: {out_path}")

    # --- Pass/Fail ---
    passed = exponent_valid and monotonic
    if passed:
        print(f"\nPASS: Energy decay exponent {fitted_exponent:.3f} in [{EXPONENT_MIN}, {EXPONENT_MAX}] and monotonic")
        sys.exit(0)
    else:
        print(f"\nFAIL: Exponent {fitted_exponent:.3f} (expected [{EXPONENT_MIN}, {EXPONENT_MAX}]) or not monotonic")
        sys.exit(1)


if __name__ == "__main__":
    main()
