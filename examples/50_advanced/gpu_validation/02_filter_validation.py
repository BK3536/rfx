"""GPU Accuracy Validation: PEC Cavity Resonance via Harminv

Validates the FDTD simulator's ability to reproduce the analytical
resonant frequency of a PEC rectangular cavity with an iris aperture.

Physics:
  A PEC-bounded rectangular cavity of dimensions a x b x L supports
  TE_10p modes at frequencies:
    f_10p = (c/2) * sqrt((1/a)^2 + (p/L)^2)

  An iris (thin PEC wall with a centred aperture) divides the cavity
  and couples two half-sections.  For validation we measure the
  fundamental resonance of the full cavity and compare against the
  analytical TE_101 frequency.

Validation criteria:
  - Harminv-extracted resonance frequency within 3% of analytical TE_101
  - At least one clear mode detected by Harminv

Reference:
  Pozar, "Microwave Engineering", 4th ed., Ch 6 (Resonant Cavities)
  WR-90: a = 22.86 mm, b = 10.16 mm

Exit 0 on PASS, 1 on FAIL.
"""

import sys
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rfx import Simulation, Box, GaussianPulse, harminv
from rfx.grid import C0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = SCRIPT_DIR

THRESHOLD_PCT = 3.0  # max allowed frequency error


def main():
    t_start = time.time()

    # --- WR-90 cross-section ---
    a_wg = 22.86e-3   # broad wall (y)
    b_wg = 10.16e-3   # narrow wall (z)

    # Cavity length (x-direction) -- one half-wavelength at ~10 GHz
    L_cav = 30.0e-3

    # Analytical TE_101 resonance: f = (c/2) * sqrt((1/a)^2 + (1/L)^2)
    f_te101 = C0 / 2.0 * np.sqrt((1.0 / a_wg) ** 2 + (1.0 / L_cav) ** 2)

    # Iris parameters: thin PEC wall at cavity midpoint with centred aperture
    iris_aperture_y = a_wg * 0.4   # 40% of broad wall open
    iris_thickness_cells = 1       # single-cell thick

    dx = 1.0e-3
    f_max = 15e9  # simulation bandwidth

    print("=" * 60)
    print("GPU VALIDATION: PEC Cavity Resonance (Harminv)")
    print("=" * 60)
    print(f"Cavity         : {L_cav*1e3:.1f} x {a_wg*1e3:.2f} x {b_wg*1e3:.2f} mm")
    print(f"Analytical TE101: {f_te101/1e9:.4f} GHz")
    print(f"Iris aperture  : {iris_aperture_y*1e3:.1f} mm ({iris_aperture_y/a_wg*100:.0f}% of a)")
    print(f"Resolution     : dx = {dx*1e3:.1f} mm")
    print()

    # --- Build PEC cavity simulation ---
    sim = Simulation(
        freq_max=f_max,
        domain=(L_cav, a_wg, b_wg),
        boundary="pec",
        dx=dx,
    )

    # Iris: two PEC blocks on either side of the centred aperture at x = L/2
    iris_x0 = L_cav / 2 - dx / 2
    iris_x1 = L_cav / 2 + dx / 2
    gap_lo = (a_wg - iris_aperture_y) / 2
    gap_hi = (a_wg + iris_aperture_y) / 2

    # Lower PEC wall of iris
    sim.add(Box((iris_x0, 0, 0), (iris_x1, gap_lo, b_wg)), material="pec")
    # Upper PEC wall of iris
    sim.add(Box((iris_x0, gap_hi, 0), (iris_x1, a_wg, b_wg)), material="pec")

    # Excitation: off-centre to excite TE101 mode
    exc_x = L_cav * 0.25
    exc_y = a_wg / 2
    exc_z = b_wg / 2

    sim.add_port(
        (exc_x, exc_y, exc_z),
        component="ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=10e9, bandwidth=0.8),
    )
    sim.add_probe((exc_x, exc_y, exc_z), component="ez")

    # Second probe in the other half of the cavity
    sim.add_probe((L_cav * 0.75, exc_y, exc_z), component="ez")

    # --- Run simulation (long enough for ringdown) ---
    n_steps = 4000
    print(f"Running FDTD ({n_steps} steps) ...")
    result = sim.run(n_steps=n_steps, compute_s_params=False)

    ts = np.asarray(result.time_series)
    dt = float(result.dt)

    # --- Harminv resonance extraction ---
    print("Extracting resonances via Harminv ...")

    # Use probe 0 (excitation side) for ringdown analysis
    signal = ts[:, 0]

    # Skip the source excitation period (source decays after ~3*tau)
    source = GaussianPulse(f0=10e9, bandwidth=0.8)
    source_decay = source.t0 + 3 * source.tau
    start_idx = int(np.ceil(source_decay / dt))
    start_idx = min(start_idx, len(signal) - 100)

    modes = harminv(
        signal[start_idx:],
        dt,
        f_min=6e9,
        f_max=14e9,
        min_Q=5.0,
    )

    print(f"Found {len(modes)} mode(s):")
    for i, m in enumerate(modes):
        print(f"  Mode {i}: f = {m.freq/1e9:.4f} GHz, Q = {m.Q:.0f}, amp = {m.amplitude:.3e}")

    # --- Find the dominant mode closest to analytical ---
    f_harminv = None
    best_mode = None
    if modes:
        # Pick the strongest mode
        best_mode = modes[0]  # harminv returns sorted by amplitude
        f_harminv = best_mode.freq

    # --- Validation ---
    elapsed = time.time() - t_start

    if f_harminv is not None:
        freq_err = abs(f_harminv - f_te101) / f_te101 * 100
    else:
        freq_err = 100.0

    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Analytical TE101   : {f_te101/1e9:.4f} GHz")
    if f_harminv is not None:
        print(f"Harminv dominant   : {f_harminv/1e9:.4f} GHz")
        print(f"Frequency error    : {freq_err:.2f}%")
        if best_mode is not None:
            print(f"Mode Q factor      : {best_mode.Q:.0f}")
    else:
        print("Harminv            : no modes found")
    print(f"Threshold          : {THRESHOLD_PCT}% frequency error")
    print(f"Elapsed time       : {elapsed:.1f}s")

    # --- Figures ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("GPU Validation: PEC Cavity Resonance (Pozar Ch 6)", fontsize=14, fontweight="bold")

    # Panel 1: Time-domain probe signals
    ax = axes[0, 0]
    t_ns = np.arange(ts.shape[0]) * dt * 1e9
    ax.plot(t_ns, ts[:, 0], "b-", lw=0.5, alpha=0.8, label="Probe 0 (excitation side)")
    if ts.shape[1] >= 2:
        ax.plot(t_ns, ts[:, 1], "r-", lw=0.5, alpha=0.6, label="Probe 1 (far side)")
    ax.axvline(source_decay * 1e9, color="gray", ls=":", alpha=0.5, label="Source decay")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Ez (V/m)")
    ax.set_title("Time-Domain Ringdown")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: FFT spectrum with analytical marker
    ax = axes[0, 1]
    nfft = len(signal) * 4
    spectrum = np.abs(np.fft.rfft(signal[start_idx:], n=nfft))
    freqs_fft = np.fft.rfftfreq(nfft, d=dt)
    f_ghz = freqs_fft / 1e9
    spectrum_db = 20 * np.log10(np.maximum(spectrum / spectrum.max(), 1e-30))
    mask = (f_ghz > 5) & (f_ghz < 15)
    ax.plot(f_ghz[mask], spectrum_db[mask], "b-", lw=1.0, label="FFT (ringdown)")
    ax.axvline(f_te101 / 1e9, color="r", ls="--", lw=2, label=f"Analytical TE101 = {f_te101/1e9:.2f} GHz")
    if f_harminv is not None:
        ax.axvline(f_harminv / 1e9, color="g", ls=":", lw=2, label=f"Harminv = {f_harminv/1e9:.2f} GHz")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("Spectrum of Cavity Ringdown")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Harminv modes on spectrum
    ax = axes[1, 0]
    ax.plot(f_ghz[mask], spectrum_db[mask], "b-", lw=0.8, alpha=0.5)
    for i, m in enumerate(modes[:5]):
        ax.axvline(m.freq / 1e9, color="orange", ls="-", lw=1.5, alpha=0.7)
        ax.annotate(f"f={m.freq/1e9:.2f}\nQ={m.Q:.0f}",
                     xy=(m.freq / 1e9, 0), fontsize=7, color="orange",
                     ha="center", va="bottom")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("Harminv Mode Extraction")
    ax.grid(True, alpha=0.3)

    # Panel 4: Summary
    ax = axes[1, 1]
    ax.axis("off")
    verdict = "PASS" if freq_err < THRESHOLD_PCT else "FAIL"
    lines = [
        "PEC Cavity Resonance Validation",
        "-" * 38,
        f"Cavity: {L_cav*1e3:.1f} x {a_wg*1e3:.2f} x {b_wg*1e3:.2f} mm",
        f"Iris aperture: {iris_aperture_y*1e3:.1f} mm",
        f"dx = {dx*1e3:.1f} mm, n_steps = {n_steps}",
        "",
        f"Analytical TE101 : {f_te101/1e9:.4f} GHz",
        f"Harminv f_dom    : {f_harminv/1e9:.4f} GHz" if f_harminv else "Harminv f_dom    : N/A",
        f"Frequency error  : {freq_err:.2f}%",
    ]
    if best_mode is not None:
        lines.append(f"Q factor         : {best_mode.Q:.0f}")
    lines.extend([
        "",
        f"Criterion: freq error < {THRESHOLD_PCT}%",
        f"Verdict: {verdict}",
        f"Time: {elapsed:.1f}s",
    ])
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes, va="top",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "02_filter_validation.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved: {out_path}")

    # --- Pass/Fail ---
    passed = freq_err < THRESHOLD_PCT
    if passed:
        print(f"\nPASS: Cavity resonance error {freq_err:.2f}% < {THRESHOLD_PCT}%")
        sys.exit(0)
    else:
        print(f"\nFAIL: Cavity resonance error {freq_err:.2f}% >= {THRESHOLD_PCT}%")
        sys.exit(1)


if __name__ == "__main__":
    main()
