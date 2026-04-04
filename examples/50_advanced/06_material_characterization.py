"""Example: S-parameter Material Characterization

Demonstrates the material characterization workflow:
  1. Define a known dispersive material (water at 20C, single Debye pole)
  2. Generate synthetic complex permittivity data across frequency
  3. Use rfx's fit_debye to recover the Debye poles from the data
  4. Simulate the original and fitted materials to compare S-parameters
  5. Visualize: eps(f) comparison, fit quality, S11, and pole summary

This showcases rfx's material fitting and dispersive simulation
capabilities for extracting material models from measured data.

Saves: examples/50_advanced/06_material_characterization.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rfx import Simulation, Box, GaussianPulse, DebyePole
from rfx.material_fit import fit_debye, eval_debye, plot_material_fit

OUT_DIR = "examples/50_advanced"


def build_slab_sim(eps_inf, debye_poles):
    """Build a transmission-line fixture with a dielectric slab (DUT).

    Parameters
    ----------
    eps_inf : float
        High-frequency permittivity.
    debye_poles : list of DebyePole
        Debye relaxation poles.
    """
    f_max = 10e9
    dx = 1.5e-3
    dom_x = 0.03
    dom_y = 0.01
    dom_z = 0.01

    sim = Simulation(
        freq_max=f_max,
        domain=(dom_x, dom_y, dom_z),
        boundary="pec",
        dx=dx,
    )

    sim.add_material("dut", eps_r=eps_inf, debye_poles=debye_poles)
    sim.add(Box((0.01, 0, 0), (0.02, dom_y, dom_z)), material="dut")

    sim.add_port(
        (0.004, dom_y / 2, dom_z / 2),
        component="ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=f_max / 2, bandwidth=0.8),
    )
    sim.add_probe((0.004, dom_y / 2, dom_z / 2), component="ez")

    sim.add_probe((0.026, dom_y / 2, dom_z / 2), component="ez")

    return sim


def debye_permittivity(freqs, eps_inf, poles):
    """Compute complex permittivity from Debye model analytically."""
    omega = 2 * np.pi * freqs
    eps = np.full_like(freqs, eps_inf, dtype=complex)
    for pole in poles:
        eps += pole.delta_eps / (1.0 + 1j * omega * pole.tau)
    return eps


def main():
    # ---- Ground truth: water at 20C ----
    eps_inf_true = 4.9
    true_poles = [DebyePole(delta_eps=74.1, tau=8.3e-12)]

    print("Ground truth: Water at 20C (single Debye pole)")
    print(f"  eps_inf     = {eps_inf_true}")
    print(f"  delta_eps   = {true_poles[0].delta_eps}")
    print(f"  tau         = {true_poles[0].tau * 1e12:.1f} ps")

    # ---- Generate synthetic permittivity data ----
    freqs = np.linspace(0.1e9, 10e9, 100)
    eps_true = debye_permittivity(freqs, eps_inf_true, true_poles)

    # Add realistic measurement noise
    rng = np.random.default_rng(42)
    noise_level = 0.02  # 2% relative noise
    eps_noisy = eps_true * (1 + noise_level * (rng.standard_normal(len(freqs)) +
                                                1j * rng.standard_normal(len(freqs))))

    print(f"\nSynthetic data: {len(freqs)} frequency points, {freqs[0]/1e9:.1f}-{freqs[-1]/1e9:.1f} GHz")
    print(f"  eps' range: {eps_noisy.real.min():.1f} to {eps_noisy.real.max():.1f}")
    print(f"  eps'' range: {eps_noisy.imag.min():.1f} to {eps_noisy.imag.max():.1f}")

    # ---- Fit Debye model ----
    print("\nFitting 1-pole Debye model ...")
    fit_1 = fit_debye(freqs, eps_noisy, n_poles=1)

    print(f"  eps_inf   = {fit_1.eps_inf:.3f}  (true: {eps_inf_true})")
    print(f"  delta_eps = {fit_1.poles[0].delta_eps:.3f}  (true: {true_poles[0].delta_eps})")
    print(f"  tau       = {fit_1.poles[0].tau * 1e12:.3f} ps  (true: {true_poles[0].tau * 1e12:.1f} ps)")
    print(f"  Fit error = {fit_1.fit_error * 100:.3f}%")

    # Also try 2-pole fit
    print("\nFitting 2-pole Debye model ...")
    fit_2 = fit_debye(freqs, eps_noisy, n_poles=2)
    print(f"  Fit error = {fit_2.fit_error * 100:.3f}%")
    for i, p in enumerate(fit_2.poles):
        print(f"  Pole {i+1}: delta_eps={p.delta_eps:.3f}, tau={p.tau*1e12:.3f} ps")

    # ---- Evaluate fitted models ----
    freq_dense = np.linspace(0.1e9, 10e9, 500)
    eps_true_dense = debye_permittivity(freq_dense, eps_inf_true, true_poles)
    eps_fit1 = eval_debye(freq_dense, fit_1.eps_inf, fit_1.poles)
    eps_fit2 = eval_debye(freq_dense, fit_2.eps_inf, fit_2.poles)

    # ---- Run FDTD simulations with true vs fitted material ----
    print("\nRunning FDTD with true material ...")
    sim_true = build_slab_sim(eps_inf_true, true_poles)
    result_true = sim_true.run(n_steps=600, compute_s_params=True)

    print("Running FDTD with 1-pole fitted material ...")
    sim_fit = build_slab_sim(fit_1.eps_inf, list(fit_1.poles))
    result_fit = sim_fit.run(n_steps=600, compute_s_params=True)

    # ---- Results summary ----
    eps_inf_err = abs(fit_1.eps_inf - eps_inf_true) / eps_inf_true * 100
    de_err = abs(fit_1.poles[0].delta_eps - true_poles[0].delta_eps) / true_poles[0].delta_eps * 100
    tau_err = abs(fit_1.poles[0].tau - true_poles[0].tau) / true_poles[0].tau * 100

    print(f"\n{'='*60}")
    print(f"Material Characterization Results (1-pole Debye)")
    print(f"{'='*60}")
    print(f"{'Parameter':<15s}  {'True':>12s}  {'Recovered':>12s}  {'Error':>8s}")
    print("-" * 52)
    print(f"{'eps_inf':<15s}  {eps_inf_true:12.3f}  {fit_1.eps_inf:12.3f}  {eps_inf_err:7.2f}%")
    print(f"{'delta_eps':<15s}  {true_poles[0].delta_eps:12.3f}  {fit_1.poles[0].delta_eps:12.3f}  {de_err:7.2f}%")
    print(f"{'tau (ps)':<15s}  {true_poles[0].tau*1e12:12.3f}  {fit_1.poles[0].tau*1e12:12.3f}  {tau_err:7.2f}%")
    print(f"{'Fit error':<15s}  {'':>12s}  {fit_1.fit_error*100:11.3f}%")

    # ---- 6-panel figure ----
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle("Material Characterization: Debye Pole Recovery from Noisy Data",
                 fontsize=14, fontweight="bold")
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)

    # Panel 1: Real part of permittivity
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(freq_dense / 1e9, eps_true_dense.real, "b-", lw=2, label="True")
    ax.plot(freq_dense / 1e9, eps_fit1.real, "r--", lw=2, label="1-pole fit")
    ax.plot(freq_dense / 1e9, eps_fit2.real, "g:", lw=1.5, label="2-pole fit")
    ax.plot(freqs / 1e9, eps_noisy.real, "k.", ms=2, alpha=0.4, label="Noisy data")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("eps' (real)")
    ax.set_title("Permittivity: Real Part")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Imaginary part (loss)
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(freq_dense / 1e9, -eps_true_dense.imag, "b-", lw=2, label="True")
    ax.plot(freq_dense / 1e9, -eps_fit1.imag, "r--", lw=2, label="1-pole fit")
    ax.plot(freq_dense / 1e9, -eps_fit2.imag, "g:", lw=1.5, label="2-pole fit")
    ax.plot(freqs / 1e9, -eps_noisy.imag, "k.", ms=2, alpha=0.4, label="Noisy data")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("-eps'' (loss)")
    ax.set_title("Permittivity: Loss Factor")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Fit residuals
    ax = fig.add_subplot(gs[0, 2])
    resid_1 = np.abs(eps_fit1[:len(freqs)] - eps_true[:len(freqs)]) if len(eps_fit1) >= len(freqs) else np.zeros(1)
    eps_eval1_at_data = eval_debye(freqs, fit_1.eps_inf, fit_1.poles)
    eps_eval2_at_data = eval_debye(freqs, fit_2.eps_inf, fit_2.poles)
    resid_1 = np.abs(eps_eval1_at_data - eps_true) / np.abs(eps_true)
    resid_2 = np.abs(eps_eval2_at_data - eps_true) / np.abs(eps_true)
    ax.semilogy(freqs / 1e9, resid_1 * 100, "r-", lw=1.5, label=f"1-pole ({fit_1.fit_error*100:.2f}%)")
    ax.semilogy(freqs / 1e9, resid_2 * 100, "g-", lw=1.5, label=f"2-pole ({fit_2.fit_error*100:.2f}%)")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Relative error (%)")
    ax.set_title("Fit Residuals vs Ground Truth")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: FDTD S11 comparison
    ax = fig.add_subplot(gs[1, 0])
    if result_true.s_params is not None and result_true.freqs is not None:
        f_ghz = np.asarray(result_true.freqs) / 1e9
        s11_true = 20 * np.log10(np.maximum(np.abs(np.asarray(result_true.s_params)[0, 0, :]), 1e-30))
        ax.plot(f_ghz, s11_true, "b-", lw=1.5, label="True material")
    if result_fit.s_params is not None and result_fit.freqs is not None:
        f_ghz2 = np.asarray(result_fit.freqs) / 1e9
        s11_fit = 20 * np.log10(np.maximum(np.abs(np.asarray(result_fit.s_params)[0, 0, :]), 1e-30))
        ax.plot(f_ghz2, s11_fit, "r--", lw=1.5, label="Fitted material")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("|S11| (dB)")
    ax.set_title("FDTD S11: True vs Fitted Material")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 5: Time-domain comparison
    ax = fig.add_subplot(gs[1, 1])
    ts_true = np.asarray(result_true.time_series)
    ts_fit = np.asarray(result_fit.time_series)
    dt = result_true.dt
    t_ns = np.arange(ts_true.shape[0]) * dt * 1e9
    ax.plot(t_ns, ts_true[:, 0], "b-", lw=0.8, alpha=0.8, label="True")
    ax.plot(t_ns, ts_fit[:, 0], "r--", lw=0.8, alpha=0.8, label="Fitted")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Ez at port")
    ax.set_title("Time-Domain Port Signal")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 6: Summary table
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    lines = [
        "Material Characterization Summary",
        "-" * 38,
        f"Material: Water at 20C",
        f"Model: Single Debye pole",
        "",
        f"{'Param':<12s} {'True':>10s} {'Fit':>10s} {'Err':>7s}",
        "-" * 38,
        f"{'eps_inf':<12s} {eps_inf_true:10.2f} {fit_1.eps_inf:10.2f} {eps_inf_err:6.1f}%",
        f"{'delta_eps':<12s} {true_poles[0].delta_eps:10.2f} {fit_1.poles[0].delta_eps:10.2f} {de_err:6.1f}%",
        f"{'tau (ps)':<12s} {true_poles[0].tau*1e12:10.2f} {fit_1.poles[0].tau*1e12:10.2f} {tau_err:6.1f}%",
        "-" * 38,
        f"Fit error (RMS): {fit_1.fit_error*100:.3f}%",
        f"Noise level: {noise_level*100:.0f}%",
        "",
        f"2-pole fit error: {fit_2.fit_error*100:.3f}%",
    ]
    ax.text(0.05, 0.95, "\n".join(lines),
            transform=ax.transAxes, va="top", fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))

    out_path = f"{OUT_DIR}/06_material_characterization.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved: {out_path}")


if __name__ == "__main__":
    main()
