"""Cross-validation: Debye Water Dispersion (Analytical)

Validates Debye dispersive material model against exact analytical Fresnel.

Setup: TFSF plane wave (Ez, +x) incident on a Debye water half-space.
Two runs with identical TFSF and probe layout:
  1. Vacuum: total-field probe → incident field E_inc(t)
  2. Water:  same probe → total field E_inc(t) + E_refl(t)
Reference subtraction: E_refl(t) = E_total(t) - E_inc(t)
R(f) = |FFT(E_refl)| / |FFT(E_inc)|

Debye model for water at 25C:
  epsilon(f) = eps_inf + (eps_s - eps_inf) / (1 + j*2*pi*f*tau)
  eps_s = 78.36, eps_inf = 5.2, tau = 8.27 ps

Fresnel normal incidence (amplitude):
  R(f) = |(sqrt(eps(f)) - 1) / (sqrt(eps(f)) + 1)|

PASS criteria:
  - |R(f)| matches analytical within 10% relative RMS (6-14 GHz)
  - Low-freq reflection > high-freq reflection (dispersion visible)

Save: examples/crossval/15_debye_water_dispersion.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from rfx import Simulation, Box
from rfx.materials.debye import DebyePole

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

C0 = 2.998e8

# Debye water parameters (25C, single pole)
eps_s = 78.36    # static permittivity
eps_inf = 5.2    # infinite frequency permittivity
tau = 8.27e-12   # relaxation time (8.27 ps)
f_relax = 1 / (2 * np.pi * tau)  # ~19.2 GHz

f0 = 10e9   # center frequency
f_max = 25e9

print("=" * 60)
print("Cross-Validation: Debye Water Dispersion")
print("=" * 60)
print(f"Debye: eps_s={eps_s}, eps_inf={eps_inf}, tau={tau*1e12:.2f} ps")
print(f"Relaxation freq: {f_relax/1e9:.1f} GHz")
print()

# Analytical Fresnel reflectance R(f) — amplitude coefficient
freqs_ana = np.linspace(1e9, f_max, 500)
omega_ana = 2 * np.pi * freqs_ana
eps_debye_ana = eps_inf + (eps_s - eps_inf) / (1 + 1j * omega_ana * tau)
n_debye_ana = np.sqrt(eps_debye_ana)
R_fresnel_amp = np.abs((n_debye_ana - 1) / (n_debye_ana + 1))

# Mesh
dx = C0 / f_max / 20  # ~0.6 mm
cpml_n = 8
tfsf_margin = 15  # cells between CPML edge and TFSF box

# Domain: long x, narrow y/z
dom_x = 200e-3   # 200 mm
dom_yz = (2 * cpml_n + 6) * dx  # CPML both sides + 6 interior
dom_y = dom_yz
dom_z = dom_yz

# TFSF box boundaries in physical coords
tfsf_x_lo_phys = tfsf_margin * dx      # ~9 mm
tfsf_x_hi_phys = dom_x - tfsf_margin * dx

# Water half-space — inside total-field region, clear of TFSF boundaries
interface_x = dom_x * 0.5  # 100 mm
water_end = tfsf_x_hi_phys - 10 * dx

# Probe: in total-field region, between TFSF boundary and water interface
# Far enough from interface for time-domain separation of incident/reflected
probe_x = tfsf_x_lo_phys + 15 * dx  # ~18 mm from x=0

# Time for round trip probe → interface → probe
round_trip = 2 * (interface_x - probe_x) / C0
print(f"Round-trip time: {round_trip*1e9:.2f} ns")

n_steps = 3000
total_time = n_steps * dx / C0 / np.sqrt(3)  # approximate
print(f"Approx sim time: {total_time*1e9:.1f} ns")

print(f"Mesh: dx={dx*1e6:.0f} um, domain={dom_x*1e3:.0f}x{dom_y*1e3:.1f}x{dom_z*1e3:.1f} mm")
print(f"TFSF box: x=[{tfsf_x_lo_phys*1e3:.1f}, {tfsf_x_hi_phys*1e3:.1f}] mm")
print(f"Water: x=[{interface_x*1e3:.0f}, {water_end*1e3:.1f}] mm")
print(f"Probe: x={probe_x*1e3:.1f} mm (total-field region)")
print()


def run_sim(with_water: bool):
    """Run TFSF simulation, return probe time series and dt."""
    sim = Simulation(
        freq_max=f_max * 1.5,
        domain=(dom_x, dom_y, dom_z),
        dx=dx,
        boundary="cpml",
        cpml_layers=cpml_n,
    )

    sim.add_tfsf_source(
        f0=f0,
        bandwidth=0.7,
        polarization="ez",
        direction="+x",
        margin=tfsf_margin,
    )

    if with_water:
        debye_pole = DebyePole(delta_eps=eps_s - eps_inf, tau=tau)
        sim.add_material("water", eps_r=eps_inf, debye_poles=[debye_pole])
        # Oversized y/z extent to fill CPML padding and prevent edge diffraction
        yz_pad = 0.1  # 100 mm extra — Box is clipped to grid automatically
        sim.add(Box(
            (interface_x, -yz_pad, -yz_pad),
            (water_end, dom_y + yz_pad, dom_z + yz_pad),
        ), material="water")

    sim.add_probe(position=(probe_x, dom_y / 2, dom_z / 2), component="ez")

    result = sim.run(n_steps=n_steps)
    return np.array(result.time_series).ravel(), result.dt


# Run 1: vacuum (incident only at total-field probe)
print("Run 1: vacuum...")
t0 = time.time()
sig_vac, dt = run_sim(with_water=False)
print(f"  {time.time() - t0:.1f}s")

# Run 2: water (incident + reflected at same total-field probe)
print("Run 2: Debye water...")
t0 = time.time()
sig_water, _ = run_sim(with_water=True)
print(f"  {time.time() - t0:.1f}s")

# Reference subtraction: reflected = total - incident
sig_refl = sig_water - sig_vac

# Diagnostics
print(f"\nDiagnostics:")
print(f"  Incident peak: {np.max(np.abs(sig_vac)):.6f} at step {np.argmax(np.abs(sig_vac))}")
print(f"  Total peak: {np.max(np.abs(sig_water)):.6f} at step {np.argmax(np.abs(sig_water))}")
print(f"  Reflected peak: {np.max(np.abs(sig_refl)):.6f} at step {np.argmax(np.abs(sig_refl))}")
print(f"  Peak ratio (refl/inc): {np.max(np.abs(sig_refl)) / (np.max(np.abs(sig_vac)) + 1e-30):.4f}")

# Time-gated FFT — gate incident and reflected separately.
# Find pulse locations from peak indices
inc_peak_step = np.argmax(np.abs(sig_vac))
refl_peak_step = np.argmax(np.abs(sig_refl))
pulse_half_width = 300  # generous half-width in steps

# Gate around each pulse
inc_lo = max(0, inc_peak_step - pulse_half_width)
inc_hi = min(len(sig_vac), inc_peak_step + pulse_half_width)
refl_lo = max(0, refl_peak_step - pulse_half_width)
refl_hi = min(len(sig_refl), refl_peak_step + pulse_half_width)

sig_inc_gated = sig_vac[inc_lo:inc_hi]
sig_refl_gated = sig_refl[refl_lo:refl_hi]

print(f"  Incident gate: steps {inc_lo}-{inc_hi} ({len(sig_inc_gated)} samples)")
print(f"  Reflected gate: steps {refl_lo}-{refl_hi} ({len(sig_refl_gated)} samples)")

# Zero-pad both to same power-of-2 length for consistent frequency resolution
n_fft = 2 ** int(np.ceil(np.log2(max(len(sig_inc_gated), len(sig_refl_gated)) * 4)))
S_inc = np.fft.rfft(sig_inc_gated, n=n_fft)
S_refl = np.fft.rfft(sig_refl_gated, n=n_fft)
freqs_fft = np.fft.rfftfreq(n_fft, d=dt)

# |R(f)| — only where source has significant energy
S_inc_abs = np.abs(S_inc)
S_inc_peak = np.max(S_inc_abs)
valid_spec = S_inc_abs > 0.05 * S_inc_peak
R_sim = np.where(valid_spec, np.abs(S_refl) / S_inc_abs, np.nan)

# R at key frequencies
for f_check in [6e9, 8e9, 10e9, 12e9, 14e9]:
    idx = np.argmin(np.abs(freqs_fft - f_check))
    r_val = R_sim[idx] if valid_spec[idx] else float("nan")
    r_ana = np.interp(f_check, freqs_ana, R_fresnel_amp)
    print(f"  R({f_check/1e9:.0f} GHz) = {r_val:.3f} (analytical: {r_ana:.3f})")

# Compare in source bandwidth (6-14 GHz ≈ f0 ± BW/2 * f0)
f_lo, f_hi = 6e9, 14e9
f_mask = (freqs_fft > f_lo) & (freqs_fft < f_hi) & valid_spec
freqs_comp = freqs_fft[f_mask]
R_sim_comp = R_sim[f_mask]

# Interpolate analytical to comparison frequencies
R_ana_interp = np.interp(freqs_comp, freqs_ana, R_fresnel_amp)

# Relative RMS error
if len(R_sim_comp) > 0:
    rms_err = np.sqrt(np.mean((R_sim_comp - R_ana_interp) ** 2))
    rms_rel = rms_err / (np.mean(R_ana_interp) + 1e-30)
else:
    rms_err = rms_rel = 1.0

print(f"\nResults:")
print(f"  Frequency points compared: {len(R_sim_comp)}")
print(f"  RMS error (amplitude): {rms_err:.4f}")
print(f"  Relative RMS error: {rms_rel*100:.1f}%")

PASS = True

if rms_rel > 0.10:
    print(f"  FAIL: Relative RMS {rms_rel*100:.1f}% > 10%")
    PASS = False
else:
    print(f"  PASS: shape agreement within 10% relative RMS")

# Dispersion check: low-freq R > high-freq R
if len(R_sim_comp) > 10:
    n_fifth = max(1, len(R_sim_comp) // 5)
    R_lo = np.mean(R_sim_comp[:n_fifth])
    R_hi = np.mean(R_sim_comp[-n_fifth:])
    if R_lo > R_hi:
        print(f"  PASS: dispersion visible (R_low={R_lo:.3f} > R_high={R_hi:.3f})")
    else:
        print(f"  FAIL: no dispersion (R_low={R_lo:.3f} <= R_high={R_hi:.3f})")
        PASS = False

    # Absolute magnitude check (water at low freq: R ~ 0.65-0.80)
    if R_lo > 0.4:
        print(f"  PASS: low-freq |R| = {R_lo:.3f} (expected ~0.6-0.8)")
    else:
        print(f"  FAIL: low-freq |R| = {R_lo:.3f} too low (expected > 0.4)")
        PASS = False

# Plot
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Debye Water: rfx TFSF vs Analytical Fresnel", fontsize=14)

# 1. |R(f)|
ax1 = axes[0]
ax1.plot(freqs_ana / 1e9, R_fresnel_amp, "k-", label="Analytical", linewidth=2)
if len(freqs_comp) > 0:
    ax1.plot(freqs_comp / 1e9, R_sim_comp, "b-", alpha=0.7,
             label="rfx FDTD", linewidth=1.5)
    ax1.plot(freqs_comp / 1e9, R_ana_interp, "k--", alpha=0.3)
ax1.set_xlabel("Frequency (GHz)")
ax1.set_ylabel("|R(f)|")
ax1.set_xlim(1, 20)
ax1.set_ylim(0, 1)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_title(f"|R(f)| — rel. RMS: {rms_rel*100:.1f}%")

# 2. Debye permittivity
ax2 = axes[1]
ax2.plot(freqs_ana / 1e9, np.real(eps_debye_ana), "b-", label="Re(eps)", linewidth=2)
ax2.plot(freqs_ana / 1e9, -np.imag(eps_debye_ana), "r-", label="-Im(eps)", linewidth=2)
ax2.axvline(f_relax / 1e9, color="k", ls="--", alpha=0.5,
            label=f"f_relax = {f_relax/1e9:.1f} GHz")
ax2.set_xlabel("Frequency (GHz)")
ax2.set_ylabel("Permittivity")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_title("Debye epsilon(f) — Water 25C")

# 3. Time-domain signals
ax3 = axes[2]
t_ns = np.arange(len(sig_vac)) * dt * 1e9
ax3.plot(t_ns, sig_vac, "k-", linewidth=0.5, label="Vacuum (incident)")
ax3.plot(t_ns, sig_water, "b-", linewidth=0.5, alpha=0.7, label="Water (total)")
ax3.plot(t_ns, sig_refl, "r-", linewidth=1, label="Reflected (subtracted)")
ax3.set_xlabel("Time (ns)")
ax3.set_ylabel("Ez")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.set_title("Time-domain at probe")

plt.tight_layout()
out_path = os.path.join(SCRIPT_DIR, "15_debye_water_dispersion.png")
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved: {out_path}")

if PASS:
    print("\nALL CHECKS PASSED")
else:
    print("\nSOME CHECKS FAILED")
