"""Accuracy Validation Case 6: Patch Antenna with Auto Mesh (P1-P3)

Same geometry as case 01 (Balanis/OpenEMS patch), but uses:
- dx=None (P1 auto mesh from geometry + materials)
- P3 thirds rule at substrate-air interface (automatic in dz_profile)
- P2 smooth grading (automatic)

Compares resonance accuracy and grid efficiency vs fixed dx=2mm.
"""

import sys
import os
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rfx import Simulation, Box
from rfx.sources.sources import GaussianPulse
from rfx.grid import C0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
THRESHOLD_PCT = 5.0

# OpenEMS tutorial parameters (same as case 01)
f0 = 2e9
fc = 1e9
eps_r = 3.38
substrate_h = 1.524e-3  # 60 mil
patch_w = 32.86e-3
patch_l = 41.37e-3
feed_x = 11.18e-3  # inset feed
ground_w = 60e-3
ground_l = 60e-3

# Analytical patch resonance
f_analytical = C0 / (2 * patch_l * np.sqrt(eps_r))

print("=" * 60)
print("Accuracy Case 6: Patch Antenna — Auto Mesh (P1-P3)")
print("=" * 60)
print(f"Substrate: eps_r={eps_r}, h={substrate_h*1e3:.3f}mm")
print(f"Patch: {patch_w*1e3:.1f} x {patch_l*1e3:.1f} mm")
print(f"Analytical f_r: {f_analytical/1e9:.3f} GHz")
print()

# Domain with margin
margin = 20e-3
domain_x = ground_l + 2 * margin
domain_y = ground_w + 2 * margin
domain_z = substrate_h + 30e-3

# Auto mesh: dx=None, let P1 auto-configure from geometry + materials
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    sim = Simulation(
        freq_max=f0 + fc,
        domain=(domain_x, domain_y, domain_z),
        boundary="cpml",
        cpml_layers=8,
    )

sim.add_material("substrate", eps_r=eps_r)
sim.add_material("pec_sheet", sigma=5.8e7)

# Ground plane (PEC, z=0)
sim.add(Box((margin, margin, 0), (margin + ground_l, margin + ground_w, 0.001)),
        material="pec_sheet")

# Substrate
sim.add(Box((margin, margin, 0), (margin + ground_l, margin + ground_w, substrate_h)),
        material="substrate")

# Patch (PEC, z=substrate_h)
patch_x0 = margin + (ground_l - patch_l) / 2
patch_y0 = margin + (ground_w - patch_w) / 2
sim.add(Box((patch_x0, patch_y0, substrate_h),
            (patch_x0 + patch_l, patch_y0 + patch_w, substrate_h + 0.001)),
        material="pec_sheet")

# Lumped port at inset feed
port_x = patch_x0 + feed_x
port_y = margin + ground_w / 2
sim.add_port(
    (port_x, port_y, substrate_h / 2), "ez",
    impedance=50,
    waveform=GaussianPulse(f0=f0, bandwidth=fc / f0),
)
sim.add_probe((port_x, port_y, substrate_h / 2), "ez")

# Print auto-configured mesh info
dx_auto = sim._dx
print(f"Auto mesh dx: {dx_auto}")

grid = sim._build_grid()
n_steps = grid.num_timesteps(num_periods=30)
print(f"Grid: {grid.nx}x{grid.ny}x{grid.nz} = {grid.nx*grid.ny*grid.nz/1e6:.2f}M cells")
print(f"Steps: {n_steps}")
print(f"dx = {grid.dx*1e3:.3f} mm")
print()

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    result = sim.run(n_steps=n_steps)

# Find resonance
modes = result.find_resonances(freq_range=(f_analytical * 0.5, f_analytical * 1.5))
if modes:
    f_sim = min(modes, key=lambda m: abs(m.freq - f_analytical)).freq
    err_pct = abs(f_sim - f_analytical) / f_analytical * 100
    print(f"Resonance: {f_sim/1e9:.4f} GHz")
    print(f"Analytical: {f_analytical/1e9:.4f} GHz")
    print(f"Error: {err_pct:.2f}%")

    if err_pct < THRESHOLD_PCT:
        print(f"PASS: {err_pct:.2f}% < {THRESHOLD_PCT}%")
    else:
        print(f"FAIL: {err_pct:.2f}% > {THRESHOLD_PCT}%")
        sys.exit(1)
else:
    print("FAIL: no resonance found")
    sys.exit(1)

# Compare with fixed mesh (case 01 result: 1.39% at dx=2mm)
print("\nComparison with fixed dx=2mm (case 01): 1.39% error")
print(f"Auto mesh (P1-P3): {err_pct:.2f}% error")
if err_pct < 1.39:
    print("IMPROVEMENT: auto mesh is more accurate")
else:
    print("NOTE: auto mesh comparable or needs finer default")

# Plot
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ts = np.array(result.time_series).ravel()
nfft = len(ts) * 4
spec = np.abs(np.fft.rfft(ts, n=nfft))
freqs = np.fft.rfftfreq(nfft, d=result.dt) / 1e9
band = (freqs > 1.0) & (freqs < 3.5)
ax.plot(freqs[band], 20 * np.log10(spec[band] / np.max(spec[band]) + 1e-30), "b-")
ax.axvline(f_analytical / 1e9, color="r", ls="--", alpha=0.5,
           label=f"Analytical {f_analytical/1e9:.3f} GHz")
if modes:
    ax.axvline(f_sim / 1e9, color="g", ls=":", alpha=0.5,
               label=f"rfx {f_sim/1e9:.3f} GHz ({err_pct:.2f}%)")
ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Relative Power (dB)")
ax.set_title(f"Patch Antenna — Auto Mesh (dx={grid.dx*1e3:.2f}mm)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(-30, 3)

plt.tight_layout()
out = os.path.join(SCRIPT_DIR, "06_patch_auto_mesh.png")
plt.savefig(out, dpi=150)
plt.close(fig)
print(f"\nPlot saved: {out}")
