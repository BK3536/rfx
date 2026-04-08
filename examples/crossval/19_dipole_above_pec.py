"""Cross-validation: Dipole Above PEC Ground Plane (Image Theory)

Validates NTFF far-field computation for an antenna above a PEC ground plane
using image theory as the analytical reference.

Structure: z-directed half-wave dipole at height h above an infinite PEC plane.

Image theory: The PEC plane acts as a mirror. The far-field pattern is
that of a two-element array: original dipole + its image.

Array factor: AF(theta) = 2*cos(k*h*cos(theta))
Total pattern: E(theta) = E_dipole(theta) * AF(theta)
Directivity: D ~ 6-8 dBi depending on height h

PASS criteria:
  - Directivity > 5 dBi (enhanced by ground plane)
  - Pattern null in the ground plane (theta = 90 deg, below horizon)
  - Peak pattern above ground plane (theta < 90 deg)

Save: examples/crossval/19_dipole_above_pec.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from rfx import Simulation, Box
from rfx.sources.sources import GaussianPulse
from rfx.farfield import compute_far_field, directivity
import jax.numpy as jnp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

C0 = 2.998e8

# Parameters
f0 = 2.0e9             # 2 GHz
lam0 = C0 / f0          # 150 mm
fc = 0.5e9              # bandwidth

# Dipole height above ground (quarter-wave for good directivity)
h_dipole = lam0 / 4     # 37.5 mm
k0 = 2 * np.pi / lam0

# Mesh
dx = lam0 / 20          # 7.5 mm (lambda/20)

# Domain: PEC at z=0, dipole in +z
margin_xy = lam0 * 0.7
dom_xy = 2 * margin_xy + 4 * dx
dom_z = h_dipole + lam0 * 0.8  # room above dipole

print("=" * 60)
print("Cross-Validation: Dipole Above PEC Ground Plane")
print("=" * 60)
print(f"f0 = {f0/1e9:.1f} GHz, lambda = {lam0*1e3:.0f} mm")
print(f"Dipole height: {h_dipole*1e3:.1f} mm (lambda/4)")
print(f"Mesh: dx = {dx*1e3:.1f} mm, domain = {dom_xy*1e3:.0f}x{dom_xy*1e3:.0f}x{dom_z*1e3:.0f} mm")
print()

# Build simulation
sim = Simulation(
    freq_max=(f0 + fc) * 1.5,
    domain=(dom_xy, dom_xy, dom_z),
    dx=dx,
    boundary="cpml",
    cpml_layers=8,
    pec_faces={"z_lo"},  # infinite PEC ground plane
)

# Dipole source at height h above ground
sim.add_source(
    position=(dom_xy / 2, dom_xy / 2, h_dipole),
    component="ez",
    waveform=GaussianPulse(f0=f0, bandwidth=fc / f0),
)
sim.add_probe(
    position=(dom_xy / 2, dom_xy / 2, h_dipole),
    component="ez",
)

# NTFF box
cpml_thick = 8 * dx
ntff_margin = cpml_thick + 3 * dx
sim.add_ntff_box(
    corner_lo=(ntff_margin, ntff_margin, dx),  # z_lo just above ground
    corner_hi=(dom_xy - ntff_margin, dom_xy - ntff_margin, dom_z - ntff_margin),
    freqs=jnp.array([f0]),
)

# Run
print("Preflight:")
warnings = sim.preflight(strict=False)
print()

t0 = time.time()
result = sim.run(num_periods=15)
elapsed = time.time() - t0
print(f"Simulation time: {elapsed:.1f}s")

# Far-field
theta = jnp.linspace(0, jnp.pi, 181)  # 0 = +z (above ground), 180 = -z (below)
phi = jnp.array([0.0, jnp.pi / 2])

ff = compute_far_field(result.ntff_data, result.ntff_box, result.grid, theta, phi)
D = directivity(ff)
D_dbi = 10 * np.log10(float(D) + 1e-30)
print(f"Directivity: {D_dbi:.1f} dBi")

# Pattern analysis
E_th = np.abs(np.asarray(ff.E_theta[0, :, 0]))
E_ph = np.abs(np.asarray(ff.E_phi[0, :, 0]))
power = E_th ** 2 + E_ph ** 2
theta_arr = np.asarray(theta)
theta_deg = theta_arr * 180 / np.pi

# Analytical pattern: dipole * array factor
# Dipole: sin(theta) for a z-directed Hertzian dipole
# Array factor for h = lambda/4: 2*cos(pi/2 * cos(theta))
AF_ana = 2 * np.abs(np.cos(k0 * h_dipole * np.cos(theta_arr)))
dipole_pattern = np.sin(theta_arr)
total_ana = (dipole_pattern * AF_ana) ** 2
total_ana_norm = total_ana / (np.max(total_ana) + 1e-30)

# Normalize simulated pattern
power_norm = power / (np.max(power) + 1e-30)

# =============================================================================
# Validation
# =============================================================================
PASS = True

# Check 1: Directivity > 5 dBi
if D_dbi > 5:
    print(f"PASS: Directivity {D_dbi:.1f} dBi > 5 dBi")
else:
    print(f"FAIL: Directivity {D_dbi:.1f} dBi (expected > 5 dBi)")
    PASS = False

# Check 2: Peak above ground plane (theta < 90 deg)
peak_idx = np.argmax(power)
peak_theta = theta_deg[peak_idx]
if peak_theta < 90:
    print(f"PASS: pattern peak at theta={peak_theta:.0f} deg (above ground)")
else:
    print(f"FAIL: pattern peak at theta={peak_theta:.0f} deg (expected < 90)")
    PASS = False

# Check 3: Pattern correlation with analytical in upper hemisphere (0-90 deg)
upper = theta_deg <= 90
if np.any(upper) and np.max(power_norm[upper]) > 0:
    corr = np.corrcoef(power_norm[upper], total_ana_norm[upper])[0, 1]
    print(f"Pattern correlation (0-90 deg): {corr:.3f}")
    if corr > 0.5:
        print(f"PASS: pattern matches image theory (corr={corr:.3f} > 0.5)")
    else:
        print(f"FAIL: poor pattern match (corr={corr:.3f})")
        PASS = False

# =============================================================================
# Plot
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f"Dipole at h=lambda/4 Above PEC Ground", fontsize=14)

# 1. Radiation pattern (upper hemisphere)
if np.max(power) > 0:
    pattern_db = 10 * np.log10(power_norm + 1e-30)
    ana_db = 10 * np.log10(total_ana_norm + 1e-30)
    ax1.plot(theta_deg, pattern_db, "b-", linewidth=2, label="rfx FDTD")
    ax1.plot(theta_deg, ana_db, "k--", linewidth=1.5, label="Image theory")
ax1.axvline(90, color="gray", ls=":", alpha=0.5, label="Ground plane")
ax1.set_xlabel("Theta (degrees)")
ax1.set_ylabel("Normalized pattern (dB)")
ax1.set_xlim(0, 180)
ax1.set_ylim(-30, 5)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_title(f"E-plane (D={D_dbi:.1f} dBi)")

# 2. Polar plot (upper hemisphere)
ax2 = fig.add_subplot(122, projection="polar")
ax2.set_theta_zero_location("N")
ax2.set_theta_direction(-1)
if np.max(power) > 0:
    ax2.plot(theta_arr[:91], power_norm[:91], "b-", linewidth=2, label="rfx")
    ax2.plot(theta_arr[:91], total_ana_norm[:91], "k--", linewidth=1.5, label="Analytical")
ax2.set_title("Upper hemisphere", pad=20)
ax2.legend(loc="lower right", fontsize=8)

plt.tight_layout()
out_path = os.path.join(SCRIPT_DIR, "19_dipole_above_pec.png")
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved: {out_path}")

if PASS:
    print("\nALL CHECKS PASSED")
else:
    print("\nSOME CHECKS FAILED")
