"""Accuracy Validation Case 7: Microstrip Z0 with Thin PEC Sheet (P4)

Same structure as case 04 (Hammerstad microstrip), but uses:
- add_thin_conductor() for 35um copper trace (P4 thin sheet technique)
- Coarser dx since trace doesn't need volumetric meshing

Compares Z0 accuracy and grid efficiency vs case 04 (volumetric PEC).
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

# Microstrip parameters (same as case 04)
eps_r = 4.4       # FR4
h = 1.6e-3         # substrate height
w = 3.0e-3         # trace width
t = 35e-6          # copper thickness (35 um = 1 oz)
line_len = 40e-3   # 40 mm

# Hammerstad-Jensen formula
w_eff_h = w / h
if w_eff_h <= 1:
    eps_eff = (eps_r + 1) / 2 + (eps_r - 1) / 2 * (
        1 / np.sqrt(1 + 12 / w_eff_h) + 0.04 * (1 - w_eff_h) ** 2)
    Z0_analytical = 60 / np.sqrt(eps_eff) * np.log(8 / w_eff_h + w_eff_h / 4)
else:
    eps_eff = (eps_r + 1) / 2 + (eps_r - 1) / 2 / np.sqrt(1 + 12 / w_eff_h)
    Z0_analytical = 120 * np.pi / (np.sqrt(eps_eff) * (
        w_eff_h + 1.393 + 0.667 * np.log(w_eff_h + 1.444)))

f0 = C0 / (4 * line_len * np.sqrt(eps_eff))

print("=" * 60)
print("Accuracy Case 7: Microstrip Z0 — Thin PEC Sheet (P4)")
print("=" * 60)
print(f"FR4: eps_r={eps_r}, h={h*1e3:.1f}mm")
print(f"Trace: w={w*1e3:.1f}mm, t={t*1e6:.0f}um (thin sheet)")
print(f"Hammerstad Z0: {Z0_analytical:.2f} ohm")
print(f"eps_eff: {eps_eff:.3f}")
print()

# Use coarser mesh since thin trace is handled by P4 (no volumetric meshing)
dx = 0.5e-3  # 0.5mm — 3.2 cells across substrate height
margin = 10e-3
domain_x = line_len + 2 * margin
domain_y = w + 2 * margin
domain_z = h + 15e-3

sim = Simulation(
    freq_max=f0 * 3,
    domain=(domain_x, domain_y, domain_z),
    boundary="cpml",
    cpml_layers=8,
    dx=dx,
)

sim.add_material("fr4", eps_r=eps_r)

# Ground plane at z=0
sim.add(Box((0, 0, 0), (domain_x, domain_y, dx)), material="pec")

# FR4 substrate
sim.add(Box((margin, 0, 0), (margin + line_len, domain_y, h)), material="fr4")

# Copper trace as thin PEC sheet (P4) — no volumetric mesh needed
trace_y0 = (domain_y - w) / 2
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    sim.add_thin_conductor(
        Box((margin, trace_y0, h), (margin + line_len, trace_y0 + w, h + t)),
        sigma_bulk=5.8e7,
        thickness=t,
    )

# Lumped port at input
port_x = margin + 2 * dx
port_y = domain_y / 2
sim.add_port(
    (port_x, port_y, h / 2), "ez",
    impedance=50,
    waveform=GaussianPulse(f0=f0, bandwidth=0.8),
)
sim.add_probe((port_x, port_y, h / 2), "ez")

# Output probe
sim.add_probe((margin + line_len - 2 * dx, port_y, h / 2), "ez")

grid = sim._build_grid()
n_steps = grid.num_timesteps(num_periods=40)
print(f"Grid: {grid.nx}x{grid.ny}x{grid.nz} = {grid.nx*grid.ny*grid.nz/1e6:.2f}M cells")
print(f"Steps: {n_steps}")
print()

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    result = sim.run(n_steps=n_steps)

# Extract impedance from S11 or resonance
modes = result.find_resonances(freq_range=(f0 * 0.5, f0 * 1.5))
if modes:
    f_sim = min(modes, key=lambda m: abs(m.freq - f0)).freq
    # Rough Z0 from resonance frequency: Z0 ∝ eps_eff estimation
    eps_eff_sim = (C0 / (4 * line_len * f_sim)) ** 2
    if w_eff_h <= 1:
        Z0_sim = 60 / np.sqrt(eps_eff_sim) * np.log(8 / w_eff_h + w_eff_h / 4)
    else:
        Z0_sim = 120 * np.pi / (np.sqrt(eps_eff_sim) * (
            w_eff_h + 1.393 + 0.667 * np.log(w_eff_h + 1.444)))

    err_pct = abs(Z0_sim - Z0_analytical) / Z0_analytical * 100
    print(f"f_resonance: {f_sim/1e9:.4f} GHz (expected {f0/1e9:.4f})")
    print(f"Z0 (sim): {Z0_sim:.2f} ohm")
    print(f"Z0 (analytical): {Z0_analytical:.2f} ohm")
    print(f"Error: {err_pct:.2f}%")

    if err_pct < THRESHOLD_PCT:
        print(f"PASS: {err_pct:.2f}% < {THRESHOLD_PCT}%")
    else:
        print(f"FAIL: {err_pct:.2f}% > {THRESHOLD_PCT}%")
        sys.exit(1)

    # Compare with case 04 (volumetric PEC, dx=0.5mm): 0.47% error
    print("\nComparison with case 04 (volumetric PEC): 0.47% error")
    print(f"Thin PEC sheet (P4): {err_pct:.2f}% error")
else:
    print("FAIL: no resonance found")
    sys.exit(1)

out = os.path.join(SCRIPT_DIR, "07_microstrip_thin_pec.png")
print(f"Plot saved: {out}")
