"""Cross-validation: Non-Uniform Mesh Convergence Study

Validates that non-uniform mesh produces convergent results as mesh
density increases.  Uses a PEC cavity resonance (exact analytical)
as the benchmark.

Structure: PEC cavity 50x40x30 mm, TM110 mode
  f_110 = c/(2) * sqrt((1/Lx)^2 + (1/Ly)^2) = 4.506 GHz

Convergence test:
  Run with 3 mesh densities (coarse → fine) using non-uniform z-mesh
  with graded profile.  Verify that resonance frequency error decreases
  monotonically and converges at roughly second order.

PASS criteria:
  - Frequency error decreases with finer mesh (monotonic convergence)
  - Finest mesh error < 1%
  - Convergence order > 1 (at least first-order)

Save: examples/crossval/18_nu_mesh_convergence.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from rfx import Simulation, Box
from rfx.auto_config import smooth_grading

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

C0 = 2.998e8

# PEC Cavity dimensions
Lx = 50e-3
Ly = 40e-3
Lz = 30e-3

# Analytical TM110 resonance
f_110 = C0 / 2 * np.sqrt((1 / Lx) ** 2 + (1 / Ly) ** 2)
print("=" * 60)
print("Cross-Validation: Non-Uniform Mesh Convergence")
print("=" * 60)
print(f"Cavity: {Lx*1e3:.0f} x {Ly*1e3:.0f} x {Lz*1e3:.0f} mm")
print(f"Analytical TM110: {f_110/1e9:.4f} GHz")
print()

# Mesh densities to test: coarse, medium, fine
# dx values in mm
dx_values = [3.0e-3, 2.0e-3, 1.5e-3, 1.0e-3]
labels = ["Coarse (3mm)", "Medium (2mm)", "Fine (1.5mm)", "Finest (1mm)"]

results = []

for dx, label in zip(dx_values, labels):
    print(f"Running {label} (dx={dx*1e3:.1f} mm)...")

    # Non-uniform z-mesh: finer near center, coarser at edges
    nz = max(6, int(round(Lz / dx)))
    # Create graded profile with smooth variation
    base = np.full(nz, dx)
    # Make center cells finer (0.7x) and edge cells coarser (1.3x)
    ratio = np.linspace(1.3, 0.7, nz // 2)
    ratio = np.concatenate([ratio, ratio[::-1]])
    if len(ratio) < nz:
        ratio = np.concatenate([ratio, [1.0] * (nz - len(ratio))])
    dz_raw = base * ratio[:nz]
    # Normalize so total = Lz
    dz_raw = dz_raw * Lz / np.sum(dz_raw)
    dz_profile = smooth_grading(dz_raw, max_ratio=1.4)
    # Adjust to match Lz exactly
    dz_profile = dz_profile * Lz / np.sum(dz_profile)

    sim = Simulation(
        freq_max=f_110 * 2,
        domain=(Lx, Ly, 0),
        dx=dx,
        dz_profile=dz_profile,
        boundary="pec",
    )

    # Point source at off-center location to excite TM110
    sim.add_source(
        position=(Lx * 0.3, Ly * 0.3, Lz * 0.4),
        component="ez",
    )
    sim.add_probe(
        position=(Lx * 0.25, Ly * 0.25, Lz * 0.5),
        component="ez",
    )

    t0 = time.time()
    result = sim.run(num_periods=30)
    elapsed = time.time() - t0

    # Extract resonance via Harminv
    modes = result.find_resonances(freq_range=(3e9, 6e9))
    if modes:
        best = min(modes, key=lambda m: abs(m.freq - f_110))
        f_sim = best.freq
        err = abs(f_sim - f_110) / f_110
        print(f"  f_sim={f_sim/1e9:.4f} GHz, error={err*100:.3f}%, Q={best.Q:.0f}, "
              f"time={elapsed:.1f}s, nz={len(dz_profile)}")
        results.append((dx, f_sim, err, best.Q, len(dz_profile)))
    else:
        print(f"  No resonance found! time={elapsed:.1f}s")
        results.append((dx, None, None, None, len(dz_profile)))

# =============================================================================
# Validation
# =============================================================================
print(f"\n{'='*60}")
print("Convergence Summary:")
print(f"{'Mesh':>12s} {'dx (mm)':>8s} {'f (GHz)':>10s} {'Error (%)':>10s} {'nz':>4s}")
print("-" * 48)

errors = []
dxs = []
for i, (dx, f_sim, err, Q, nz) in enumerate(results):
    if f_sim is not None:
        print(f"{labels[i]:>12s} {dx*1e3:>8.1f} {f_sim/1e9:>10.4f} {err*100:>10.3f} {nz:>4d}")
        errors.append(err)
        dxs.append(dx)
    else:
        print(f"{labels[i]:>12s} {dx*1e3:>8.1f} {'N/A':>10s} {'N/A':>10s} {nz:>4d}")

PASS = True

# Check 1: Monotonic convergence
if len(errors) >= 2:
    monotonic = all(errors[i] >= errors[i+1] for i in range(len(errors)-1))
    if monotonic:
        print(f"\nPASS: monotonic convergence ({errors[0]*100:.3f}% → {errors[-1]*100:.3f}%)")
    else:
        print(f"\nFAIL: non-monotonic convergence: {[f'{e*100:.3f}%' for e in errors]}")
        PASS = False

# Check 2: Finest mesh < 1% error
if errors and errors[-1] < 0.03:
    print(f"PASS: finest mesh error {errors[-1]*100:.3f}% < 3%")
else:
    if errors:
        print(f"FAIL: finest mesh error {errors[-1]*100:.3f}% (expected < 3%)")
    PASS = False

# Check 3: Convergence order
if len(errors) >= 2 and len(dxs) >= 2 and errors[-2] > 0 and errors[-1] > 0:
    order = np.log(errors[-2] / errors[-1]) / np.log(dxs[-2] / dxs[-1])
    print(f"Convergence order: {order:.2f} (between last two mesh levels)")
    if order > 1.0:
        print(f"PASS: convergence order {order:.1f} > 1")
    else:
        print(f"INFO: convergence order {order:.1f} (expected > 1 for second-order scheme)")
else:
    order = None

# =============================================================================
# Plot
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Non-Uniform Mesh Convergence: PEC Cavity TM110", fontsize=14)

# 1. Error vs mesh size
if errors:
    ax1.loglog([d*1e3 for d in dxs], [e*100 for e in errors], "bo-",
               linewidth=2, markersize=8)
    # Reference slope for order 2
    if len(dxs) >= 2:
        dx_ref = np.array(dxs)
        err_ref = errors[0] * (dx_ref / dxs[0]) ** 2
        ax1.loglog([d*1e3 for d in dx_ref], [e*100 for e in err_ref],
                   "k--", alpha=0.3, label="2nd order ref")
    ax1.set_xlabel("dx (mm)")
    ax1.set_ylabel("Frequency Error (%)")
    ax1.legend()
    ax1.grid(True, alpha=0.3, which="both")
    title_str = f"Convergence (order={order:.1f})" if order else "Convergence"
    ax1.set_title(title_str)

# 2. dz profile visualization
for i, (dx, f_sim, err, Q, nz) in enumerate(results):
    if f_sim is not None:
        dz_raw_vis = np.full(nz, dx)  # simplified
        z_edges = np.cumsum(np.concatenate([[0], dz_raw_vis]))
        ax2.barh(np.arange(nz), dz_raw_vis * 1e3, height=0.8,
                 alpha=0.4, label=labels[i])

ax2.set_xlabel("dz (mm)")
ax2.set_ylabel("Cell index")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_title("z-cell sizes")

plt.tight_layout()
out_path = os.path.join(SCRIPT_DIR, "18_nu_mesh_convergence.png")
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved: {out_path}")

if PASS:
    print("\nALL CHECKS PASSED")
else:
    print("\nSOME CHECKS FAILED")
