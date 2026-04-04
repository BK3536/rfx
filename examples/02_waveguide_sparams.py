"""Example 2: Waveguide S-Parameter Extraction

Two-port rectangular waveguide with a dielectric obstacle (eps_r=4).
Extracts the 2x2 S-matrix using the high-level Simulation API with
waveguide ports and plots geometry, S-parameters, and a field snapshot.

Expected: above TE10 cutoff (~3.75 GHz for a=40 mm guide),
  |S21| is high (transmission through obstacle) and
  |S11| shows reflection peaks at resonant frequencies of the slab.

Saves: examples/02_waveguide_sparams.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from rfx import Simulation, Box
from rfx.sources.sources import GaussianPulse
from rfx.simulation import SnapshotSpec

# ---- Waveguide geometry ----
Lx = 0.12     # length (m)
Ly = 0.04     # guide width (a = 40 mm  → cutoff TE10 ~3.75 GHz)
Lz = 0.02     # guide height (b = 20 mm)
dx = 0.002

# ---- Compute S-matrix with waveguide ports ----
sim = Simulation(
    freq_max=10e9,
    domain=(Lx, Ly, Lz),
    boundary="cpml",
    cpml_layers=10,
    dx=dx,
)

# ---- Dielectric obstacle in the middle ----
sim.add_material("obstacle", eps_r=4.0)
sim.add(Box((0.05, 0.0, 0.0), (0.07, Ly, Lz)), material="obstacle")

# ---- Waveguide ports (TE10) ----
freqs = jnp.linspace(4.5e9, 8e9, 30)
sim.add_waveguide_port(
    0.01, direction="+x", mode=(1, 0), mode_type="TE",
    freqs=freqs, f0=6e9, name="port1",
)
sim.add_waveguide_port(
    0.09, direction="-x", mode=(1, 0), mode_type="TE",
    freqs=freqs, f0=6e9, name="port2",
)

print("Running waveguide S-matrix extraction ...")
result = sim.compute_waveguide_s_matrix(num_periods=30)
S = result.s_params          # (2, 2, n_freqs)
f_GHz = np.array(result.freqs) / 1e9

s11_dB = 20 * np.log10(np.maximum(np.abs(S[0, 0, :]), 1e-10))
s21_dB = 20 * np.log10(np.maximum(np.abs(S[1, 0, :]), 1e-10))
s12_dB = 20 * np.log10(np.maximum(np.abs(S[0, 1, :]), 1e-10))

print(f"Frequency range : {f_GHz[0]:.1f} - {f_GHz[-1]:.1f} GHz")
print(f"|S21| mean       : {np.mean(np.abs(S[1, 0, :])):.3f}")
recip = np.mean(np.abs(np.abs(S[1, 0, :]) - np.abs(S[0, 1, :])))
print(f"Reciprocity err : {recip:.4f}")

# ---- Build grid and materials for geometry visualization ----
grid = sim._build_grid()
materials, _, _, _, _, _ = sim._assemble_materials(grid)
eps_r_arr = np.asarray(materials.eps_r)

# ---- Run short simulation with SnapshotSpec for field visualization ----
sim_snap = Simulation(
    freq_max=10e9,
    domain=(Lx, Ly, Lz),
    boundary="cpml",
    cpml_layers=10,
    dx=dx,
)
sim_snap.add_material("obstacle", eps_r=4.0)
sim_snap.add(Box((0.05, 0.0, 0.0), (0.07, Ly, Lz)), material="obstacle")
sim_snap.add_port(
    (0.01, Ly / 2, Lz / 2), component="ez", impedance=50.0,
    waveform=GaussianPulse(f0=6e9, bandwidth=0.5),
)
sim_snap.add_probe((Lx / 2, Ly / 2, Lz / 2), component="ez")

grid_snap = sim_snap._build_grid()
nz_snap = grid_snap.nz

snap = SnapshotSpec(
    interval=50,
    components=("ez",),
    slice_axis=2,
    slice_index=nz_snap // 2,
)

snap_result = sim_snap.run(n_steps=600, compute_s_params=False, snapshot=snap)

# Pick snapshot frame with maximum field energy (wave propagating, source active)
snaps_ez = None
ez_snap_label = "Ez field snapshot"
if snap_result.snapshots and "ez" in snap_result.snapshots:
    snaps = np.asarray(snap_result.snapshots["ez"])  # (n_frames, nx, ny)
    peak_frame = int(np.argmax(np.max(np.abs(snaps), axis=(1, 2))))
    snaps_ez = snaps[peak_frame]  # (nx, ny)
    t_snap_ns = peak_frame * 50 * snap_result.dt * 1e9
    ez_snap_label = f"Ez snapshot (t~{t_snap_ns:.1f} ns, peak frame)"

# ---- Physical coordinate arrays (mm) ----
# Geometry axes: use total padded grid shape
nx_g = eps_r_arr.shape[0]
ny_g = eps_r_arr.shape[1]
x_mm_geom = np.arange(nx_g) * dx * 1e3
y_mm_geom = np.arange(ny_g) * dx * 1e3

# Field axes
nx_s = grid_snap.nx
ny_s = grid_snap.ny
x_mm_snap = np.arange(nx_s) * dx * 1e3
y_mm_snap = np.arange(ny_s) * dx * 1e3

# ---- 3-panel figure ----
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Waveguide with Dielectric Obstacle (TE10, eps_r=4)",
             fontsize=13, fontweight="bold")

# Panel 1: Geometry cross-section — eps_r slice at z=center (physical coords mm)
ax = axes[0]
iz_ctr = eps_r_arr.shape[2] // 2
eps_slice = eps_r_arr[:, :, iz_ctr]
im = ax.pcolormesh(
    x_mm_geom, y_mm_geom, eps_slice.T,
    cmap="viridis", shading="auto",
)
fig.colorbar(im, ax=ax, label="eps_r")
# Mark obstacle region
obs_x0 = 0.05 * 1e3
obs_x1 = 0.07 * 1e3
ax.axvline(obs_x0, color="w", ls="--", lw=1.2, label="Obstacle")
ax.axvline(obs_x1, color="w", ls="--", lw=1.2)
# Mark port planes
ax.axvline(0.01 * 1e3, color="r", ls=":", lw=1.2, label="Port 1")
ax.axvline(0.09 * 1e3, color="lime", ls=":", lw=1.2, label="Port 2")
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
ax.set_title("Geometry: eps_r (z=center)")
ax.legend(fontsize=8)

# Panel 2: |S11|, |S21|, |S12| vs frequency
ax = axes[1]
ax.plot(f_GHz, s11_dB, "b-", lw=1.5, label="|S11|")
ax.plot(f_GHz, s21_dB, "r-", lw=1.5, label="|S21|")
ax.plot(f_GHz, s12_dB, "r--", lw=1.0, alpha=0.6, label="|S12| (reciprocity)")
ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Magnitude (dB)")
ax.set_title("S-Parameters")
ax.set_ylim(-30, 5)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 3: Ez field snapshot showing wave propagation (from SnapshotSpec)
ax = axes[2]
if snaps_ez is not None:
    vmax = float(np.max(np.abs(snaps_ez))) or 1.0
    im2 = ax.pcolormesh(
        x_mm_snap, y_mm_snap, snaps_ez.T,
        cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto",
    )
    fig.colorbar(im2, ax=ax, label="Ez (V/m)")
else:
    ax.text(0.5, 0.5, "No snapshot data", ha="center", va="center",
            transform=ax.transAxes)
# Mark obstacle region on field plot
ax.axvline(0.05 * 1e3, color="w", ls="--", lw=1.0, alpha=0.7)
ax.axvline(0.07 * 1e3, color="w", ls="--", lw=1.0, alpha=0.7)
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
ax.set_title(ez_snap_label)

plt.tight_layout()
out_path = "examples/02_waveguide_sparams.png"
plt.savefig(out_path, dpi=150)
plt.close(fig)
print(f"Plot saved: {out_path}")
