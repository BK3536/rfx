"""Example 4: 2.4 GHz Microstrip Patch Antenna on FR4

Showcase example demonstrating non-uniform mesh (graded dz for thin substrate),
analytical patch design via Hammerstad formula, Harminv resonance extraction,
and comprehensive 4-panel visualization.

Design flow:
  1. Hammerstad formula -> patch length L, width W, effective eps_eff
  2. Simulation(dz_profile=...) for non-uniform z-grid (fine in substrate)
  3. add_source + add_probe for soft excitation and ring-down recording
  4. SnapshotSpec to capture Ez at substrate mid-plane during simulation
  5. find_resonances() via Harminv for accurate frequency and Q extraction
  6. 4-panel figure: geometry (xz cross-section) + Ez snapshot + spectrum + summary

Saves: examples/04_patch_antenna.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rfx import Simulation, Box
from rfx.sources.sources import GaussianPulse
from rfx.simulation import SnapshotSpec

# ---- Physical constants ----
C0 = 2.998e8   # speed of light (m/s)

# ---- Design parameters ----
f0 = 2.4e9
eps_r = 4.4          # FR4 relative permittivity
tan_d = 0.02         # loss tangent
h = 1.6e-3           # substrate thickness

# ---- Hammerstad patch dimensions ----
W = C0 / (2 * f0) * np.sqrt(2.0 / (eps_r + 1.0))
eps_eff = (eps_r + 1.0) / 2.0 + (eps_r - 1.0) / 2.0 * (1.0 + 12.0 * h / W) ** (-0.5)
dL = 0.412 * h * (
    (eps_eff + 0.3) * (W / h + 0.264) /
    ((eps_eff - 0.258) * (W / h + 0.8))
)
L = C0 / (2.0 * f0 * np.sqrt(eps_eff)) - 2.0 * dL

print(f"Patch dimensions : L={L * 1e3:.2f} mm  W={W * 1e3:.2f} mm")
print(f"Substrate        : h={h * 1e3:.1f} mm  eps_r={eps_r}  tan_d={tan_d}")
print(f"eps_eff          : {eps_eff:.4f}")

# ---- Mesh parameters ----
dx = 0.5e-3           # lateral cell size (0.5 mm)
margin = 15e-3        # air margin around patch
dom_x = L + 2 * margin
dom_y = W + 2 * margin

# ---- Non-uniform z-profile: fine cells in substrate, coarse in air ----
n_sub = max(4, int(np.ceil(h / dx)))   # at least 4 cells through substrate
dz_sub = h / n_sub                     # fine z cell
n_air = max(6, int(np.ceil(margin / dx)))
dz_profile = np.concatenate([
    np.full(n_sub, dz_sub),            # substrate region
    np.full(n_air, dx),                # air above substrate
])
dom_z_total = float(np.sum(dz_profile))

print(f"\nDomain           : {dom_x * 1e3:.0f} x {dom_y * 1e3:.0f} x {dom_z_total * 1e3:.1f} mm")
print(f"dz_sub           : {dz_sub * 1e3:.3f} mm  ({n_sub} cells in substrate)")
print(f"dx               : {dx * 1e3:.1f} mm")

# ---- Build simulation ----
sigma_sub = 2.0 * np.pi * f0 * 8.854e-12 * eps_r * tan_d

sim = Simulation(
    freq_max=f0 * 2.0,
    domain=(dom_x, dom_y, 0),   # z=0 sentinel; actual z from dz_profile
    dx=dx,
    dz_profile=dz_profile,
    cpml_layers=12,
)

# ---- Materials ----
sim.add_material("substrate", eps_r=eps_r, sigma=sigma_sub)

# ---- Geometry ----
# Ground plane: bottom face (z=0 plane, 1 cell thick)
sim.add(Box((0, 0, 0), (dom_x, dom_y, 0)), material="pec")
# FR4 substrate
sim.add(Box((0, 0, 0), (dom_x, dom_y, h)), material="substrate")
# Patch: top surface of substrate
px0, py0 = margin, margin
sim.add(Box((px0, py0, h), (px0 + L, py0 + W, h)), material="pec")

# ---- Source: soft point source near feed point ----
src_x = px0 + L / 3.0
src_y = py0 + W / 2.0
src_z = h / 2.0    # inside substrate

sim.add_source(
    (src_x, src_y, src_z),
    component="ez",
    waveform=GaussianPulse(f0=f0, bandwidth=0.8),
)
sim.add_probe((src_x, src_y, src_z), component="ez")

# ---- Build non-uniform grid to find substrate mid-plane index ----
nu_grid = sim._build_nonuniform_grid()

# Locate iz_sub_mid: cell index at z = h/2 within the non-uniform grid
dz_np = np.array(nu_grid.dz)
z_edges = np.concatenate([[0.0], np.cumsum(dz_np)])
iz_sub_mid = int(np.argmin(np.abs(z_edges[:-1] + dz_np / 2.0 - h / 2.0)))

# ---- SnapshotSpec: capture Ez at substrate mid-plane every 50 steps ----
snap = SnapshotSpec(
    interval=50,
    components=("ez",),
    slice_axis=2,
    slice_index=iz_sub_mid,
)

# Run ~15 ns for good Harminv ring-down
n_steps = int(np.ceil(15e-9 / nu_grid.dt))
print(f"\nRunning {n_steps} steps  (dt={nu_grid.dt * 1e12:.3f} ps) ...")
result = sim.run(n_steps=n_steps, snapshot=snap)

# ---- Resonance extraction ----
modes = result.find_resonances(
    freq_range=(f0 * 0.5, f0 * 1.5),
    probe_idx=0,
)

if modes:
    best = min(modes, key=lambda m: abs(m.freq - f0))
    f_sim = best.freq
    Q_sim = best.Q
    print(f"Harminv modes found: {len(modes)}")
else:
    # FFT fallback
    ts_arr = np.asarray(result.time_series).ravel()
    spectrum_fb = np.abs(np.fft.rfft(ts_arr, n=len(ts_arr) * 8))
    freqs_fb = np.fft.rfftfreq(len(ts_arr) * 8, d=result.dt)
    band = (freqs_fb > f0 * 0.5) & (freqs_fb < f0 * 1.5)
    f_sim = freqs_fb[np.argmax(spectrum_fb * band)]
    Q_sim = float("nan")
    print("Harminv found no modes; using FFT peak")

err_pct = abs(f_sim - f0) / f0 * 100
print(f"\nDesign frequency : {f0 / 1e9:.4f} GHz")
print(f"Simulated        : {f_sim / 1e9:.4f} GHz")
print(f"Error            : {err_pct:.2f} %")
if not np.isnan(Q_sim):
    print(f"Q factor         : {Q_sim:.1f}")

# ---- Assemble materials for geometry visualization ----
materials_nu, _debye_nu, _lorentz_nu, pec_mask_nu = sim._assemble_materials_nu(nu_grid)
eps_r_arr = np.asarray(materials_nu.eps_r)

# ---- Probe time series ----
ts_arr = np.asarray(result.time_series)
if ts_arr.ndim == 2:
    ts_probe = ts_arr[:, 0]
else:
    ts_probe = ts_arr.ravel()

# ---- FFT spectrum ----
nfft = len(ts_probe) * 8
spectrum = np.abs(np.fft.rfft(ts_probe, n=nfft))
freqs_fft = np.fft.rfftfreq(nfft, d=result.dt) / 1e9   # GHz

# ---- Pick best Ez snapshot frame (peak field energy) ----
snaps_ez = None
ez_snap_label = "Ez at substrate mid-plane"
if result.snapshots and "ez" in result.snapshots:
    snaps = np.asarray(result.snapshots["ez"])  # (n_frames, nx, ny)
    peak_frame = int(np.argmax(np.max(np.abs(snaps), axis=(1, 2))))
    snaps_ez = snaps[peak_frame]  # (nx, ny)
    t_peak_ns = peak_frame * 50 * result.dt * 1e9
    ez_snap_label = f"Ez at z=h/2 (t~{t_peak_ns:.1f} ns, peak frame)"

# ---- Physical coordinate arrays (mm) ----
x_mm = np.arange(nu_grid.nx) * dx * 1e3
y_mm = np.arange(nu_grid.ny) * dx * 1e3

# xz cross-section: physical z coordinates (non-uniform)
z_phys_mm = (z_edges[:-1] + dz_np / 2.0) * 1e3   # cell centers in mm
iy_ctr = nu_grid.ny // 2

# ---- 4-panel figure ----
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    f"2.4 GHz Patch Antenna on FR4  "
    f"(L={L * 1e3:.1f} mm, W={W * 1e3:.1f} mm, h={h * 1e3:.1f} mm)",
    fontsize=13, fontweight="bold",
)

# Panel 1: Geometry — eps_r xz cross-section at y=center (physical mm)
ax1 = axes[0, 0]
eps_xz = eps_r_arr[:, iy_ctr, :]
im1 = ax1.pcolormesh(
    x_mm, z_phys_mm, eps_xz.T,
    cmap="viridis", shading="auto",
)
fig.colorbar(im1, ax=ax1, label="eps_r")
# Mark substrate top surface
ax1.axhline(h * 1e3, color="white", ls="--", lw=0.8, label=f"Substrate top (z={h*1e3:.1f} mm)")
ax1.axhline(h / 2 * 1e3, color="cyan", ls=":", lw=0.8, label=f"Snapshot plane (z=h/2)")
ax1.set_xlabel("x (mm)")
ax1.set_ylabel("z (mm)")
ax1.set_title("Geometry: eps_r xz cross-section (y=center)")
ax1.legend(fontsize=7)

# Panel 2: Ez field at substrate mid-plane (from SnapshotSpec, peak frame)
ax2 = axes[0, 1]
if snaps_ez is not None:
    vmax = float(np.max(np.abs(snaps_ez))) or 1.0
    im2 = ax2.pcolormesh(
        x_mm, y_mm, snaps_ez.T,
        cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto",
    )
    fig.colorbar(im2, ax=ax2, label="Ez (V/m)")
    # Overlay patch boundary
    ax2.add_patch(plt.Rectangle(
        (px0 * 1e3, py0 * 1e3), L * 1e3, W * 1e3,
        linewidth=1.5, edgecolor="black", facecolor="none",
        label="Patch outline",
    ))
    ax2.legend(fontsize=7)
else:
    ax2.text(0.5, 0.5, "No snapshot data", ha="center", va="center",
             transform=ax2.transAxes)
ax2.set_xlabel("x (mm)")
ax2.set_ylabel("y (mm)")
ax2.set_title(ez_snap_label)

# Panel 3: Frequency spectrum with design frequency marker
ax3 = axes[1, 0]
spec_db = 20 * np.log10(np.maximum(spectrum / (spectrum.max() or 1.0), 1e-10))
band_mask = (freqs_fft > f0 * 0.4 / 1e9) & (freqs_fft < f0 * 1.6 / 1e9)
ax3.plot(freqs_fft[band_mask], spec_db[band_mask], lw=1.0)
ax3.axvline(f0 / 1e9, color="g", ls="--", lw=1.5,
            label=f"Design {f0 / 1e9:.2f} GHz")
ax3.axvline(f_sim / 1e9, color="r", ls=":", lw=1.5,
            label=f"Simulated {f_sim / 1e9:.3f} GHz")
ax3.set_xlabel("Frequency (GHz)")
ax3.set_ylabel("Normalized (dB)")
ax3.set_title("Frequency spectrum")
ax3.set_ylim(-60, 5)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Panel 4: Summary annotation
ax4 = axes[1, 1]
ax4.axis("off")
lines = [
    "Patch Antenna Summary",
    chr(0x2500) * 28,
    f"Design freq  : {f0 / 1e9:.4f} GHz",
    f"Simulated    : {f_sim / 1e9:.4f} GHz",
    f"Error        : {err_pct:.3f} %",
]
if not np.isnan(Q_sim):
    lines.append(f"Q factor     : {Q_sim:.1f}")
lines += [
    "",
    f"L = {L * 1e3:.2f} mm",
    f"W = {W * 1e3:.2f} mm",
    f"h = {h * 1e3:.1f} mm  (FR4)",
    f"eps_eff = {eps_eff:.4f}",
    "",
    f"dx = {dx * 1e3:.1f} mm",
    f"dz_sub = {dz_sub * 1e3:.3f} mm",
    f"Steps = {n_steps}",
    f"dt = {result.dt * 1e12:.3f} ps",
]
ax4.text(0.05, 0.97, "\n".join(lines), transform=ax4.transAxes,
         va="top", ha="left", fontsize=9, family="monospace",
         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))
ax4.set_title("Simulation summary")

plt.tight_layout()
out_path = "examples/04_patch_antenna.png"
plt.savefig(out_path, dpi=150)
plt.close(fig)
print(f"\nPlot saved: {out_path}")
