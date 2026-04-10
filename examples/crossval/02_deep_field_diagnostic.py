"""Deep field diagnostic: rfx vs Meep waveguide bend

Focus on QUALITATIVE field distribution differences:
1. Lag-compensated field comparison (remove timing offset, see spatial diffs)
2. PML boundary region zoom — is rfx PML absorbing correctly?
3. Bend corner zoom — radiation/scattering differences
4. Effective epsilon comparison (rfx subpixel vs Meep subpixel)
5. Sigma profile comparison (rfx UPML vs Meep PML)
"""

import os, sys, math
os.environ.setdefault("JAX_ENABLE_X64", "1")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
C0 = 2.998e8

# =========================================================================
# Parameters (identical to 01_field_progression_review.py)
# =========================================================================
a = 1.0e-6
eps_wg = 12.0
w_wg = 1.0
fcen = 0.15
fwidth = 0.1
resolution = 10
dpml = 1.0

sx_meep = 16.0
sy_meep = 16.0
cell_x = sx_meep + 2 * dpml
cell_y = sy_meep + 2 * dpml

dx = a / resolution
domain_x = sx_meep * a
domain_y = sy_meep * a
cpml_n = int(dpml / (1.0 / resolution))
COORD_OFFSET = sx_meep / 2

src_x_meep = -sx_meep / 2 + 0.1
src_x_rfx = (src_x_meep + COORD_OFFSET) * a
bw_rfx = fwidth / (fcen * math.pi * math.sqrt(2))
fcen_hz = fcen * C0 / a

pml_cells = int(dpml * resolution)  # 10

N_STEPS = 6000

# =========================================================================
# PART 0: Courant factor comparison
# =========================================================================
print("=" * 70)
print("PART 0: Courant factor and timestep comparison")
print("=" * 70)

# rfx: dt = 0.99 * dx / (c * sqrt(2))  [2D CFL]
rfx_dt = 0.99 * dx / (C0 * math.sqrt(2))
# Meep: dt = Courant / resolution, default Courant = 0.5
meep_courant = 0.5
meep_dt_meep_units = meep_courant / resolution  # 0.05
meep_dt_si = meep_dt_meep_units * a / C0

print(f"  rfx  dt = {rfx_dt:.4e} s  (CFL ratio = 0.99)")
print(f"  Meep dt = {meep_dt_si:.4e} s  (Courant = {meep_courant})")
print(f"  Ratio rfx_dt/meep_dt = {rfx_dt/meep_dt_si:.3f}")
print(f"  rfx  Courant S = {C0*rfx_dt/dx:.4f}")
print(f"  Meep Courant S = {C0*meep_dt_si/dx:.4f}")
print()

# =========================================================================
# PART 1: Effective epsilon comparison (subpixel)
# =========================================================================
print("=" * 70)
print("PART 1: Effective epsilon — rfx subpixel vs Meep subpixel")
print("=" * 70)

import meep as mp

# Meep epsilon (includes Meep's subpixel)
cell_meep = mp.Vector3(cell_x, cell_y)
pml_meep = [mp.PML(dpml)]
geo_meep = [
    mp.Block(size=mp.Vector3(sx_meep / 2 + 0.5, 1),
             center=mp.Vector3(-sx_meep / 4 + 0.25, 0),
             material=mp.Medium(epsilon=eps_wg)),
    mp.Block(size=mp.Vector3(1, sy_meep / 2 + 0.5),
             center=mp.Vector3(0, sy_meep / 4 - 0.25),
             material=mp.Medium(epsilon=eps_wg)),
]
sim_meep = mp.Simulation(cell_size=cell_meep, boundary_layers=pml_meep,
                         geometry=geo_meep, sources=[], resolution=resolution)
sim_meep.init_sim()
eps_meep_full = sim_meep.get_array(center=mp.Vector3(), size=cell_meep,
                                    component=mp.Dielectric)
eps_meep = eps_meep_full[pml_cells:-pml_cells, pml_cells:-pml_cells]

# rfx epsilon (with subpixel smoothing)
from rfx import Simulation, Box
from rfx.sources.sources import ModulatedGaussian
from rfx.simulation import SnapshotSpec
import jax.numpy as jnp

sim_rfx = Simulation(freq_max=0.25 * C0 / a, domain=(domain_x, domain_y, dx),
                     dx=dx, boundary="upml", cpml_layers=cpml_n, mode="2d_tmz")
sim_rfx.add_material("wg", eps_r=eps_wg)
sim_rfx.add(Box((0, 7.5*a, 0), (8.5*a, 8.5*a, dx)), material="wg")
sim_rfx.add(Box((7.5*a, 7.5*a, 0), (8.5*a, 16*a, dx)), material="wg")

grid = sim_rfx._build_grid()
pad = grid.pad_x
n_domain = int(np.ceil(domain_x / dx)) + 1

# Get rfx subpixel-smoothed epsilon
from rfx.geometry.smoothing import compute_smoothed_eps
shape_eps_pairs = [
    (entry.shape, sim_rfx._resolve_material(entry.material_name).eps_r)
    for entry in sim_rfx._geometry
]
aniso_eps = compute_smoothed_eps(grid, shape_eps_pairs, background_eps=1.0)
eps_rfx_ez = np.asarray(aniso_eps[2][:, :, 0])  # Ez component smoothed eps
eps_rfx_domain = eps_rfx_ez[pad:pad+n_domain, pad:pad+n_domain]

# Also get staircase for comparison
base_mat = sim_rfx._assemble_materials(grid)
eps_rfx_staircase = np.asarray(base_mat[0].eps_r[:, :, 0])
eps_rfx_staircase_d = eps_rfx_staircase[pad:pad+n_domain, pad:pad+n_domain]

n_common = min(eps_meep.shape[0], eps_rfx_domain.shape[0])

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Row 1: Full domain
axes[0,0].imshow(eps_rfx_domain[:n_common, :n_common].T, origin="lower",
                  cmap="hot", vmin=1, vmax=12)
axes[0,0].set_title("rfx subpixel Ez-eps (domain)")
axes[0,1].imshow(eps_meep[:n_common, :n_common].T, origin="lower",
                  cmap="hot", vmin=1, vmax=12)
axes[0,1].set_title("Meep eps (with subpixel)")
diff_eps = eps_rfx_domain[:n_common, :n_common] - eps_meep[:n_common, :n_common]
axes[0,2].imshow(diff_eps.T, origin="lower", cmap="bwr", vmin=-6, vmax=6)
axes[0,2].set_title(f"rfx - Meep (max diff={np.max(np.abs(diff_eps)):.1f})")

# Row 2: Zoom to bend corner (x=70-95, y=70-95)
zx, zy = slice(70, 95), slice(70, 95)
axes[1,0].imshow(eps_rfx_domain[zx, zy].T, origin="lower", cmap="hot",
                  vmin=1, vmax=12, extent=[70,95,70,95])
axes[1,0].set_title("rfx subpixel (bend corner)")
axes[1,1].imshow(eps_meep[zx, zy].T, origin="lower", cmap="hot",
                  vmin=1, vmax=12, extent=[70,95,70,95])
axes[1,1].set_title("Meep eps (bend corner)")
axes[1,2].imshow(diff_eps[zx, zy].T, origin="lower", cmap="bwr",
                  vmin=-6, vmax=6, extent=[70,95,70,95])
axes[1,2].set_title("Difference (bend corner)")

for ax in axes.flat:
    ax.set_xlabel("x cell"); ax.set_ylabel("y cell")
plt.suptitle("Part 1: Effective epsilon comparison — subpixel smoothing", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "02_part1_epsilon.png"), dpi=150)
plt.close()

# Print quantitative eps profile along horizontal waveguide
print("  Eps profile at y=80 (waveguide center), x=70..90:")
print(f"  {'x':>4} {'rfx_subpix':>10} {'rfx_stair':>10} {'Meep':>10}")
for xi in range(70, 90):
    if xi < n_common:
        print(f"  {xi:>4} {eps_rfx_domain[xi, 80]:>10.2f} "
              f"{eps_rfx_staircase_d[xi, 80]:>10.2f} "
              f"{eps_meep[xi, 80]:>10.2f}")

# =========================================================================
# PART 2: PML sigma profile comparison
# =========================================================================
print(f"\n{'='*70}")
print("PART 2: PML sigma profile comparison — rfx UPML vs Meep")
print("=" * 70)

from rfx.boundaries.upml import _sigma_profile_1d

sig_E, sig_H = _sigma_profile_1d(cpml_n, rfx_dt, dx)
print(f"  rfx UPML: {cpml_n} layers, order=2, R=1e-15")
print(f"  rfx sigma_max = {sig_E[0]:.6e}")
print(f"  sigma_E profile (outermost→innermost): {np.array2string(sig_E, precision=3)}")
print(f"  sigma_H profile: {np.array2string(sig_H, precision=3)}")

# Meep sigma: can extract from meep.pml_params or compute analytically
# Meep default: polynomial order = 3, σ_max = -(p+1)*ln(R)/(2*η*d)
# But Meep uses dimensionless σ where η=1 (Meep units)
meep_pml_order = 3  # Meep default (I think)
meep_R = 1e-15
eta = math.sqrt(4*math.pi*1e-7 / 8.854e-12)  # ~377 ohm
d_pml = cpml_n * dx
meep_sigma_max = -math.log(meep_R) * (meep_pml_order + 1) / (2 * eta * d_pml)
rfx_sigma_max = -math.log(1e-15) * (2 + 1) / (2 * eta * d_pml)

print(f"\n  Meep-style sigma_max (order=3): {meep_sigma_max:.6e}")
print(f"  rfx sigma_max (order=2):        {rfx_sigma_max:.6e}")
print(f"  Ratio (Meep/rfx): {meep_sigma_max/rfx_sigma_max:.3f}")

# Plot sigma profiles
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
u = np.linspace(0, 1, 100)
# rfx: order 2
rfx_sigma_profile = rfx_sigma_max * u**2
# Meep order 3
meep_sigma_profile = meep_sigma_max * u**3
ax.plot(u, rfx_sigma_profile, 'b-', lw=2, label=f"rfx UPML (order=2, σ_max={rfx_sigma_max:.2e})")
ax.plot(u, meep_sigma_profile, 'r-', lw=2, label=f"Meep PML (order=3, σ_max={meep_sigma_max:.2e})")
ax.plot(np.linspace(0, 1, cpml_n), sig_E, 'b^', ms=8, label="rfx σ_E positions")
ax.plot(np.linspace(0, 1, cpml_n), sig_H, 'bs', ms=6, label="rfx σ_H positions")
ax.set_xlabel("Normalized depth into PML (0=boundary, 1=outermost)")
ax.set_ylabel("Conductivity σ (S/m)")
ax.set_title("Part 2: PML sigma profiles — rfx vs Meep")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "02_part2_sigma.png"), dpi=150)
plt.close()

# =========================================================================
# PART 3: Run both solvers, lag-compensated field comparison
# =========================================================================
print(f"\n{'='*70}")
print("PART 3: Lag-compensated field snapshot comparison")
print("=" * 70)

# Run rfx
sim_rfx2 = Simulation(freq_max=0.25 * C0 / a, domain=(domain_x, domain_y, dx),
                       dx=dx, boundary="upml", cpml_layers=cpml_n, mode="2d_tmz")
sim_rfx2.add_material("wg", eps_r=eps_wg)
sim_rfx2.add(Box((0, 7.5*a, 0), (8.5*a, 8.5*a, dx)), material="wg")
sim_rfx2.add(Box((7.5*a, 7.5*a, 0), (8.5*a, 16*a, dx)), material="wg")
for i in range(10):
    y = 7.5*a + (i + 0.5) * a / 10
    sim_rfx2.add_source(position=(src_x_rfx, y, 0), component="ez",
        waveform=ModulatedGaussian(f0=fcen_hz, bandwidth=bw_rfx,
                                   amplitude=1.0/10,
                                   cutoff=5.0/math.sqrt(2)))
sim_rfx2.add_probe(position=(COORD_OFFSET * a, COORD_OFFSET * a, 0), component="ez")

snap = SnapshotSpec(components=("ez",), slice_axis=2, slice_index=0)
print("  [rfx] Running...", flush=True)
res_rfx = sim_rfx2.run(n_steps=N_STEPS, snapshot=snap, subpixel_smoothing=True)
rfx_dt_actual = res_rfx.dt

# Extract rfx domain snapshots (all timesteps)
ez_rfx_all = np.asarray(res_rfx.snapshots["ez"])
rfx_domain = ez_rfx_all[:, pad:pad+n_domain, pad:pad+n_domain]

# Determine lag from probe time series
rfx_ts = np.asarray(res_rfx.time_series[:, 0])
rfx_t = np.arange(N_STEPS) * rfx_dt_actual

# Run Meep
print("  [Meep] Running...", flush=True)
src_meep = [mp.Source(mp.GaussianSource(fcen, fwidth=fwidth),
                      component=mp.Ez,
                      center=mp.Vector3(src_x_meep, 0),
                      size=mp.Vector3(0, w_wg))]
sim_meep3 = mp.Simulation(cell_size=cell_meep, boundary_layers=pml_meep,
                          geometry=geo_meep, sources=src_meep, resolution=resolution)

# Capture Meep fields at times that compensate for the ~113 fs lag
# rfx is AHEAD by ~113 fs, so to see the "same physical state",
# Meep needs to simulate 113 fs LONGER (or rfx 113 fs shorter)
lag_ps = 0.114  # from previous measurement
lag_si = lag_ps * 1e-12

# Comparison times: physical events in the waveguide
# We'll capture at these rfx times, and at rfx_time + lag for Meep
capture_times_rfx_ps = [0.15, 0.25, 0.40, 0.55, 0.80, 1.20]

# Direct (same time) and lag-compensated Meep frames
sim_meep3.init_sim()
meep_frames_direct = []
meep_frames_lag = []

print("  [Meep] Capturing lag-compensated snapshots...", flush=True)
# We need two Meep runs: one at direct times, one at lag-compensated times
# Actually just capture at both sets of times in one run (sorted)
all_meep_times_si = []
for t_ps in capture_times_rfx_ps:
    all_meep_times_si.append(t_ps * 1e-12)           # direct
    all_meep_times_si.append(t_ps * 1e-12 + lag_si)  # lag-compensated

all_meep_times_meep = [t * C0 / a for t in all_meep_times_si]
# Sort and track which frame is which
frame_order = sorted(range(len(all_meep_times_meep)),
                     key=lambda i: all_meep_times_meep[i])

all_frames = [None] * len(all_meep_times_meep)
for idx in frame_order:
    target = all_meep_times_meep[idx]
    remaining = target - sim_meep3.meep_time()
    if remaining > 0:
        sim_meep3.run(until=remaining)
    ez = sim_meep3.get_array(center=mp.Vector3(), size=cell_meep, component=mp.Ez)
    all_frames[idx] = ez.copy()

# Split into direct and lag-compensated
for i, t_ps in enumerate(capture_times_rfx_ps):
    meep_frames_direct.append(all_frames[2*i])
    meep_frames_lag.append(all_frames[2*i + 1])

meep_frames_direct = np.array(meep_frames_direct)
meep_frames_lag = np.array(meep_frames_lag)

# Extract interiors
meep_direct = meep_frames_direct[:, pml_cells:-pml_cells, pml_cells:-pml_cells]
meep_lag = meep_frames_lag[:, pml_cells:-pml_cells, pml_cells:-pml_cells]

# Get rfx frames at corresponding steps
rfx_steps = [min(N_STEPS-1, int(t*1e-12 / rfx_dt_actual)) for t in capture_times_rfx_ps]
rfx_frames = np.array([rfx_domain[s] for s in rfx_steps])

n_c = min(rfx_frames.shape[1], meep_direct.shape[1])
rfx_f = rfx_frames[:, :n_c, :n_c]
meep_d = meep_direct[:, :n_c, :n_c]
meep_l = meep_lag[:, :n_c, :n_c]

def envelope_2d(field):
    env = np.zeros_like(field)
    for j in range(field.shape[1]):
        env[:, j] = np.abs(hilbert(field[:, j]))
    return env

# =========================================================================
# PART 3A: 6-row comparison: rfx | Meep(direct) | Meep(lag) | diff(lag)
# =========================================================================
fig, axes = plt.subplots(len(capture_times_rfx_ps), 4,
                          figsize=(20, 4*len(capture_times_rfx_ps)))

for i, t_ps in enumerate(capture_times_rfx_ps):
    r_env = envelope_2d(rfx_f[i])
    md_env = envelope_2d(meep_d[i])
    ml_env = envelope_2d(meep_l[i])

    rn = r_env / max(r_env.max(), 1e-30)
    mdn = md_env / max(md_env.max(), 1e-30)
    mln = ml_env / max(ml_env.max(), 1e-30)
    vmax = max(rn.max(), mdn.max(), mln.max()) * 0.9 or 1.0

    # Correlation
    def corr(a, b):
        if np.std(a) > 1e-20 and np.std(b) > 1e-20:
            return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        return float('nan')

    c_direct = corr(rn, mdn)
    c_lag = corr(rn, mln)

    axes[i, 0].imshow(rn.T, origin="lower", cmap="hot", vmin=0, vmax=vmax)
    axes[i, 0].set_title(f"rfx |Ez| (t={t_ps:.2f}ps)", fontsize=10)
    axes[i, 0].set_ylabel("y")

    axes[i, 1].imshow(mdn.T, origin="lower", cmap="hot", vmin=0, vmax=vmax)
    axes[i, 1].set_title(f"Meep same-t (corr={c_direct:.3f})", fontsize=10)

    axes[i, 2].imshow(mln.T, origin="lower", cmap="hot", vmin=0, vmax=vmax)
    axes[i, 2].set_title(f"Meep +{lag_ps*1e3:.0f}fs (corr={c_lag:.3f})", fontsize=10)

    diff = rn - mln
    vd = max(np.abs(diff).max(), 1e-30)
    axes[i, 3].imshow(diff.T, origin="lower", cmap="bwr", vmin=-vd, vmax=vd)
    axes[i, 3].set_title(f"rfx - Meep(lag)", fontsize=10)

axes[-1, 0].set_xlabel("x"); axes[-1, 1].set_xlabel("x")
axes[-1, 2].set_xlabel("x"); axes[-1, 3].set_xlabel("x")
fig.suptitle(f"Part 3: Lag-compensated field comparison (lag={lag_ps*1e3:.0f}fs)\n"
             f"Left: rfx, Mid-L: Meep(same time), Mid-R: Meep(+lag), Right: diff(lag-comp)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "02_part3_fields.png"), dpi=150)
plt.close()

# =========================================================================
# PART 4: PML boundary zoom — raw Ez near domain edges
# =========================================================================
print(f"\n{'='*70}")
print("PART 4: PML boundary zoom — raw Ez fields")
print("=" * 70)

# Use late-time frame (t=1.20ps) to see standing waves and PML absorption
late_idx = -1  # last frame
rfx_late = rfx_f[late_idx]
meep_late = meep_l[late_idx]  # lag-compensated

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Row 1: Bottom edge (y=0..20) — PML-adjacent region
axes[0, 0].imshow(rfx_late[:, :20].T, origin="lower", cmap="RdBu_r",
                  vmin=-np.abs(rfx_late).max()*0.3, vmax=np.abs(rfx_late).max()*0.3)
axes[0, 0].set_title("rfx Ez (y=0..20, near PML)")
axes[0, 1].imshow(meep_late[:, :20].T, origin="lower", cmap="RdBu_r",
                  vmin=-np.abs(meep_late).max()*0.3, vmax=np.abs(meep_late).max()*0.3)
axes[0, 1].set_title("Meep Ez (y=0..20, near PML)")
diff_pml = rfx_late[:, :20] / max(np.abs(rfx_late).max(), 1e-30) - \
           meep_late[:, :20] / max(np.abs(meep_late).max(), 1e-30)
axes[0, 2].imshow(diff_pml.T, origin="lower", cmap="bwr",
                  vmin=-0.5, vmax=0.5)
axes[0, 2].set_title("Normalized diff (bottom PML)")

# Row 2: Right edge (x=140..160) — PML-adjacent region
axes[1, 0].imshow(rfx_late[140:, :].T, origin="lower", cmap="RdBu_r",
                  vmin=-np.abs(rfx_late).max()*0.3, vmax=np.abs(rfx_late).max()*0.3,
                  aspect='auto')
axes[1, 0].set_title("rfx Ez (x=140..end, near PML)")
axes[1, 1].imshow(meep_late[140:, :].T, origin="lower", cmap="RdBu_r",
                  vmin=-np.abs(meep_late).max()*0.3, vmax=np.abs(meep_late).max()*0.3,
                  aspect='auto')
axes[1, 1].set_title("Meep Ez (x=140..end, near PML)")
rx = rfx_late[140:, :] / max(np.abs(rfx_late).max(), 1e-30)
mx = meep_late[140:, :] / max(np.abs(meep_late).max(), 1e-30)
nc2 = min(rx.shape[0], mx.shape[0])
diff_pml_r = rx[:nc2, :] - mx[:nc2, :]
axes[1, 2].imshow(diff_pml_r.T, origin="lower", cmap="bwr",
                  vmin=-0.5, vmax=0.5, aspect='auto')
axes[1, 2].set_title("Normalized diff (right PML)")

for ax in axes.flat:
    ax.set_xlabel("x cell"); ax.set_ylabel("y cell")
plt.suptitle("Part 4: PML boundary region — raw Ez at t=1.20ps (lag-compensated)",
             fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "02_part4_pml_zoom.png"), dpi=150)
plt.close()

# =========================================================================
# PART 5: Bend corner zoom with raw Ez (not envelope)
# =========================================================================
print(f"\n{'='*70}")
print("PART 5: Bend corner zoom — raw Ez at t=0.40ps")
print("=" * 70)

# Use t=0.40ps frame for bend corner analysis
bend_idx = 2  # t=0.40ps
rfx_bend = rfx_f[bend_idx]
meep_bend_lag = meep_l[bend_idx]

# Bend corner region: x=65-100, y=65-100
bx = slice(65, 100)
by = slice(65, 100)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
vmax_bend = max(np.abs(rfx_bend[bx, by]).max(), np.abs(meep_bend_lag[bx, by]).max()) * 0.8

axes[0].imshow(rfx_bend[bx, by].T, origin="lower", cmap="RdBu_r",
               vmin=-vmax_bend, vmax=vmax_bend, extent=[65,100,65,100])
axes[0].set_title("rfx Ez (bend corner)")
axes[1].imshow(meep_bend_lag[bx, by].T, origin="lower", cmap="RdBu_r",
               vmin=-vmax_bend, vmax=vmax_bend, extent=[65,100,65,100])
axes[1].set_title("Meep Ez +114fs (bend corner)")
# Normalized diff
rn_b = rfx_bend[bx, by] / max(np.abs(rfx_bend[bx, by]).max(), 1e-30)
mn_b = meep_bend_lag[bx, by] / max(np.abs(meep_bend_lag[bx, by]).max(), 1e-30)
axes[2].imshow((rn_b - mn_b).T, origin="lower", cmap="bwr",
               vmin=-1, vmax=1, extent=[65,100,65,100])
axes[2].set_title("Normalized diff (bend)")
for ax in axes:
    ax.set_xlabel("x cell"); ax.set_ylabel("y cell")
    # Draw waveguide outline
    ax.axhline(75, color='lime', ls='--', lw=0.5, alpha=0.5)
    ax.axhline(85, color='lime', ls='--', lw=0.5, alpha=0.5)
    ax.axvline(75, color='lime', ls='--', lw=0.5, alpha=0.5)
    ax.axvline(85, color='lime', ls='--', lw=0.5, alpha=0.5)
plt.suptitle("Part 5: Bend corner raw Ez at t=0.40ps (lag-compensated)", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "02_part5_bend_zoom.png"), dpi=150)
plt.close()

# =========================================================================
# Summary
# =========================================================================
print(f"\n{'='*70}")
print("SUMMARY — Qualitative differences identified")
print("=" * 70)

# Compute correlations for summary
print("\n  Lag-compensated envelope correlations:")
for i, t_ps in enumerate(capture_times_rfx_ps):
    r_env = envelope_2d(rfx_f[i])
    ml_env = envelope_2d(meep_l[i])
    rn = r_env / max(r_env.max(), 1e-30)
    mln = ml_env / max(ml_env.max(), 1e-30)
    if np.std(rn) > 1e-20 and np.std(mln) > 1e-20:
        c = float(np.corrcoef(rn.ravel(), mln.ravel())[0, 1])
    else:
        c = float('nan')
    print(f"    t={t_ps:.2f}ps: env corr = {c:.4f}")

print("\n  Output files:")
for f in ["02_part1_epsilon.png", "02_part2_sigma.png",
          "02_part3_fields.png", "02_part4_pml_zoom.png",
          "02_part5_bend_zoom.png"]:
    print(f"    {f}")
print()
