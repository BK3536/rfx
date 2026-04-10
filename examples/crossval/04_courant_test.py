"""Courant number test: Does matching S=0.5 close the speed gap?

Hypothesis: rfx S=0.700 ≈ magic step (1/√2=0.707) gives near-zero
numerical dispersion for axis-aligned waves, while Meep S=0.500
has more dispersion. This makes rfx fields propagate faster = closer to
physical speed.

Test: Override rfx dt to match Meep's Courant, run same physical time,
compare field distributions visually.
"""

import os, sys, math
os.environ.setdefault("JAX_ENABLE_X64", "1")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
C0 = 2.998e8

a = 1.0e-6
eps_wg = 12.0
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
pml_cells = int(dpml * resolution)

import meep as mp
from rfx import Simulation, Box
from rfx.sources.sources import ModulatedGaussian
from rfx.simulation import SnapshotSpec
import jax.numpy as jnp

# Physical simulation time (~1.4 ps)
T_physical = 1.4e-12  # same as 6000 rfx steps at normal CFL

# =========================================================================
# Run rfx at NORMAL CFL (S=0.700)
# =========================================================================
print("=" * 70)
print("rfx at S=0.700 (normal CFL)")
print("=" * 70)

sim_rfx1 = Simulation(freq_max=0.25 * C0 / a, domain=(domain_x, domain_y, dx),
                       dx=dx, boundary="upml", cpml_layers=cpml_n, mode="2d_tmz")
sim_rfx1.add_material("wg", eps_r=eps_wg)
sim_rfx1.add(Box((0, 7.5*a, 0), (8.5*a, 8.5*a, dx)), material="wg")
sim_rfx1.add(Box((7.5*a, 7.5*a, 0), (8.5*a, 16*a, dx)), material="wg")
for i in range(10):
    y = 7.5*a + (i + 0.5) * a / 10
    sim_rfx1.add_source(position=(src_x_rfx, y, 0), component="ez",
        waveform=ModulatedGaussian(f0=fcen_hz, bandwidth=bw_rfx,
                                   amplitude=1.0/10, cutoff=5.0/math.sqrt(2)))
sim_rfx1.add_probe(position=(COORD_OFFSET*a, COORD_OFFSET*a, 0), component="ez")
snap = SnapshotSpec(components=("ez",), slice_axis=2, slice_index=0)

n_steps_1 = int(T_physical / (0.99 * dx / (C0 * math.sqrt(2))))
print(f"  dt = {0.99*dx/(C0*math.sqrt(2)):.4e} s, n_steps = {n_steps_1}")
res1 = sim_rfx1.run(n_steps=n_steps_1, snapshot=snap, subpixel_smoothing=True)

grid1 = sim_rfx1._build_grid()
pad1 = grid1.pad_x
n_domain = int(np.ceil(domain_x / dx)) + 1

# =========================================================================
# Run rfx at MEEP'S CFL (S=0.500)
# =========================================================================
print(f"\n{'='*70}")
print("rfx at S=0.500 (Meep's Courant)")
print("=" * 70)

# Build simulation normally, then override dt before running
sim_rfx2 = Simulation(freq_max=0.25 * C0 / a, domain=(domain_x, domain_y, dx),
                       dx=dx, boundary="upml", cpml_layers=cpml_n, mode="2d_tmz")
sim_rfx2.add_material("wg", eps_r=eps_wg)
sim_rfx2.add(Box((0, 7.5*a, 0), (8.5*a, 8.5*a, dx)), material="wg")
sim_rfx2.add(Box((7.5*a, 7.5*a, 0), (8.5*a, 16*a, dx)), material="wg")
for i in range(10):
    y = 7.5*a + (i + 0.5) * a / 10
    sim_rfx2.add_source(position=(src_x_rfx, y, 0), component="ez",
        waveform=ModulatedGaussian(f0=fcen_hz, bandwidth=bw_rfx,
                                   amplitude=1.0/10, cutoff=5.0/math.sqrt(2)))
sim_rfx2.add_probe(position=(COORD_OFFSET*a, COORD_OFFSET*a, 0), component="ez")

# Override the grid dt to match Meep's Courant by monkey-patching _build_grid
meep_dt = 0.5 * dx / C0  # S=0.5
_orig_build = sim_rfx2._build_grid
def _patched_build(**kwargs):
    g = _orig_build(**kwargs)
    g.dt = meep_dt
    return g
sim_rfx2._build_grid = _patched_build

n_steps_2 = int(T_physical / meep_dt)
print(f"  dt = {meep_dt:.4e} s (overridden), n_steps = {n_steps_2}")

res2 = sim_rfx2.run(n_steps=n_steps_2, snapshot=snap, subpixel_smoothing=True)

pad2 = pad1  # Same grid layout, just different dt

# =========================================================================
# Run Meep (S=0.500)
# =========================================================================
print(f"\n{'='*70}")
print("Meep at S=0.500 (default)")
print("=" * 70)

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
src_meep = [mp.Source(mp.GaussianSource(fcen, fwidth=fwidth),
                      component=mp.Ez,
                      center=mp.Vector3(src_x_meep, 0),
                      size=mp.Vector3(0, 1.0))]
sim_meep = mp.Simulation(cell_size=cell_meep, boundary_layers=pml_meep,
                         geometry=geo_meep, sources=src_meep, resolution=resolution)
sim_meep.init_sim()

capture_times_ps = [0.10, 0.20, 0.35, 0.50, 0.80, 1.20]
meep_times = [t*1e-12 * C0 / a for t in capture_times_ps]

meep_frames = []
for ci, target_t in enumerate(meep_times):
    remaining = target_t - sim_meep.meep_time()
    if remaining > 0:
        sim_meep.run(until=remaining)
    ez = sim_meep.get_array(center=mp.Vector3(), size=cell_meep, component=mp.Ez)
    meep_frames.append(ez[pml_cells:-pml_cells, pml_cells:-pml_cells].copy())
meep_frames = np.array(meep_frames)

# =========================================================================
# Extract rfx frames at same physical times
# =========================================================================
dt1 = res1.dt
dt2 = res2.dt

rfx1_all = np.asarray(res1.snapshots["ez"])[:, pad1:pad1+n_domain, pad1:pad1+n_domain]
rfx2_all = np.asarray(res2.snapshots["ez"])[:, pad2:pad2+n_domain, pad2:pad2+n_domain]

rfx1_steps = [min(n_steps_1-1, int(t*1e-12 / dt1)) for t in capture_times_ps]
rfx2_steps = [min(n_steps_2-1, int(t*1e-12 / dt2)) for t in capture_times_ps]

rfx1_frames = np.array([rfx1_all[s] for s in rfx1_steps])
rfx2_frames = np.array([rfx2_all[s] for s in rfx2_steps])

n_c = min(rfx1_frames.shape[1], meep_frames.shape[1])
rfx1_f = rfx1_frames[:, :n_c, :n_c]
rfx2_f = rfx2_frames[:, :n_c, :n_c]
meep_f = meep_frames[:, :n_c, :n_c]

def envelope_2d(field):
    env = np.zeros_like(field)
    for j in range(field.shape[1]):
        env[:, j] = np.abs(hilbert(field[:, j]))
    return env

def corr(a, b):
    if np.std(a) > 1e-20 and np.std(b) > 1e-20:
        return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    return float('nan')

# =========================================================================
# Compare: rfx(S=0.7) vs Meep vs rfx(S=0.5) vs Meep
# =========================================================================
print(f"\n{'='*70}")
print("Field comparison: does matching Courant close the gap?")
print("=" * 70)

print(f"\n  {'t (ps)':>8} {'rfx(0.7) vs Meep':>18} {'rfx(0.5) vs Meep':>18}")
print("  " + "-" * 50)

corr_07 = []
corr_05 = []
for i, t_ps in enumerate(capture_times_ps):
    r1_env = envelope_2d(rfx1_f[i])
    r2_env = envelope_2d(rfx2_f[i])
    m_env = envelope_2d(meep_f[i])

    r1n = r1_env / max(r1_env.max(), 1e-30)
    r2n = r2_env / max(r2_env.max(), 1e-30)
    mn = m_env / max(m_env.max(), 1e-30)

    c07 = corr(r1n, mn)
    c05 = corr(r2n, mn)
    corr_07.append(c07)
    corr_05.append(c05)
    marker = " ←" if abs(c05 - c07) > 0.02 and c05 > c07 else ""
    print(f"  {t_ps:>8.2f} {c07:>18.4f} {c05:>18.4f}{marker}")

print(f"\n  Mean:    {np.nanmean(corr_07):>18.4f} {np.nanmean(corr_05):>18.4f}")

# =========================================================================
# Visual comparison: 3-column (rfx S=0.7, rfx S=0.5, Meep)
# =========================================================================
fig, axes = plt.subplots(len(capture_times_ps), 4,
                          figsize=(22, 4*len(capture_times_ps)))

for i, t_ps in enumerate(capture_times_ps):
    r1_env = envelope_2d(rfx1_f[i])
    r2_env = envelope_2d(rfx2_f[i])
    m_env = envelope_2d(meep_f[i])

    r1n = r1_env / max(r1_env.max(), 1e-30)
    r2n = r2_env / max(r2_env.max(), 1e-30)
    mn = m_env / max(m_env.max(), 1e-30)
    vmax = max(r1n.max(), r2n.max(), mn.max()) * 0.9 or 1.0

    axes[i, 0].imshow(r1n.T, origin="lower", cmap="hot", vmin=0, vmax=vmax)
    axes[i, 0].set_title(f"rfx S=0.70 (t={t_ps:.2f}ps)", fontsize=10)
    axes[i, 0].set_ylabel("y")

    axes[i, 1].imshow(r2n.T, origin="lower", cmap="hot", vmin=0, vmax=vmax)
    axes[i, 1].set_title(f"rfx S=0.50 (corr={corr_05[i]:.3f})", fontsize=10)

    axes[i, 2].imshow(mn.T, origin="lower", cmap="hot", vmin=0, vmax=vmax)
    axes[i, 2].set_title(f"Meep S=0.50 (corr@0.7={corr_07[i]:.3f})", fontsize=10)

    # Diff: rfx(S=0.5) - Meep
    diff = r2n - mn
    vd = max(np.abs(diff).max(), 1e-30)
    axes[i, 3].imshow(diff.T, origin="lower", cmap="bwr", vmin=-vd, vmax=vd)
    axes[i, 3].set_title(f"rfx(0.5)-Meep", fontsize=10)

axes[-1, 0].set_xlabel("x"); axes[-1, 1].set_xlabel("x")
axes[-1, 2].set_xlabel("x"); axes[-1, 3].set_xlabel("x")
fig.suptitle(f"Courant test: rfx S=0.70 vs rfx S=0.50 vs Meep S=0.50\n"
             f"Mean env corr: S=0.7→{np.nanmean(corr_07):.3f}, S=0.5→{np.nanmean(corr_05):.3f}",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "04_courant_test.png"), dpi=150)
plt.close()

print(f"\n  Output: 04_courant_test.png")
