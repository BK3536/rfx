"""Cross-validation: Kerr nonlinear — third harmonic generation.

Replicates: Meep nonlinear materials tutorial concept
Structure: Kerr (chi3) slab, monochromatic CW source at f0
Comparison: Third harmonic at 3*f0 should appear in transmitted spectrum
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

from rfx import Simulation, Box
from rfx.sources.sources import GaussianPulse
from rfx.grid import C0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

f0 = 5e9       # fundamental
chi3 = 1e-2    # strong Kerr nonlinearity (exaggerated for visibility)
eps_r = 2.25   # linear permittivity

print("=" * 60)
print("Cross-Validation: Kerr Third Harmonic Generation")
print("=" * 60)
print(f"Fundamental: {f0/1e9:.0f} GHz")
print(f"Kerr chi3: {chi3}")
print(f"Expected THG at: {3*f0/1e9:.0f} GHz")
print()

dx = 1.0e-3  # 1mm
domain_x = 0.10  # 100mm

sim = Simulation(
    freq_max=4 * f0,  # capture up to 4th harmonic
    domain=(domain_x, domain_x, dx),
    boundary="cpml",
    cpml_layers=10,
    dx=dx,
    mode="2d_tmz",
)

# Kerr slab in the middle
sim.add_material("kerr_medium", eps_r=eps_r, chi3=chi3)
slab_lo = domain_x * 0.4
slab_hi = domain_x * 0.6
cy = domain_x / 2
sim.add(Box((slab_lo, 0, 0), (slab_hi, domain_x, dx)),
        material="kerr_medium")

# Narrowband source (almost CW) for clean harmonic generation
sim.add_source((domain_x * 0.15, cy, 0), "ez",
               waveform=GaussianPulse(f0=f0, bandwidth=0.1))

# Transmitted probe (after slab)
sim.add_probe((domain_x * 0.85, cy, 0), "ez")

grid = sim._build_grid()
# Long run for narrowband: need many periods for spectral resolution
n_steps = int(np.ceil(20e-9 / grid.dt))
print(f"Grid: {grid.nx}x{grid.ny}x{grid.nz}, steps={n_steps}")

result = sim.run(n_steps=n_steps)

ts = np.array(result.time_series).ravel()

# FFT
nfft = len(ts) * 4
spec = np.abs(np.fft.rfft(ts, n=nfft))
freqs_hz = np.fft.rfftfreq(nfft, d=result.dt)
freqs_ghz = freqs_hz / 1e9

# Find fundamental and third harmonic peaks
fund_band = (freqs_ghz > f0 / 1e9 * 0.8) & (freqs_ghz < f0 / 1e9 * 1.2)
thg_band = (freqs_ghz > 3 * f0 / 1e9 * 0.8) & (freqs_ghz < 3 * f0 / 1e9 * 1.2)

peak_fund = np.max(spec[fund_band]) if np.any(fund_band) else 0
peak_thg = np.max(spec[thg_band]) if np.any(thg_band) else 0

thg_ratio = peak_thg / (peak_fund + 1e-30)

print(f"\nFundamental peak: {peak_fund:.4e}")
print(f"Third harmonic peak: {peak_thg:.4e}")
print(f"THG/fundamental ratio: {thg_ratio:.4e}")

if peak_fund > 1e-10 and peak_thg > peak_fund * 1e-6:
    print("PASS: third harmonic detected")
else:
    if peak_fund < 1e-10:
        print("FAIL: no fundamental signal")
    else:
        print(f"FAIL: THG too weak (ratio={thg_ratio:.2e})")

# Plot
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
band = (freqs_ghz > 0) & (freqs_ghz < 25)
ax.plot(freqs_ghz[band], 20 * np.log10(spec[band] / np.max(spec[band]) + 1e-30), "b-")
ax.axvline(f0 / 1e9, color="g", ls="--", alpha=0.5, label=f"f0={f0/1e9:.0f} GHz")
ax.axvline(3 * f0 / 1e9, color="r", ls="--", alpha=0.5, label=f"3f0={3*f0/1e9:.0f} GHz")
ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Spectrum (dB)")
ax.set_title(f"Kerr THG: chi3={chi3}, eps_r={eps_r}")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(-80, 5)

plt.tight_layout()
out = os.path.join(SCRIPT_DIR, "09_kerr_thg.png")
plt.savefig(out, dpi=150)
plt.close(fig)
print(f"Plot saved: {out}")
