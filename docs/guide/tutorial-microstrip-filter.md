# Tutorial: Coupled-Line Bandpass Filter Design

This tutorial walks through designing a coupled microstrip bandpass
filter from specification to simulation. The workflow covers analytical
even/odd mode impedance calculation, translating those into physical
geometry, simulating with rfx, and running a parametric sweep to
fine-tune the design.

**Prerequisites:** Completed the [Patch Antenna tutorial](tutorial-patch-antenna.md),
familiarity with S-parameter concepts, basic filter theory.

---

## 1. Design Specification

| Parameter | Value |
|---|---|
| Centre frequency f0 | 2.45 GHz |
| 3-dB bandwidth | ~100 MHz |
| Filter order | 2nd-order (single coupled section) |
| Substrate | FR4 (eps_r = 4.4, tan_d = 0.02, h = 1.6 mm) |
| Port impedance Z0 | 50 ohm |

A single coupled-line section produces a bandpass response through
even/odd mode coupling. While a real Chebyshev or Butterworth design
requires multiple sections, a single section demonstrates the core
physics and the rfx workflow clearly.

---

## 2. Analytical Design

### 2.1 Background: Even and Odd Mode Impedances

A pair of parallel coupled microstrip lines supports two propagation
modes:

- **Even mode** -- both lines at the same potential (currents in phase).
  The coupling gap acts like an open circuit.
- **Odd mode** -- lines at opposite potentials (currents anti-phase).
  The coupling gap acts like a short circuit.

For a coupled-line bandpass filter, the even and odd mode impedances
(Z0e, Z0o) determine the coupling strength and hence the bandwidth.
From Pozar (*Microwave Engineering*, 4th ed., Sec. 8.7):

```
Z0e = Z0 * sqrt(1 + J*Z0 + (J*Z0)^2)
Z0o = Z0 * sqrt(1 - J*Z0 + (J*Z0)^2)
```

where J is the coupling coefficient related to the fractional bandwidth:

```
J = (pi/2) * (BW / f0) / sqrt(g1)
```

For a single section with g1 = 1 (maximally flat prototype):

```
BW / f0 = 100 / 2450 ~ 0.041
J*Z0 ~ (pi/2) * 0.041 ~ 0.064
Z0e ~ 53.3 ohm
Z0o ~ 46.9 ohm
```

### 2.2 Coupled-Line Geometry

The even/odd mode impedances are controlled by two geometric parameters:
line width W and coupling gap g. For FR4 at 2.45 GHz:

- **Line width W ~ 3.0 mm** (close to the 50-ohm microstrip width on
  1.6 mm FR4).
- **Coupling gap g ~ 0.5--1.0 mm** -- tighter gaps increase coupling
  (decrease Z0o, increase Z0e).

The coupling length is approximately one quarter-wavelength at f0:

```
eps_eff = (eps_r + 1) / 2 + (eps_r - 1) / 2 * (1 + 12*h/W)^(-0.5)
lambda_eff = c / (f0 * sqrt(eps_eff))
coupling_length = lambda_eff / 4
```

For our parameters: eps_eff ~ 3.34, lambda_eff ~ 67 mm,
coupling_length ~ 17 mm. We will use 15 mm as a starting point and
then sweep to optimise.

### 2.3 Python Design Calculations

```python
import numpy as np

C0 = 2.998e8

# Substrate
f0    = 2.45e9
eps_r = 4.4
tan_d = 0.02
h     = 1.6e-3

# Line dimensions
W   = 3.0e-3       # line width
gap = 0.8e-3       # coupling gap
line_length = 15e-3 # coupling section length

# Effective permittivity
eps_eff = (eps_r + 1) / 2 + (eps_r - 1) / 2 * (1 + 12 * h / W) ** (-0.5)
lambda_eff = C0 / (f0 * np.sqrt(eps_eff))

print(f"eps_eff     = {eps_eff:.3f}")
print(f"lambda_eff  = {lambda_eff*1e3:.1f} mm")
print(f"lambda/4    = {lambda_eff/4*1e3:.1f} mm")
print(f"Line length = {line_length*1e3:.1f} mm")
```

---

## 3. Building the rfx Simulation

### 3.1 Domain and Materials

The domain must be large enough to contain both lines with adequate
margin for the CPML absorbing boundary.

```python
from rfx import Simulation, Box, GaussianPulse

# Domain sizing
margin_x = 5e-3
margin_y = 6e-3
total_width = 2 * W + gap           # both lines + gap
dom_x = line_length + 2 * margin_x
dom_y = total_width + 2 * margin_y
dom_z = h + 5e-3                    # substrate + air above

dx = 0.5e-3  # 0.5 mm cell size

sim = Simulation(
    freq_max=f0 * 2,
    domain=(dom_x, dom_y, dom_z),
    boundary="cpml",
    cpml_layers=8,
    dx=dx,
)

# FR4 substrate with loss
sigma_sub = 2 * np.pi * f0 * 8.854e-12 * eps_r * tan_d
sim.add_material("substrate", eps_r=eps_r, sigma=sigma_sub)
```

### 3.2 Geometry

The structure has four layers:

1. **Ground plane** at z = 0 (PEC).
2. **FR4 substrate** from z = 0 to z = h.
3. **Line 1** (input) on top of substrate.
4. **Line 2** (coupled output) parallel to Line 1, separated by the gap.

```python
# Ground plane
sim.add(Box((0, 0, 0), (dom_x, dom_y, 0)), material="pec")

# Substrate
sim.add(Box((0, 0, 0), (dom_x, dom_y, h)), material="substrate")

# Line 1 (input)
y1_lo = (dom_y - total_width) / 2.0
y1_hi = y1_lo + W
x_start = margin_x
x_end   = margin_x + line_length
sim.add(Box((x_start, y1_lo, h), (x_end, y1_hi, h)), material="pec")

# Line 2 (coupled output)
y2_lo = y1_hi + gap
y2_hi = y2_lo + W
sim.add(Box((x_start, y2_lo, h), (x_end, y2_hi, h)), material="pec")
```

### 3.3 Ports and Probes

We place a lumped port on Line 1 (input) for S-parameter extraction,
and probes on both lines to observe coupling.

```python
feed_y1 = (y1_lo + y1_hi) / 2.0    # centre of Line 1
feed_y2 = (y2_lo + y2_hi) / 2.0    # centre of Line 2

# Driven port on Line 1 (input end)
sim.add_port(
    (x_start + 1e-3, feed_y1, 0),
    component="ez",
    impedance=50.0,
    waveform=GaussianPulse(f0=f0, bandwidth=0.8),
    extent=h,
)

# Probe on Line 2 (coupled output, far end)
sim.add_probe((x_end - 1e-3, feed_y2, h / 2), component="ez")

# Probe on Line 1 (through port, far end)
sim.add_probe((x_end - 1e-3, feed_y1, h / 2), component="ez")
```

---

## 4. Running the Simulation

```python
grid = sim._build_grid()
n_steps = int(np.ceil(12e-9 / grid.dt))
print(f"Running {n_steps} steps (dt = {grid.dt*1e12:.2f} ps) ...")

result = sim.run(n_steps=n_steps, compute_s_params=True)
```

---

## 5. Analysing the Results

### 5.1 S-Parameter Extraction

The primary figures of merit for a bandpass filter are:

- **S21** (insertion loss / transmission): should peak near f0.
- **S11** (return loss / reflection): should dip near f0.

```python
import matplotlib.pyplot as plt

# Extract S-parameters
s11 = result.s_params[0, 0, :]
s11_dB = 20 * np.log10(np.abs(s11) + 1e-12)
freqs_GHz = result.freqs / 1e9

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(freqs_GHz, s11_dB, label="|S11| (dB)")
ax.axhline(-10, color="gray", ls=":", alpha=0.5, label="-10 dB ref")
ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Magnitude (dB)")
ax.set_title("Coupled-Line Filter: S11")
ax.set_xlim(1.0, 4.0)
ax.set_ylim(-30, 0)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("filter_s11.png", dpi=150)
```

### 5.2 Coupled vs Through Signal

The probes on Line 1 (through) and Line 2 (coupled) let us compute a
proxy for S21 by comparing their spectra:

```python
ts = np.asarray(result.time_series)
coupled_signal = ts[:, 0]    # Line 2 (coupled output)
through_signal = ts[:, 1]    # Line 1 (through)

nfft = len(coupled_signal) * 4
freqs_Hz = np.fft.rfftfreq(nfft, d=result.dt)
spec_coupled = np.abs(np.fft.rfft(coupled_signal, n=nfft))
spec_through = np.abs(np.fft.rfft(through_signal, n=nfft))

# Normalised coupled power (proxy for S21)
spec_ratio = spec_coupled / (spec_through.max() + 1e-30)
ratio_dB = 20 * np.log10(np.maximum(spec_ratio, 1e-10))

fig, ax = plt.subplots(figsize=(8, 5))
band = (freqs_Hz > 1e9) & (freqs_Hz < 5e9)
ax.plot(freqs_Hz[band] / 1e9, ratio_dB[band])
ax.axvline(f0 / 1e9, color="r", ls="--", label=f"f0 = {f0/1e9:.2f} GHz")
ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Coupled/Through (dB)")
ax.set_title("Coupling Response")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("filter_coupling.png", dpi=150)
```

**What to look for:**

- A peak in the coupled response near the design frequency f0.
- The peak width corresponds to the filter bandwidth.
- Energy at f0 should be selectively transferred from Line 1 to Line 2.

### 5.3 Quantitative Validation Checks

```python
# Find coupling peak
band_mask = (freqs_Hz > f0 * 0.3) & (freqs_Hz < f0 * 1.7)
peak_idx = np.argmax(spec_coupled[band_mask])
f_peak = freqs_Hz[band_mask][peak_idx]
freq_err = abs(f_peak - f0) / f0 * 100

# Energy coupling ratio
coupled_energy = np.sum(coupled_signal ** 2)
through_energy = np.sum(through_signal ** 2)
coupling_ratio = coupled_energy / (through_energy + 1e-30)

print(f"Coupling peak:  {f_peak/1e9:.2f} GHz (error: {freq_err:.1f}%)")
print(f"Coupling ratio: {coupling_ratio:.4f}")
print(f"Energy transfer: {'Good' if coupling_ratio > 1e-3 else 'Weak'}")
```

---

## 6. Parametric Sweep: Tuning the Gap

The coupling gap is the most sensitive parameter. A parametric sweep
reveals how it affects the filter response.

### 6.1 Using `parametric_sweep()`

```python
from rfx import parametric_sweep, plot_sweep

def build_filter(gap_val):
    """Build coupled-line filter with the given gap."""
    total_w = 2 * W + gap_val
    dy = total_w + 2 * margin_y

    sim = Simulation(
        freq_max=f0 * 2,
        domain=(dom_x, dy, dom_z),
        boundary="cpml",
        cpml_layers=8,
        dx=dx,
    )
    sim.add_material("substrate", eps_r=eps_r, sigma=sigma_sub)

    # Ground + substrate
    sim.add(Box((0, 0, 0), (dom_x, dy, 0)), material="pec")
    sim.add(Box((0, 0, 0), (dom_x, dy, h)), material="substrate")

    # Lines
    y1_lo = (dy - total_w) / 2.0
    y1_hi = y1_lo + W
    y2_lo = y1_hi + gap_val
    y2_hi = y2_lo + W
    sim.add(Box((x_start, y1_lo, h), (x_end, y1_hi, h)), material="pec")
    sim.add(Box((x_start, y2_lo, h), (x_end, y2_hi, h)), material="pec")

    # Port on Line 1
    fy1 = (y1_lo + y1_hi) / 2.0
    sim.add_port(
        (x_start + 1e-3, fy1, 0),
        component="ez", impedance=50.0,
        waveform=GaussianPulse(f0=f0, bandwidth=0.8),
        extent=h,
    )

    # Probes on both lines
    fy2 = (y2_lo + y2_hi) / 2.0
    sim.add_probe((x_end - 1e-3, fy2, h / 2), "ez")
    sim.add_probe((x_end - 1e-3, fy1, h / 2), "ez")

    return sim

# Sweep the gap from 0.4 mm to 1.6 mm
sweep = parametric_sweep(
    sim_factory=build_filter,
    param_name="Coupling gap (mm)",
    param_values=np.array([0.4, 0.6, 0.8, 1.0, 1.2, 1.6]) * 1e-3,
    n_steps=n_steps,
    run_kwargs={"compute_s_params": True},
)

fig = plot_sweep(sweep, metric="s11_min_db",
                 title="Gap vs Return Loss")
fig.savefig("filter_gap_sweep.png", dpi=150)
```

### 6.2 Custom Metric: Coupling Strength

You can also sweep with a custom metric function:

```python
def coupling_metric(result):
    """Coupling energy ratio (higher = stronger coupling)."""
    ts = np.asarray(result.time_series)
    if ts.ndim == 2 and ts.shape[1] >= 2:
        coupled = np.sum(ts[:, 0] ** 2)
        through = np.sum(ts[:, 1] ** 2)
        return coupled / (through + 1e-30)
    return 0.0

fig = plot_sweep(sweep, metric=coupling_metric,
                 title="Gap vs Coupling Strength",
                 ylabel="Coupled/Through energy ratio")
fig.savefig("filter_gap_coupling.png", dpi=150)
```

**Expected trend:**

- Smaller gaps produce stronger coupling (higher coupling ratio, deeper
  S11 notch).
- Gaps below ~0.5 mm may be below the mesh resolution (dx = 0.5 mm)
  and produce unreliable results. Refine the mesh if you need sub-mm
  gaps.

---

## 7. Comparing with Theory

For a quarter-wave coupled-line section, the coupling peak should occur
near f0 = c / (4 * L_coupled * sqrt(eps_eff_coupled)). The simulated
result may differ by 5--20% due to:

- Fringing fields at the open ends of the coupled lines.
- Even/odd mode velocity dispersion (different eps_eff for each mode).
- Finite substrate width and ground plane truncation.
- FDTD discretisation error.

These are real effects that analytical formulas cannot fully capture --
which is precisely why full-wave simulation is valuable.

---

## 8. Next Steps

- **Multi-section filter.** Add more coupled sections for sharper
  roll-off (higher order). Each section adds a pair of transmission
  zeros.
- **Gradient-based optimisation.** Use rfx's `optimize()` or
  `topology_optimize()` to automatically tune line widths and gaps
  for a target S21 mask.
- **Touchstone export.** Save S-parameters for import into circuit
  simulators:

  ```python
  from rfx import write_touchstone
  write_touchstone("filter.s1p", result.freqs, result.s_params)
  ```

- **Convergence study.** Run `convergence_study()` on the filter
  with the coupling peak frequency as the metric to verify grid
  independence (see the [Convergence tutorial](tutorial-convergence.md)).

---

## 9. Complete Example

The file `examples/15_coupled_filter.py` in the rfx repository contains
a self-contained validation script for this coupled-line filter. Run it
with:

```bash
python examples/15_coupled_filter.py
```

---

## References

- D. M. Pozar, *Microwave Engineering*, 4th ed., Wiley, 2012,
  Sec. 7.6 (coupled lines) and Sec. 8.7 (coupled-line filters).
- G. Matthaei, L. Young, E. M. T. Jones, *Microwave Filters,
  Impedance-Matching Networks, and Coupling Structures*, Artech House,
  1980, Ch. 5.
- R. S. Elliott, *An Introduction to Guided Waves and Microwave
  Circuits*, Prentice Hall, 1993, Ch. 7.
