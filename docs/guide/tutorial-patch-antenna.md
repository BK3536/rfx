# Tutorial: 2.4 GHz Patch Antenna Design

This tutorial walks through the complete design flow for a rectangular
microstrip patch antenna on FR4 substrate using rfx. By the end you will
have a working simulation that predicts the resonant frequency, extracts
the Q factor, and visualises the result -- all from first principles.

**Prerequisites:** rfx installed (`pip install -e .`), basic NumPy/JAX
familiarity, undergraduate-level microwave engineering.

---

## 1. Analytical Design

Every patch antenna design starts with closed-form equations that give a
good initial geometry. We use the Hammerstad formula set, which is the
standard reference for rectangular microstrip patch antennas (Balanis,
*Antenna Theory*, 4th ed., Ch. 14; Hammerstad, 1975).

### 1.1 Specifications

| Parameter | Value |
|---|---|
| Centre frequency f0 | 2.4 GHz |
| Substrate | FR4 (eps_r = 4.4, tan_d = 0.02) |
| Substrate thickness h | 1.6 mm |
| Feed type | Probe-style soft source |

### 1.2 Patch Width

The radiating width W is chosen so that the patch radiates efficiently.
The standard expression is:

```
W = c / (2 * f0) * sqrt(2 / (eps_r + 1))
```

For our parameters: W = 38.04 mm.

### 1.3 Effective Permittivity

A microstrip patch sits at the air--dielectric interface, so the fields
see an effective permittivity somewhere between 1 and eps_r:

```
eps_eff = (eps_r + 1) / 2 + (eps_r - 1) / 2 * (1 + 12 * h / W)^(-0.5)
```

For our patch: eps_eff ~ 4.07.

### 1.4 Physical Length

The electrical length of the patch is lambda_eff / 2, but fringing
fields at each open edge make the patch appear electrically longer than
its physical dimension. The Hammerstad extension correction is:

```
dL = 0.412 * h * (eps_eff + 0.3) * (W/h + 0.264)
                / ((eps_eff - 0.258) * (W/h + 0.8))
```

The physical length is then:

```
L_eff = c / (2 * f0 * sqrt(eps_eff))
L     = L_eff - 2 * dL
```

For our patch: L ~ 29.0 mm. These equations give us the starting-point
geometry; the FDTD simulation will tell us how accurate this initial
design really is.

---

## 2. Setting Up the Simulation

### 2.1 Full Script

```python
import numpy as np
from rfx import Simulation, Box, GaussianPulse

# ---- Physical constants ----
C0 = 2.998e8  # m/s

# ---- Design parameters ----
f0    = 2.4e9       # target frequency (Hz)
eps_r = 4.4         # FR4 relative permittivity
tan_d = 0.02        # FR4 loss tangent
h     = 1.6e-3      # substrate thickness (m)

# ---- Hammerstad patch dimensions ----
W = C0 / (2 * f0) * np.sqrt(2.0 / (eps_r + 1.0))

eps_eff = (
    (eps_r + 1.0) / 2.0
    + (eps_r - 1.0) / 2.0 * (1.0 + 12.0 * h / W) ** (-0.5)
)

dL = 0.412 * h * (
    (eps_eff + 0.3) * (W / h + 0.264)
    / ((eps_eff - 0.258) * (W / h + 0.8))
)

L = C0 / (2.0 * f0 * np.sqrt(eps_eff)) - 2.0 * dL

print(f"Patch: L = {L*1e3:.2f} mm, W = {W*1e3:.2f} mm")
print(f"eps_eff = {eps_eff:.4f}")
```

### 2.2 Mesh and Domain

Two key decisions: cell size `dx` and domain margins.

**Cell size.** The rule of thumb is dx <= lambda_min / 20. At 2.4 GHz
the free-space wavelength is 125 mm, so dx = 0.5 mm (lambda/250) is
very comfortable. More importantly, the 1.6 mm substrate must be
resolved in z. With a uniform grid, 1.6 mm / 0.5 mm = 3.2 cells -- just
barely adequate. We use non-uniform z meshing (`dz_profile`) to place
finer cells inside the substrate without over-refining the air region.

**Margins.** The CPML absorbing boundary needs some clearance. A margin
of ~15 mm (roughly lambda/8) around the patch keeps CPML reflections
negligible for this resonance study.

```python
dx     = 0.5e-3     # lateral cell size (m)
margin = 15e-3      # air margin around patch (m)
dom_x  = L + 2 * margin
dom_y  = W + 2 * margin

# Non-uniform z-profile: fine cells in substrate, coarse in air
n_sub = max(4, int(np.ceil(h / dx)))   # >= 4 cells through substrate
dz_sub = h / n_sub                      # fine z cell
n_air  = max(6, int(np.ceil(margin / dx)))
dz_profile = np.concatenate([
    np.full(n_sub, dz_sub),   # substrate region
    np.full(n_air, dx),       # air above
])
```

**Why non-uniform z?** A uniform 0.5 mm grid gives only ~3 cells through
the 1.6 mm substrate -- too few to resolve the vertical field variation.
With `dz_profile` we pack 4 cells of 0.4 mm each into the substrate
while keeping coarser 0.5 mm cells in the air. This is one of rfx's
key features for practical PCB-like structures (see the
[Non-Uniform Mesh guide](nonuniform_mesh.md) for details).

### 2.3 Creating the Simulation Object

```python
sim = Simulation(
    freq_max=f0 * 2.0,        # sets auto-dx upper bound
    domain=(dom_x, dom_y, 0), # z=0 sentinel; actual z from dz_profile
    dx=dx,
    dz_profile=dz_profile,
    cpml_layers=12,
)
```

The `domain` z-component is set to 0 because the actual z extent comes
from `dz_profile`. `freq_max` should be at least 2x the centre
frequency so the Gaussian pulse has adequate spectral content.

### 2.4 Materials and Geometry

A microstrip patch antenna has three layers:

1. **Ground plane** -- PEC at z = 0.
2. **Substrate** -- FR4 dielectric filling the space from z = 0 to z = h.
3. **Patch** -- PEC rectangle on top of the substrate at z = h.

```python
# Substrate material: convert loss tangent to conductivity
sigma_sub = 2.0 * np.pi * f0 * 8.854e-12 * eps_r * tan_d

sim.add_material("substrate", eps_r=eps_r, sigma=sigma_sub)

# Ground plane (z = 0)
sim.add(Box((0, 0, 0), (dom_x, dom_y, 0)), material="pec")

# FR4 substrate
sim.add(Box((0, 0, 0), (dom_x, dom_y, h)), material="substrate")

# Patch on top of substrate
px0, py0 = margin, margin
sim.add(Box((px0, py0, h), (px0 + L, py0 + W, h)), material="pec")
```

**Note on loss tangent conversion.** FDTD materials use conductivity
sigma (S/m), not loss tangent directly. The approximate relationship at
a single frequency is sigma = 2 * pi * f0 * eps_0 * eps_r * tan_d.
This is exact only at f0 but adequate for narrowband patch antennas.

### 2.5 Excitation and Probe

For resonance extraction we use a soft source with a broadband Gaussian
pulse. The source is placed inside the substrate at L/3 from the patch
edge -- a standard offset that couples well to the fundamental TM010
mode.

```python
# Feed location: L/3 from edge, centred in y
src_x = px0 + L / 3.0
src_y = py0 + W / 2.0
src_z = h / 2.0   # mid-substrate

sim.add_source(
    (src_x, src_y, src_z),
    component="ez",
    waveform=GaussianPulse(f0=f0, bandwidth=0.8),
)
sim.add_probe((src_x, src_y, src_z), component="ez")
```

**Why a soft source, not a port?** For resonance extraction we only need
the ring-down behaviour to identify the resonant frequency and Q factor.
A port with impedance loading is needed when you want calibrated
S-parameters (S11), but for a first-pass design a soft source is simpler
and well-validated in rfx (see the [Validation guide](validation.md)).

---

## 3. Running and Interpreting Results

### 3.1 Running the Simulation

The simulation must run long enough for the Gaussian pulse to excite the
patch and for the resonant ring-down to be captured. About 15 ns is
sufficient for a Q ~ 30-50 patch at 2.4 GHz (the ring-down decays as
exp(-pi * f0 * t / Q), so 15 ns gives ~86 cycles).

```python
nu_grid = sim._build_nonuniform_grid()
n_steps = int(np.ceil(15e-9 / nu_grid.dt))

print(f"Running {n_steps} steps (dt = {nu_grid.dt*1e12:.3f} ps) ...")
result = sim.run(n_steps=n_steps)
```

Alternatively, use the automatic decay criterion to let rfx decide:

```python
result = sim.run(until_decay=1e-3)
```

This stops when the probe signal drops to 0.1% of its peak -- often the
more robust choice when you do not know the Q factor in advance.

### 3.2 Resonance Extraction with Harminv

rfx wraps Harminv (a filter-diagonalisation algorithm) to extract
resonant frequencies and Q factors from the time-domain ring-down. This
is far more accurate than raw FFT peak-finding.

```python
modes = result.find_resonances(
    freq_range=(f0 * 0.5, f0 * 1.5),
    probe_idx=0,
)

if modes:
    best = min(modes, key=lambda m: abs(m.freq - f0))
    print(f"Resonant frequency: {best.freq / 1e9:.4f} GHz")
    print(f"Q factor:           {best.Q:.1f}")
    err_pct = abs(best.freq - f0) / f0 * 100
    print(f"Error vs. design:   {err_pct:.2f}%")
```

**Expected result.** With the Hammerstad formula, the simulated
resonance typically lands within 1--3% of the 2.4 GHz design target.
The discrepancy arises from grid discretisation and the simplified
feed model.

### 3.3 Frequency-Domain Analysis via FFT

For quick checks without Harminv, you can use a zero-padded FFT:

```python
ts = np.asarray(result.time_series).ravel()
nfft = len(ts) * 8            # 8x zero-padding for smooth spectrum
spectrum = np.abs(np.fft.rfft(ts, n=nfft))
freqs = np.fft.rfftfreq(nfft, d=result.dt) / 1e9  # GHz

# Plot spectrum around the design frequency
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4))
band = (freqs > 1.0) & (freqs < 4.0)
spec_dB = 20 * np.log10(spectrum / spectrum.max() + 1e-12)
ax.plot(freqs[band], spec_dB[band])
ax.axvline(f0 / 1e9, color="r", ls="--", label=f"Design: {f0/1e9:.2f} GHz")
ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Normalised spectrum (dB)")
ax.set_title("Patch Antenna Resonance")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("patch_spectrum.png", dpi=150)
```

### 3.4 Input Impedance on a Smith Chart

If you want impedance information, switch from `add_source` to
`add_port` and enable S-parameter extraction:

```python
# Replace add_source with add_port for S-parameter extraction
sim.add_port(
    position=(src_x, src_y, 0),
    component="ez",
    impedance=50.0,
    extent=h,     # wire port spanning ground to patch
    waveform=GaussianPulse(f0=f0, bandwidth=0.8),
)

result = sim.run(n_steps=n_steps, compute_s_params=True)

# S11 in dB
s11 = result.s_params[0, 0, :]
s11_dB = 20 * np.log10(np.abs(s11) + 1e-12)

# Smith chart
from rfx import plot_smith
plot_smith(s11, freqs=result.freqs, title="Patch Antenna S11")
```

**Caution.** The lumped-port feed model is a simplified representation
of a real probe feed. The resonant frequency from the port run should
be close to the source-based result, but the absolute S11 level depends
on feed-point location and port extent. Treat the Smith chart as a
qualitative guide at this stage. See the
[Validation guide](validation.md) for current feed/port caveats.

---

## 4. Mesh Convergence Study

Never trust a single simulation at one grid resolution. A convergence
study verifies that your result is not an artefact of the mesh.

### 4.1 Using `convergence_study()`

rfx provides a built-in convergence study tool. You supply a factory
function that builds the simulation at each cell size and a metric
function that extracts the quantity of interest.

```python
from rfx import convergence_study

def build_patch_sim(dx):
    """Build a patch antenna simulation at the given cell size."""
    n_sub = max(4, int(np.ceil(h / dx)))
    dz_sub = h / n_sub
    n_air = max(6, int(np.ceil(margin / dx)))
    dz_prof = np.concatenate([
        np.full(n_sub, dz_sub),
        np.full(n_air, dx),
    ])

    sim = Simulation(
        freq_max=f0 * 2.0,
        domain=(L + 2 * margin, W + 2 * margin, 0),
        dx=dx,
        dz_profile=dz_prof,
        cpml_layers=12,
    )
    sigma_s = 2 * np.pi * f0 * 8.854e-12 * eps_r * tan_d
    sim.add_material("substrate", eps_r=eps_r, sigma=sigma_s)
    sim.add(Box((0, 0, 0), (dom_x, dom_y, 0)), material="pec")
    sim.add(Box((0, 0, 0), (dom_x, dom_y, h)), material="substrate")
    sim.add(Box((margin, margin, h),
                (margin + L, margin + W, h)), material="pec")
    sim.add_source(
        (margin + L / 3, margin + W / 2, h / 2),
        component="ez",
        waveform=GaussianPulse(f0=f0, bandwidth=0.8),
    )
    sim.add_probe((margin + L / 3, margin + W / 2, h / 2), "ez")
    return sim

def extract_resonance(result):
    """Extract the resonant frequency closest to f0."""
    modes = result.find_resonances(freq_range=(f0 * 0.5, f0 * 1.5))
    if modes:
        best = min(modes, key=lambda m: abs(m.freq - f0))
        return best.freq
    # FFT fallback
    ts = np.asarray(result.time_series).ravel()
    spec = np.abs(np.fft.rfft(ts, n=len(ts) * 8))
    freqs = np.fft.rfftfreq(len(ts) * 8, d=result.dt)
    band = (freqs > f0 * 0.5) & (freqs < f0 * 1.5)
    return float(freqs[np.argmax(spec * band)])

conv = convergence_study(
    sim_factory=build_patch_sim,
    dx_values=[1.0e-3, 0.75e-3, 0.5e-3],
    metric_fn=extract_resonance,
    until_decay=1e-3,
)
```

### 4.2 Interpreting the Results

```python
# Print the convergence table
conv.summary()

# Two-panel plot: metric vs dx and log-log error vs dx
fig = conv.plot(title="Patch Antenna Convergence")
fig.savefig("patch_convergence.png", dpi=150)
```

The summary includes a Richardson-extrapolated estimate of the
grid-independent resonant frequency and the observed convergence order.
For the standard Yee scheme on axis-aligned structures, you should see
approximately second-order convergence (order ~ 2).

**What to look for:**

- The metric should monotonically approach the extrapolated value as dx
  decreases.
- The relative error on the log-log plot should follow a straight line
  with slope ~ 2.
- If the error plateaus, you may be limited by other factors (PML
  reflections, simulation time, feed model).

### 4.3 Quick Convergence Shortcut

If you already have a configured `Simulation` object you can use
`quick_convergence()`, which clones the simulation at multiple dx
values automatically:

```python
from rfx import quick_convergence

conv = quick_convergence(
    sim,
    metric="resonance",
    dx_factors=[2.0, 1.5, 1.0, 0.75],
    until_decay=1e-3,
)
conv.summary()
```

---

## 5. Bandwidth Enhancement via Topology Optimization (Optional)

One of rfx's unique strengths is differentiable simulation. You can
optimise the patch shape to improve bandwidth using gradient-based
topology optimisation.

```python
from rfx import topology_optimize, TopologyDesignRegion
import jax.numpy as jnp

# Set up a port-based simulation for S11
sim_opt = Simulation(
    freq_max=f0 * 2.0,
    domain=(dom_x, dom_y, 0),
    dx=dx,
    dz_profile=dz_profile,
    cpml_layers=12,
)
sim_opt.add_material("substrate", eps_r=eps_r, sigma=sigma_sub)
sim_opt.add(Box((0, 0, 0), (dom_x, dom_y, 0)), material="pec")
sim_opt.add(Box((0, 0, 0), (dom_x, dom_y, h)), material="substrate")
sim_opt.add_port(
    position=(px0 + L / 3, py0 + W / 2, 0),
    component="ez",
    impedance=50.0,
    extent=h,
    waveform=GaussianPulse(f0=f0, bandwidth=0.8),
)

# Define a design region covering the patch area
region = TopologyDesignRegion(
    corner_lo=(px0, py0, h),
    corner_hi=(px0 + L, py0 + W, h),
    material_bg="air",
    material_fg="pec",
    filter_radius=1.5e-3,
)

# Objective: maximise bandwidth (minimise S11 over a band)
def bandwidth_objective(result):
    s11 = result.s_params[0, 0, :]
    return jnp.mean(jnp.abs(s11) ** 2)

topo_result = topology_optimize(
    sim_opt, region,
    objective=bandwidth_objective,
    n_iterations=50,
)

print(f"Initial loss: {topo_result.loss_history[0]:.4e}")
print(f"Final loss:   {topo_result.loss_history[-1]:.4e}")
```

This is an advanced workflow; start with the analytical design and
convergence study before attempting topology optimisation.

---

## 6. Complete Example Script

The file `examples/04_patch_antenna.py` in the rfx repository contains
the full working script with a 6-panel visualisation (geometry slices,
time-domain signal, spectrum, and summary annotation). Run it with:

```bash
python examples/04_patch_antenna.py
```

---

## 7. Summary and Key Takeaways

1. **Analytical first.** The Hammerstad formulas give a reliable starting
   point. The FDTD simulation validates and refines the design.

2. **Non-uniform z mesh.** For thin substrates, use `dz_profile` to
   resolve the dielectric layer without over-refining the air region.

3. **Soft source for resonance.** Use `add_source()` + Harminv for
   accurate resonance and Q extraction. Switch to `add_port()` only
   when you need S-parameters.

4. **Always check convergence.** Use `convergence_study()` or
   `quick_convergence()` to verify that your result is grid-independent.

5. **Differentiable advantage.** Once the baseline design is validated,
   rfx's JAX-based topology optimisation can improve bandwidth or
   matching beyond what analytical formulas achieve.

---

## References

- C. A. Balanis, *Antenna Theory: Analysis and Design*, 4th ed., Wiley,
  2016, Ch. 14.
- E. O. Hammerstad, "Equations for Microstrip Circuit Design," *Proc.
  European Microwave Conf.*, 1975, pp. 268--272.
- D. M. Pozar, *Microwave Engineering*, 4th ed., Wiley, 2012, Ch. 3.
- V. A. Mandelshtam and H. S. Taylor, "Harmonic inversion of time
  signals and its applications," *J. Chem. Phys.*, 107(17), 1997
  (Harminv algorithm).
