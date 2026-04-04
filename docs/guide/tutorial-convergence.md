# Tutorial: Mesh Convergence and Result Verification

How do you know whether your simulation result is accurate? A single run
at one grid resolution tells you what the simulator computed, but not
whether the answer has converged to the true physical result. This
tutorial teaches you how to perform a rigorous mesh convergence study,
interpret the results, and avoid common pitfalls that produce misleading
answers.

**Prerequisites:** Basic rfx usage ([Quick Start](quickstart.md)),
familiarity with at least one of the other tutorials
([Patch Antenna](tutorial-patch-antenna.md) or
[Microstrip Filter](tutorial-microstrip-filter.md)).

---

## 1. Why Convergence Matters

FDTD solves Maxwell's equations on a discrete grid. The grid spacing
`dx` introduces a numerical error that scales as O(dx^p), where p is the
convergence order (p = 2 for the standard Yee scheme on axis-aligned
structures).

**The problem.** If you pick one arbitrary dx value and run a single
simulation, you cannot distinguish between:

- A well-resolved simulation with a trustworthy answer.
- An under-resolved simulation that happens to give a plausible-looking
  (but wrong) answer.

**The solution.** Run the same problem at multiple grid resolutions.
If the metric of interest (resonant frequency, S11 minimum, peak field,
etc.) converges to a stable value as dx decreases, you can trust the
result. If it does not, you need a finer mesh.

---

## 2. Theory: Richardson Extrapolation

Given simulation results at two or more grid sizes, Richardson
extrapolation estimates the grid-independent value (dx -> 0). For a
quantity f(dx) with convergence order p:

```
f(dx) = f_exact + C * dx^p + O(dx^(p+1))
```

With two grid sizes dx1 and dx2 (and r = dx1/dx2):

```
f_exact ~ (r^p * f(dx2) - f(dx1)) / (r^p - 1)
```

With three or more grid sizes, rfx also estimates the actual convergence
order p from the data, rather than assuming p = 2. This is valuable
because real problems often have effective orders that differ from the
theoretical value due to PML effects, stairstepping, source
singularities, and other factors.

---

## 3. Using `convergence_study()`

rfx provides a structured convergence study tool. You supply three
things:

1. **`sim_factory(dx)`** -- a function that builds a complete
   `Simulation` object at the given cell size.
2. **`dx_values`** -- a list of cell sizes to test (coarse to fine).
3. **`metric_fn(result)`** -- a function that extracts the scalar
   quantity of interest from a simulation result.

### 3.1 Minimal Example: Cavity Resonance

A PEC cavity has an analytical solution, making it the ideal convergence
test case.

```python
import numpy as np
from rfx import Simulation, Box, GaussianPulse, convergence_study

# Analytical TE101 frequency for a 40x20x15 mm PEC cavity
a, b, d = 0.04, 0.02, 0.015  # metres
C0 = 2.998e8
f_te101 = (C0 / 2) * np.sqrt((1/a)**2 + (0/b)**2 + (1/d)**2)
print(f"Analytical TE101: {f_te101/1e9:.4f} GHz")

def build_cavity(dx):
    """Build a PEC cavity simulation at cell size dx."""
    sim = Simulation(
        freq_max=12e9,
        domain=(a, b, d),
        boundary="pec",
        dx=dx,
    )
    # Soft source off-centre to excite multiple modes
    sim.add_source(
        (a * 0.3, b * 0.4, d * 0.3),
        component="ez",
        waveform=GaussianPulse(f0=6e9, bandwidth=0.8),
    )
    sim.add_probe((a * 0.3, b * 0.4, d * 0.3), "ez")
    return sim

def extract_te101(result):
    """Extract the resonance closest to the TE101 frequency."""
    modes = result.find_resonances(freq_range=(4e9, 10e9))
    if modes:
        best = min(modes, key=lambda m: abs(m.freq - f_te101))
        return best.freq
    return float("nan")

conv = convergence_study(
    sim_factory=build_cavity,
    dx_values=[2.0e-3, 1.5e-3, 1.0e-3, 0.75e-3],
    metric_fn=extract_te101,
    until_decay=1e-3,
)
```

### 3.2 Interpreting the Output

```python
# Print the convergence table
conv.summary()
```

This prints a table like:

```
Mesh Convergence Summary
==================================================
    dx (mm)          Metric     Rel. Error
--------------------------------------------------
    2.0000    9.31500e+09       0.3521%
    1.5000    9.33200e+09       0.1697%
    1.0000    9.34100e+09       0.0733%
    0.7500    9.34500e+09       0.0304%
--------------------------------------------------
Extrapolated value : 9.3479e+09
Convergence order  : 2.04
```

Key information:

- **Extrapolated value:** the Richardson estimate of f at dx = 0.
- **Convergence order:** should be near 2 for Yee FDTD on axis-aligned
  structures. Values significantly below 2 indicate a problem (see
  Section 6).
- **Relative error:** how far each simulation is from the extrapolated
  value. Should decrease systematically as dx shrinks.

### 3.3 The Convergence Plot

```python
fig = conv.plot(title="Cavity TE101 Convergence")
fig.savefig("cavity_convergence.png", dpi=150)
```

The plot has two panels:

- **Left panel (Metric vs dx):** The simulated resonant frequency at
  each dx, with the extrapolated value shown as a dashed red line. The
  data points should approach the red line from one direction.

- **Right panel (Log-log error):** Relative error vs dx on a log-log
  scale. For second-order convergence, the data should follow a line
  with slope 2 (shown as a dashed reference).

**Reading the log-log plot.** This is the most informative view:

- **Slope ~ 2:** Standard Yee convergence. Your simulation is behaving
  as expected. Trust the Richardson extrapolation.
- **Slope < 2:** Something is limiting convergence (see Section 6 for
  common causes).
- **Slope > 2:** Unusual. May indicate super-convergence from
  subpixel smoothing or fortuitous error cancellation at specific grid
  sizes. Run more dx values to confirm.
- **Plateau at fine dx:** The error stops decreasing. You have hit a
  floor set by some other factor (PML reflections, simulation duration,
  machine precision).

---

## 4. Quick Convergence with `quick_convergence()`

If you already have a configured `Simulation` object and do not want to
write a factory function, use `quick_convergence()`. It clones your
simulation at scaled dx values automatically.

```python
from rfx import Simulation, Box, GaussianPulse, quick_convergence

sim = Simulation(
    freq_max=12e9,
    domain=(0.04, 0.02, 0.015),
    boundary="pec",
    dx=1e-3,
)
sim.add_source(
    (0.012, 0.008, 0.0045),
    component="ez",
    waveform=GaussianPulse(f0=6e9, bandwidth=0.8),
)
sim.add_probe((0.012, 0.008, 0.0045), "ez")

conv = quick_convergence(
    sim,
    metric="resonance",          # built-in: first resonance frequency
    dx_factors=[2.0, 1.5, 1.0, 0.75],
    until_decay=1e-3,
)

conv.summary()
fig = conv.plot()
fig.savefig("quick_convergence.png", dpi=150)
```

**Built-in metrics:**

- `"resonance"` -- first resonance frequency from `find_resonances()`.
- `"peak_field"` -- peak absolute probe amplitude.
- Any callable `metric(result) -> float` for custom quantities.

---

## 5. Convergence Study for a Patch Antenna

Antenna simulations are more challenging than cavities because the open
boundary (CPML) introduces additional error. Here is the complete
convergence workflow for the patch antenna from the
[Patch Antenna tutorial](tutorial-patch-antenna.md).

```python
import numpy as np
from rfx import Simulation, Box, GaussianPulse, convergence_study

C0 = 2.998e8
f0 = 2.4e9
eps_r = 4.4
tan_d = 0.02
h = 1.6e-3
margin = 15e-3

# Hammerstad dimensions
W = C0 / (2 * f0) * np.sqrt(2 / (eps_r + 1))
eps_eff = (eps_r + 1) / 2 + (eps_r - 1) / 2 * (1 + 12 * h / W) ** (-0.5)
dL = 0.412 * h * ((eps_eff + 0.3) * (W / h + 0.264)
                   / ((eps_eff - 0.258) * (W / h + 0.8)))
L = C0 / (2 * f0 * np.sqrt(eps_eff)) - 2 * dL

sigma_sub = 2 * np.pi * f0 * 8.854e-12 * eps_r * tan_d
dom_x = L + 2 * margin
dom_y = W + 2 * margin

def build_patch(dx):
    n_sub  = max(4, int(np.ceil(h / dx)))
    dz_sub = h / n_sub
    n_air  = max(6, int(np.ceil(margin / dx)))
    dz_prof = np.concatenate([
        np.full(n_sub, dz_sub),
        np.full(n_air, dx),
    ])

    sim = Simulation(
        freq_max=f0 * 2,
        domain=(dom_x, dom_y, 0),
        dx=dx,
        dz_profile=dz_prof,
        cpml_layers=12,
    )
    sim.add_material("sub", eps_r=eps_r, sigma=sigma_sub)
    sim.add(Box((0, 0, 0), (dom_x, dom_y, 0)), material="pec")
    sim.add(Box((0, 0, 0), (dom_x, dom_y, h)), material="sub")
    sim.add(Box((margin, margin, h),
                (margin + L, margin + W, h)), material="pec")
    sim.add_source(
        (margin + L / 3, margin + W / 2, h / 2),
        "ez",
        waveform=GaussianPulse(f0=f0, bandwidth=0.8),
    )
    sim.add_probe((margin + L / 3, margin + W / 2, h / 2), "ez")
    return sim

def get_resonance(result):
    modes = result.find_resonances(freq_range=(f0 * 0.5, f0 * 1.5))
    if modes:
        return min(modes, key=lambda m: abs(m.freq - f0)).freq
    # FFT fallback
    ts = np.asarray(result.time_series).ravel()
    spec = np.abs(np.fft.rfft(ts, n=len(ts) * 8))
    freqs = np.fft.rfftfreq(len(ts) * 8, d=result.dt)
    band = (freqs > f0 * 0.5) & (freqs < f0 * 1.5)
    return float(freqs[np.argmax(spec * band)])

conv = convergence_study(
    sim_factory=build_patch,
    dx_values=[1.0e-3, 0.75e-3, 0.5e-3],
    metric_fn=get_resonance,
    until_decay=1e-3,
)

conv.summary()
fig = conv.plot(title="Patch Antenna Resonance Convergence")
fig.savefig("patch_convergence.png", dpi=150)
```

**Interpreting the result.** For the patch antenna, expect:

- Convergence order near 1.5--2.0 (slightly below 2 because the PEC
  patch boundary is staircased, introducing first-order error).
- The extrapolated frequency should be within 1--2% of the Hammerstad
  analytical prediction.

---

## 6. Common Pitfalls

### 6.1 PML Too Thin

**Symptom.** The metric oscillates or does not converge monotonically.

**Cause.** The CPML absorbing boundary reflects energy back into the
domain. With too few PML layers (e.g., 4--6), the reflected wave
contaminates the probe signal.

**Fix.** Increase `cpml_layers`. The default (10--12) is adequate for
most cases. For very low-frequency problems or high dynamic range
requirements, use 15--20 layers.

```python
sim = Simulation(..., cpml_layers=15)
```

### 6.2 Domain Too Small

**Symptom.** The resonant frequency shifts significantly when you
increase the domain size, even without changing dx.

**Cause.** The structure is too close to the domain boundary, and
near-field coupling to the CPML distorts the result.

**Fix.** Increase the margin around the structure. A good minimum is
lambda/4 at the lowest frequency of interest.

**Diagnostic.** Run a "domain convergence" study alongside the mesh
convergence: keep dx fixed and increase the margin. The metric should
stabilise.

### 6.3 Not Enough Timesteps

**Symptom.** Harminv fails to find modes, or the FFT spectrum shows
broad peaks without clear resonances.

**Cause.** The simulation stopped before the transient fully decayed.
The ring-down was truncated, robbing Harminv of the data it needs.

**Fix.** Use `until_decay=1e-3` (or `1e-4` for high-Q structures)
instead of a fixed `n_steps`. If you must use fixed steps, ensure
T_sim >> Q / (pi * f0) where Q is the expected quality factor.

```python
result = sim.run(until_decay=1e-4, decay_max_steps=50000)
```

### 6.4 Source Inside a PEC Region

**Symptom.** Zero or near-zero probe signal; Harminv finds no modes.

**Cause.** The source position snapped onto a PEC cell, which forces
the field to zero. The source energy is immediately zeroed out.

**Fix.** Place the source inside the dielectric (substrate) or in the
air region, not at a PEC boundary. For microstrip structures, h/2
(mid-substrate) is a safe choice.

### 6.5 Comparing the Wrong Metric

**Symptom.** Convergence looks perfect for one metric but poor for
another.

**Cause.** Different quantities converge at different rates. For
example, resonant frequency converges at order 2, but near-field
amplitude at a specific point may converge at order 1 (due to
stairstepping).

**Fix.** Always run the convergence study on the actual metric you
care about. If your design target is S11 < -10 dB, converge on S11
at f0, not on the resonant frequency.

---

## 7. When to Trust Your Results

Use this checklist before reporting a simulation result:

- [ ] **Convergence study done.** At least 3 dx values, with monotonic
  convergence toward the extrapolated value.
- [ ] **Order is reasonable.** Observed convergence order between 1.5
  and 2.5 for standard Yee problems.
- [ ] **Error is acceptable.** Relative error at your working dx is
  below your accuracy requirement (1% for resonance, 0.5 dB for
  S-parameters, etc.).
- [ ] **PML is adequate.** At least 10 CPML layers; margins >= lambda/8.
- [ ] **Simulation ran long enough.** Used `until_decay` or confirmed
  that probe signal has decayed to at least 1% of peak.
- [ ] **No NaN or divergence.** Checked `np.any(np.isnan(result.time_series))`.
- [ ] **Result is physically plausible.** Resonant frequency within
  ~20% of analytical prediction; S11 < 0 dB (passive device).

---

## 8. Advanced: Subpixel Smoothing and Convergence Order

For structures with curved dielectric interfaces, enabling subpixel
smoothing lifts the convergence order from ~1 (stairstepping) to ~2:

```python
conv_no_smooth = convergence_study(
    sim_factory=build_sim,
    dx_values=[2e-3, 1.5e-3, 1e-3, 0.75e-3],
    metric_fn=metric,
    run_kwargs={"subpixel_smoothing": False},
)

conv_smooth = convergence_study(
    sim_factory=build_sim,
    dx_values=[2e-3, 1.5e-3, 1e-3, 0.75e-3],
    metric_fn=metric,
    run_kwargs={"subpixel_smoothing": True},
)

print(f"Without smoothing: order = {conv_no_smooth.order:.2f}")
print(f"With smoothing:    order = {conv_smooth.order:.2f}")
```

**Important:** Subpixel smoothing only helps dielectric interfaces.
PEC stairstepping remains first-order regardless. For curved PEC
structures, the only remedy is finer meshing.

---

## 9. Summary

| Step | What to do | rfx API |
|---|---|---|
| Define factory | Build sim at variable dx | `sim_factory(dx) -> Simulation` |
| Choose metric | Extract the quantity you care about | `metric_fn(result) -> float` |
| Run study | Test 3--4 dx values | `convergence_study(...)` |
| Check order | Should be ~2 for axis-aligned Yee | `conv.order` |
| Check error | Must be below your tolerance | `conv.errors` |
| Visualise | Log-log error plot confirms rate | `conv.plot()` |
| Extrapolate | Estimate the true answer | `conv.extrapolated` |

**Golden rule:** If you have not done a convergence study, you do not
know whether your result is right.

---

## References

- L. F. Richardson, "The approximate arithmetical solution by finite
  differences of physical problems involving differential equations,"
  *Phil. Trans. Royal Soc. A*, 210, 1911, pp. 307--357.
- A. Taflove and S. C. Hagness, *Computational Electrodynamics: The
  Finite-Difference Time-Domain Method*, 3rd ed., Artech House, 2005,
  Ch. 4 (numerical dispersion and stability).
- K. S. Yee, "Numerical solution of initial boundary value problems
  involving Maxwell's equations in isotropic media," *IEEE Trans.
  Antennas Propag.*, 14(3), 1966, pp. 302--307.
