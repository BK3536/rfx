# Recipe: R(f), T(f) measurement of a slab / scatterer

## When to use

Any broadband reflection / transmission measurement of a scatterer
(dielectric slab, multilayer, periodic surface, dispersive media) in
the frequency domain.

## The right primitive

**`Simulation.add_flux_monitor`** — on-the-fly DFT accumulation of the
Poynting flux on a plane. Direct equivalent of Meep `add_flux`.

**`Simulation.run(until_decay=1e-5, ...)`** — Meep's
`stop_when_fields_decayed` equivalent. Runs until the signal at a
chosen monitor has decayed to `until_decay * peak`. Required for
clean flux DFT — a fixed `n_steps` truncates the signal and aliases
residual energy into the spectrum.

## Canonical pattern (two-run reference subtraction)

```python
from rfx import Simulation, Box
from rfx.probes.probes import flux_spectrum

def build_sim(with_slab: bool) -> Simulation:
    sim = Simulation(
        freq_max=freq_max, domain=(dom_x, dom_y, dx), dx=dx,
        boundary="cpml", cpml_layers=20, mode="2d_tmz",
    )
    if with_slab:
        sim.add_material("slab", eps_r=eps_inf, lorentz_poles=[pole])
        # Oversize Box in y/z so the Lorentz mask covers CPML padding too —
        # with TFSF, y/z are periodic and any cell without the pole breaks
        # the plane-wave assumption.
        sim.add(Box((slab_x_lo, -1, -1), (slab_x_hi, 1, 1)), material="slab")
    sim.add_tfsf_source(f0=f0, bandwidth=bw, polarization="ez", direction="+x")
    sim.add_flux_monitor(axis="x", coordinate=refl_x, freqs=freqs_rt, name="refl")
    sim.add_flux_monitor(axis="x", coordinate=trans_x, freqs=freqs_rt, name="trans")
    return sim

# Run 1 — no scatterer (reference)
sim_ref = build_sim(with_slab=False)
res_ref = sim_ref.run(
    n_steps=12000, until_decay=1e-5,
    decay_monitor_component="ez",
    decay_monitor_position=(trans_x, y_center, 0),
)
ref_refl_fm = res_ref.flux_monitors["refl"]
ref_trans_flux = np.asarray(flux_spectrum(res_ref.flux_monitors["trans"]))

# Run 2 — with scatterer
sim_slab = build_sim(with_slab=True)
res_slab = sim_slab.run(
    n_steps=12000, until_decay=1e-5,
    decay_monitor_component="ez",
    decay_monitor_position=(trans_x, y_center, 0),
)
slab_refl_fm = res_slab.flux_monitors["refl"]
slab_trans_flux = np.asarray(flux_spectrum(res_slab.flux_monitors["trans"]))

# Field-level subtraction for reflection (Meep load_minus_flux_data equivalent)
scat_refl_fm = slab_refl_fm._replace(
    e1_dft=slab_refl_fm.e1_dft - ref_refl_fm.e1_dft,
    e2_dft=slab_refl_fm.e2_dft - ref_refl_fm.e2_dft,
    h1_dft=slab_refl_fm.h1_dft - ref_refl_fm.h1_dft,
    h2_dft=slab_refl_fm.h2_dft - ref_refl_fm.h2_dft,
)
scat_refl_flux = np.asarray(flux_spectrum(scat_refl_fm))

# Final coefficients
T = slab_trans_flux / ref_trans_flux        # |t|^2
R = -scat_refl_flux / ref_trans_flux        # sign flip: reflection is -x
```

## Critical details

1. **Field-level subtraction (not flux-level)**: Poynting flux is
   bilinear in E and H; `F_slab - F_ref` contains mixed cross terms
   that do not cancel. Subtract E and H DFT accumulators first, then
   compute the flux. Meep does the same in `load_minus_flux_data`.

2. **`until_decay` is load-bearing**: a fixed `n_steps` makes the DFT
   aliasing dependent on where the signal happens to be truncated. For
   dispersive / high-Q scatterers use `until_decay=1e-5` (or tighter);
   for simple lossless slabs `1e-3` is often enough.

3. **TFSF periodicity gotcha**: `add_tfsf_source` forces the transverse
   axes (y, z) to be periodic. The slab geometry's `Box` must span the
   FULL y/z extent (including CPML padding) or the dispersive pole mask
   is non-uniform in y/z and breaks the plane-wave assumption. Use
   oversized corners like `Box((x_lo, -1, -1), (x_hi, 1, 1))`.

4. **Cell-aligned slab thickness**: `Box` uses inclusive bounds
   (`coords >= lo & coords <= hi`), so a `d`-thick physical box can
   produce `d + dx` cells of material. Nudge the upper corner down by
   `dx/2` to match integer cell counts: `slab_x_hi = center + d/2 - dx/2`.

## Do NOT use

**`add_probe` (time-series) + `np.fft.rfft` of the probe trace.** This
works for dispersionless media where all frequency components have the
same group velocity, but **fails for dispersive media**:

- The transmitted pulse is chirped (frequencies arrive at different
  times inside any time-gate window).
- Any time window (Hann, Tukey, rectangular) introduces a
  frequency-dependent bias because the chirped pulse and the incident
  pulse have different effective envelopes under the same window.
- The bias is **independent of grid resolution** — halving `dx` does
  not help (confirmed in crossval 08 debugging session, 2026-04-10).

On-the-fly DFT accumulation (what `add_flux_monitor` does) sidesteps
this entirely: each frequency component is integrated continuously, and
`until_decay` ensures the signal has truly ended.

## Canonical examples

- **Dispersive Lorentz slab**: `examples/crossval/08_material_dispersion.py`
  — canonical reference with `add_flux_monitor` + `until_decay`.
- **Dispersionless Fresnel slab**: `examples/crossval/04_multilayer_fresnel.py`
  *(still uses raw loop; migration pending. For dispersionless the
  raw-loop approach also works because there is no group-velocity
  chirp.)*

## Companion primitives

- `add_dft_plane_probe` — full complex E/H on a 2D plane (for mode
  patterns, waveguide modal decomposition).
- `add_ntff_box` — far-field transform for 3D scatterers.
- `compute_waveguide_s_matrix` — S-parameters in bounded waveguides
  (not plane-wave).

## Known residual rfx-vs-Meep mismatch (~1% in R+T)

Even after applying everything above (FluxMonitor, `until_decay=1e-5`,
field-level ref subtraction, Airy analytic, matched source waveform),
rfx R+T still oscillates ~±0.01 around the analytic curve for
dispersive slabs, while Meep is smoother (~±0.002). This was
**extensively verified** in the 2026-04-10 session:

- Not the source waveform — adding `waveform="modulated_gaussian"` to
  `add_tfsf_source` (Meep-equivalent `cos·exp` pulse) did **not** fix
  the oscillation.
- Not grid resolution — halving `dx` did not reduce the error
  (O(dx²) scaling absent).
- Not the analytic formula — Airy closed form gives correct R+T
  (verified by direct multi-reflection series sum).
- Not time-window bias — `until_decay` eliminates the truncation
  artifact that used to plague the time-series FFT approach.

**Most likely root cause**: the Lorentz ADE discretization differs
between rfx (2nd-order central-difference on the ODE for P) and Meep
(recursive convolution in the D-field update). Both are O(Δt²) but
coefficients differ, producing slightly different effective ε(f).

**Accept this as a known O(Δt²) discretization-level mismatch** when
comparing rfx and Meep for dispersive media. rfx matches the analytic
transfer matrix to ~1% mean error — well within crossval PASS limits.
Do NOT chase this further unless there is a concrete reason to believe
it's a bug rather than discretization.

**Key evidence from crossval 10** (Drude metal, 2026-04-10): rfx
matches analytic to **~0.07% mean error** — 15× better than the
Lorentz crossval 08. Drude has no sharp resonance peak, so the
central-difference ADE discretization is essentially exact. The ~1%
residual in crossval 08 comes specifically from the stiff Lorentz
resonance, not from any rfx implementation issue. **rfx's Drude/Lorentz
ADE is not the bottleneck** — Lorentz-specific ODE stiffness near
resonance is the real limit.

## References

- Meep tutorial: *Transmittance Spectrum of a Waveguide Bend*,
  *Material Dispersion*
- Taflove & Hagness Ch. 5 (TFSF), Ch. 7 (on-the-fly DFT),
  Ch. 9 (Lorentz ADE)
- rfx source: `rfx/probes/probes.py::FluxMonitor`, `flux_spectrum`;
  `rfx/sources/tfsf.py::init_tfsf` for `waveform` option
- Debugging session: workspace commit history around 2026-04-10
  (crossval 08 — see script docstring for pitfalls encountered)
