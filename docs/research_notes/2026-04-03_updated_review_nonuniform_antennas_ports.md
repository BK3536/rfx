# 2026-04-03 Updated Read-Only Review: Non-Uniform Mesh, Patch Antennas, Feed/Port Modeling

## Scope

This note updates the earlier 2026-04-03 subgridding/adaptive-meshing review
after additional repository changes. The focus here is:

1. recent `rfx` updates around non-uniform meshing,
2. correctness of antenna examples,
3. especially feed / port setup,
4. cross-checking against proven `Meep` and `openEMS` workflows.

This is a **read-only review**. No simulator code was modified as part of the
analysis.

## Executive Summary

The strategic picture has shifted:

- `rfx` now looks strongest for **thin-substrate antennas** when using
  **non-uniform z-meshing** plus **current-source excitation**.
- For this problem class, that is a better practical direction than the
  current SBP-SAT antenna path.
- However, the repository's **patch-port / S11 examples are not yet reliable as
  validated feed examples**.

### Current recommendation ranking for `rfx`

1. **Non-uniform z mesh for layered antennas** — best current path
2. **Static SBP-SAT block refinement** — still useful, but less mature for
   current antenna validation
3. **Dynamic AMR / runtime adaptive meshing** — still low priority

## Repository State Reviewed

### Recent changes observed

Recent commits indicate strong momentum in the non-uniform direction:

- `100b4f9` — non-uniform Yee update functions
- `4be2932` — non-uniform Yee mesh runner + patch antenna
- `9303f2f` — non-uniform runner with CPML
- `f00d878` — claimed `<1%` patch result via non-uniform Yee + CPML
- `de65a74` / `b0e4f4a` / `755f586` — generalized validation and port work

### Files reviewed

Core:

- `rfx/api.py`
- `rfx/nonuniform.py`
- `rfx/auto_config.py`
- `rfx/sources/sources.py`
- `rfx/geometry/csg.py`

Examples:

- `examples/04_patch_antenna.py`
- `examples/05_patch_antenna_subgrid.py`
- `examples/06_patch_uniform_ref.py`
- `examples/09_patch_fixed.py`
- `examples/10_patch_point_source.py`
- `examples/13_patch_profiled.py`
- `examples/24_patch_sub1pct.py`
- `examples/28_generalized_validation.py`
- `examples/29_nu_port_sparam.py`
- `examples/30_current_source_generalized.py`
- `examples/31_nu_api_validation.py`

Docs / notes:

- `docs/research_notes/2026-04-03_nonuniform_mesh_plan.md`
- `docs/guides/simulation_methodology.md`
- prior subgridding notes

Tests / verification:

- `tests/test_nonuniform_api.py`
- `tests/test_wire_sparam.py`

## Verification Performed

### Local test run

Executed:

```bash
pytest -q tests/test_nonuniform_api.py tests/test_wire_sparam.py
```

Result:

- **16 passed**

Notes:

- This is useful evidence that the new non-uniform API path is wired up and
  does not immediately regress core smoke paths.
- It does **not** yet prove physical correctness of patch-port extraction.

### Static inspection findings

- `examples/04_patch_antenna.py` appears internally inconsistent: it later
  references `s11_approx` / `s11_min` in a way that does not match the visible
  code path.
- The non-uniform `Simulation(dz_profile=...)` path is active in `api.py`.
- The new constructor behavior for `Simulation(..., dz_profile=..., domain=(..., ..., 0))`
  is now supported by runtime code.

## Main Updated Assessment

## 1. Non-uniform z mesh is now the most promising antenna path

For microstrip / patch structures, this is a good architectural move.

Why:

- thin substrate thickness is the dominant meshing pain point,
- lateral dimensions often do **not** require the same refinement as the
  substrate-normal direction,
- this matches how practical EM tools often handle layered structures,
- it is much more JAX-friendly than dynamic AMR.

The current `rfx/nonuniform.py` design:

- uses uniform `dx`, `dy`,
- graded `dz`,
- preserves fixed shapes,
- keeps the time loop in `jax.lax.scan`.

That is a very good fit for JAX/XLA.

## 2. The best current patch examples are the current-source / resonance ones

Examples that are directionally the most credible right now:

- `examples/24_patch_sub1pct.py`
- `examples/28_generalized_validation.py`
- `examples/30_current_source_generalized.py`
- `examples/31_nu_api_validation.py`
- and partially `examples/13_patch_profiled.py`

These are strongest when interpreted as:

- resonance validation,
- field validation,
- decay / Harminv validation,
- radiation-style source studies.

They are **not** yet proof of calibrated feed-port behavior.

## 3. The patch-port / S11 examples are still not trustworthy as validated references

This is the most important conclusion from the review.

Affected examples include:

- `examples/04_patch_antenna.py`
- `examples/05_patch_antenna_subgrid.py`
- `examples/06_patch_uniform_ref.py`
- `examples/09_patch_fixed.py`
- `examples/29_nu_port_sparam.py`

### 3.1 Wire port geometry does not actually span conductor-to-conductor in several examples

Examples `04`, `05`, `06`, and `09` use a pattern like:

```python
position=(feed_x, feed_y, dx)
extent=h - 2*dx
```

For representative patch settings:

- ground PEC is at `z = 0`
- patch PEC is at `z = h`
- port starts at `z = dx`
- port ends at `z = h - dx`

So the wire excludes the terminal conductor sheets.

For example 04 with `dx = 0.5 mm`, `h = 1.6 mm`:

- ground index ≈ 0
- wire start ≈ index 1
- wire end ≈ index 2
- patch ≈ index 3

This means the wire is **not actually modeled from ground to patch**.

That is not equivalent to the proven openEMS patch tutorial, which uses a
lumped port extending from:

- `z = 0`
- to `z = substrate_thickness`

with the port physically connecting the feed terminals.

### 3.2 FFT(probe) / FFT(source) is not a validated S11 method

Several patch examples compute something like:

- FFT of a local field probe,
- divided by FFT of the source waveform,
- then plotted and interpreted as S11 / resonance dip.

This is a useful **diagnostic spectral response**, but it is not the same as:

- openEMS `CalcPort(...)` incident/reflected decomposition,
- Meep flux-normalized reflectance,
- or a rigorously defined feed-port scattering parameter.

Conclusion:

> These examples should not be treated as validated `S11` examples.

They are acceptable as:

- resonance heuristics,
- spectral diagnostics,
- exploratory debugging plots.

### 3.3 Non-uniform wire-port extraction still looks physically incomplete

In the non-uniform path:

- per-cell port conductivity still uses `grid.dx`,
- midpoint voltage extraction uses `E * grid.dx`.

For a z-directed vertical probe in a non-uniform z-mesh, this is not obviously
the correct physical line integral / terminal quantity.

This matters because openEMS-style port extraction depends on:

- properly defined port terminals,
- incident/reflected wave decomposition,
- integrated V/I over the port cross-section / terminal region.

The current implementation may be adequate as an internal proxy, but it is not
yet convincing as an openEMS-grade port model for patch-feed validation.

## 4. Feed position itself looks fine; the feed model is the real issue

The lateral feed inset used in `rfx` is not the problem.

Typical `rfx` patch examples use:

- feed at approximately `L/3` from one patch edge.

The official openEMS simple patch tutorial places the feed at about:

- `31.25%` from one edge of the patch width axis.

That is close to `1/3 = 33.3%`.

So:

- **feed location is plausible**
- **feed terminal model / extraction method is the problem**

## 5. Board/domain modeling is mixed in quality

Some older examples use:

- ground or substrate spanning the entire computational footprint,
- broad dielectric under most of the domain,
- less clear separation between actual board and surrounding air region.

This is weaker than the proven openEMS patch style, where:

- substrate has finite board dimensions,
- surrounding domain is larger air space,
- the board is a distinct object inside the simulation box.

The newer non-uniform examples are moving in a better direction.

## Cross-Check Against Proven Reference Workflows

## openEMS

The official openEMS simple patch tutorial uses:

- explicit finite substrate geometry,
- explicit substrate-thickness meshing,
- a **lumped port spanning from ground to patch**,
- `CalcPort(...)` postprocessing,
- reflected / incident decomposition:
  - `s11 = port.uf_ref / port.uf_inc`
- NF2FF evaluated at the extracted resonance.

This is significantly more rigorous than the current `rfx` FFT-ratio patch
examples.

It strongly suggests that:

1. `rfx` should not label FFT-ratio diagnostics as S11,
2. a proper patch-feed validation example should mirror the openEMS workflow
   more closely.

## Meep

For antenna / radiation examples, Meep commonly uses:

- electric-current sources,
- PML-bounded air regions,
- pulsed excitation,
- DFT accumulation / near-to-far transforms,
- field-decay-based stopping.

This aligns well with the newer `rfx` current-source antenna examples.

For scattering / reflectance workflows, Meep relies on:

- flux monitors,
- normalization runs,
- and, for guided systems, mode/flux decomposition.

This again argues that:

- current-source resonance examples in `rfx` are reasonable,
- but port/S11 patch examples still need a better measurement model.

## Recommended Interpretation of Current Example Families

### A. Good for resonance / radiation validation

Use these as the main “physics looks right” patch examples:

- `24_patch_sub1pct.py`
- `28_generalized_validation.py`
- `30_current_source_generalized.py`
- `31_nu_api_validation.py`
- `13_patch_profiled.py`

Interpretation:

- resonant frequency accuracy,
- field decay,
- Harminv mode extraction,
- current-source behavior,
- graded-z meshing benefits.

### B. Not yet reliable as calibrated S11 examples

Treat these as provisional / diagnostic only:

- `04_patch_antenna.py`
- `05_patch_antenna_subgrid.py`
- `06_patch_uniform_ref.py`
- `09_patch_fixed.py`
- `29_nu_port_sparam.py`

Interpretation:

- useful debugging scripts,
- not final validation references.

## Recommended Next Steps

## Priority 1: Separate “resonance/radiation” from “port/S11” examples

Documentation and example naming should clearly distinguish:

1. **Current-source resonance/radiation examples**
2. **True feed-port S11 examples**

This will prevent users from assuming the current FFT-ratio scripts are
validated port measurements.

## Priority 2: Rework the patch-port model to match openEMS more closely

Needed characteristics:

- port physically spans the actual feed terminals,
- integrated terminal voltage/current,
- incident/reflected decomposition,
- clear measurement plane definition,
- optional de-embedding.

The current midpoint/proxy approach is not yet enough for a reference patch
feed example.

## Priority 3: Keep investing in non-uniform z mesh

This is the most practically useful antenna meshing path currently visible in
the repo.

It is especially good for:

- thin substrates,
- stacked dielectric layers,
- microstrip / patch / stripline-like structures.

## Priority 4: Only use SBP-SAT subgridding later where geometry really needs local 3D refinement

For layered antennas, graded-z non-uniform meshing may solve the highest-value
problem with much lower complexity than full subgridding.

## Bottom-Line Conclusion

The best current message for `rfx` is:

> **For microstrip / patch antenna validation, non-uniform z meshing plus
> current-source excitation is now the strongest path.**

But also:

> **Patch-port / S11 examples are not yet physically mature enough to serve as
> validated reference antenna-feed examples.**

So the repository is in a good place directionally, but the antenna example
set should be split into:

- **trusted resonance/radiation examples**
- **provisional port/S11 examples**

until the port model is brought closer to openEMS-grade practice.

## External References

- openEMS simple patch antenna tutorial:
  https://docs.openems.de/python/openEMS/Tutorials/Simple_Patch_Antenna.html
- openEMS patch tutorial (with `CalcPort`):
  https://wiki.openems.de/index.php/Tutorial%3A_Simple_Patch_Antenna.html
- openEMS microstrip port example:
  https://docs.openems.de/python/openEMS/Tutorials/MSL_NotchFilter.html
- Meep basics:
  https://meep.readthedocs.io/en/latest/Python_Tutorials/Basics/
- Meep near-to-far antenna tutorial:
  https://meep.readthedocs.io/en/latest/Python_Tutorials/Near_to_Far_Field_Spectra/
- Meep eigenmode / flux normalization note:
  https://meep.readthedocs.io/en/master/Scheme_Tutorials/Eigenmode_Source/

## Local Evidence Notes

- `pytest -q tests/test_nonuniform_api.py tests/test_wire_sparam.py` passed
  (`16 passed`)
- `examples/04_patch_antenna.py` appears internally inconsistent by inspection
- current wire-port examples often stop short of both conductor terminal sheets
- non-uniform current-source examples are more aligned with Meep-style antenna
  validation
