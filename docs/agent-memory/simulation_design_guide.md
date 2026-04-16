# rfx Simulation Design Guide (Agent-Facing)

Pre-design decision tree for AI agents configuring rfx simulations.
Read BEFORE writing `Simulation(...)`. Prevents the classes of errors
that code-level `sim.preflight()` cannot catch (physics modeling
decisions vs geometric placement mistakes).

---

## 1. Mesh Strategy Selection

- **Uniform**: default. Use when geometry has no thin layers requiring
  sub-cell resolution (e.g., free-space scattering, waveguide modes).
- **NU (dz_profile / dx_profile)**: use when a thin substrate, coating,
  or layer needs finer resolution than the surrounding air/vacuum.
  Always set `dz_sub` proportional to `dx` (see §3).

## 2. Cell Size Derivation

```
dx = min(lambda_min / 20,  smallest_feature / 4)
```

- `lambda_min = c / freq_max` in vacuum; for dielectric `lambda_eff = lambda_min / sqrt(eps_r)`.
- Substrate thickness `h_sub` should be resolved by ≥ 6 cells: `dz_sub ≤ h_sub / 6`.
- **Co-refine rule**: `dz_sub = dx / K` where K is the baseline aspect ratio (typically 4). Never change `dx` without scaling `dz_sub`.

## 3. Convergence Protocol

**Correct**: co-refine dx AND dz at fixed aspect ratio.

| Level | dx | dz_sub (K=4) | Expected |
| ----- | -- | ------------ | -------- |
| L1 | 1.0 mm | 0.250 mm | baseline |
| L2 | 0.5 mm | 0.125 mm | better |
| L3 | 0.25 mm | 0.0625 mm | converged |

**Wrong**: fix dz, refine only dx → CFL dt stays pinned to dz → xy
Courant number `nu_xy = c·dt/dx` increases → anti-convergence
(Taflove Ch. 4, confirmed by rfx VESSL evidence 2026-04-16).

Evidence (patch antenna f_res vs Balanis):
- Co-refine: 9.31% → 5.76% (converges)
- Fixed dz: 9.31% → 12.75% (diverges)

## 4. Material Modeling Checklist

| Condition | Action |
| --------- | ------ |
| Open structure (CPML boundaries) + dielectric | **Must include loss tangent** (tan δ). Lossless dielectric in CPML domain creates an artificial cavity with Q >> 1000 — energy is trapped, never radiates. |
| Closed PEC cavity (no CPML) | Lossless OK — cavity Q is physical. |
| Substrate FR4 at ≤ 10 GHz | tan δ ≈ 0.02. `sigma = 2π·f·ε₀·εᵣ·tan_δ`. |
| Metals (copper, aluminium) | Use `material="pec"` for FDTD. Skin depth (~μm) is sub-cell; PEC is the canonical FDTD model. |

**FR4 cavity trap** (issue #48 session evidence): lossless FR4 +
CPML → harminv reports Q ≈ 972 (cavity mode, not radiation). Lossy
FR4 → Q ≈ 28 (realistic patch antenna). FFRP goes from θ=88°
(grazing) to θ=1° (broadside).

## 5. Domain and CPML Sizing

- Air margin from geometry to CPML: ≥ `lambda_max / 4`.
- `cpml_layers ≥ 8` (standard). Increase to 12-16 for grazing incidence.
- Below a finite ground plane: leave ≥ `lambda_max / 4` air so
  NTFF captures backscatter.

## 6. Source and Probe Placement

- **Never place inside a PEC cell.** Preflight catches this
  (`_validate_simulation_config`), but the zero-field symptom
  (uniform dx=0.5mm, VESSL #588 case C/D) is silent otherwise.
- Source at ≥ 2 cells from PEC surface and ≥ cpml_layers + 2 from
  domain edge.
- For patch antenna: place inside substrate, 2-3 dz_sub above ground.
- Probe: offset from source (avoid self-coupling at early timesteps).

## 7. NTFF Box Rules

- Must enclose all radiators.
- ≥ cpml_layers + 2 cells from domain boundary on each face.
- ≥ λ/4 air gap from geometry surfaces.
- **Never overlap CPML** — preflight warns, but NaN/garbage pattern results.
- Metal planes must sit on cell edges with symmetric neighbouring
  cells inside the NTFF box (Meep/OpenEMS convention, issue #48).

## 8. Simulation Time and Termination

- `num_periods = 60` is a good default for broadband harminv extraction.
- For lossy structures (Q < 50), `num_periods = 30` may suffice.
- Time for ringdown: `T_ring ≈ Q / (π · f_res)`. If Q = 30,
  f = 2.4 GHz → T_ring ≈ 4 ns ≈ 8000 steps at dt = 0.5 ps.

## 9. Anti-Patterns (Do Not Repeat)

1. **Lossless FR4 + CPML** → artificial Q >> 1000, FFRP at grazing.
   Fix: add `sigma = 2π·f·ε₀·εᵣ·tan_δ`.
2. **Refine dx but hold dz fixed** → anti-convergent f_res.
   Fix: co-refine `dz_sub = dx / K`.
3. **Source in PEC cell** → zero field, harminv finds nothing.
   Fix: verify source z-index ≠ PEC z-index in preflight or diagnostic.
4. **NTFF box in CPML** → corrupted radiation pattern.
   Fix: keep NTFF ≥ 3 cells inside interior boundary.
5. **`smooth_grading` without `preserve_regions`** → substrate z-coords
   shift, metal no longer on cell edge.
   Fix: use explicit mesh-aligned dz or `preserve_regions=[(lo, hi)]`.
6. **`Box._axis_mask` with thin PEC on graded axis** → PEC snaps onto
   multiple cells (fixed in v1.6.0 via per-cell `dc`).
