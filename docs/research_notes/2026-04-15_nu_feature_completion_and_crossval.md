# Non-uniform runner feature completion, crossval, and inverse-design unblock

**Date**: 2026-04-15
**Branch**: `nonuniform-completion`
**Context**: continuation of the NU-completion work from the morning session;
wired the missing features into the NU runner, validated against OpenEMS on
GPU, and cleared four GitHub feature-request / bug issues that had been
blocking gradient-based inverse design on NU grids.

---

## Scope

1. Wire all remaining simulation primitives into `rfx/runners/nonuniform.py`
   (Steps 4a–4e of the NU roadmap).
2. Run an end-to-end antenna demo on the NU runner and confirm physical
   accuracy against the Balanis TL analytic target and against OpenEMS.
3. Quantify the real memory saving of NU vs uniform at matched physical
   accuracy.
4. Address open GitHub issues that blocked NU / inverse-design workflows:
   #29, #30, #32, #33.

---

## NU runner feature parity (Step 4 series)

Prior commits (already on branch at session start):

| Step | Commit | Feature |
|------|--------|---------|
| 4a   | `65a73b8` | DFT plane probe on NU runner |
| 4b   | `9deaa1b` | Lumped RLC element support on NU runner |
| 4c   | `4aa5ff4` | NTFF box support on NU runner |
| 4d   | `a475354` | Waveguide port support on NU runner |
| 4e   | `d42b659` | TFSF +x/-x plane-wave source on NU runner |

The NU runner now exposes the same primitive set as the uniform runner.
UPML on NU is unblocked at `rfx/boundaries/upml.py` via per-axis inverse
spacing (`inv_dx`, `inv_dy`, `inv_dz`).

Still missing on NU forward: Floquet ports, `pec_occupancy_override`,
`checkpoint=True` plumbed into the NU scan body. Tracked as known gaps
in the commit message of the NU-forward patch.

---

## 2.4 GHz FR4 patch antenna demo (examples/nonuniform_patch_demo.py)

Standalone 198-line demo, substrate 0.25 mm / air 1 mm NU z mesh
(`smooth_grading`, max_ratio=1.3), x/y uniform 1 mm,
probe-fed Gaussian source, `boundary="cpml"`, 8 CPML layers,
finite 60×55 mm ground plane.

**Local CPU result** (1.7 s wall, 324k NU cells):
- rfx Harminv f = 2.4621 GHz
- Balanis TL analytic f = 2.4235 GHz
- Error: **1.59 %** (pass criterion < 8 %: PASS)

Plots saved to `examples/nonuniform_patch_demo/`:
- `probe_ez_timeseries.png`
- `dz_mesh_profile.png`

---

## GPU + OpenEMS crossval (VESSL `369367233458`)

VESSL job on `remilab-c0 / gpu-rtx4090` using the microwave-energy
OpenEMS install recipe pattern ported to the `nvcr.io/nvidia/jax:24.10-py3`
image (py3.10, Ubuntu 22.04):

```yaml
apt-get install -y --no-install-recommends \
  openems python3-openems libopenems0 libcsxcad0 python3-matplotlib git
/usr/bin/python3 -m pip install -e ".[dev]" "numpy<2"
```

The image's bundled JAX (0.4.33 dev) saw the GPU (`[CudaDevice(id=0)]`),
openEMS + CSXCAD imports were verified before the runs.

**Final crossval output (examples/crossval/05_patch_antenna.py on GPU):**

| Measurement                         | f_res (GHz) | Δ vs Balanis TL |
|-------------------------------------|-------------|-----------------|
| Balanis TL analytic                 | 2.4235      | —               |
| OpenEMS Harminv on port V(t)        | 2.4868      | +2.61 %         |
| rfx Harminv (probe ringdown)        | 2.4621      | +1.59 %         |

**Pass criteria (all PASS):**
- rfx internal self-consistency (Harminv vs lumped-port S11): **0.08 %**
- rfx vs analytic: **1.59 %**
- rfx vs OpenEMS Harminv: **0.99 %**
- S11 passivity |S11| ≤ 1: max **0.991**

rfx and OpenEMS agree to within 1 % on the TM010 patch resonance on
a 2.4 GHz FR4 patch with finite ground plane, CPML absorbers, and
probe-fed lumped excitation.

Two earlier VESSL attempts failed and are the reason this note carries
the install recipe explicitly:
- first attempt: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime` —
  bash heredoc termination failed inside `run: |-` block scalar. Use
  one-line `python3 -c "..."` instead.
- second attempt: same image — `python3-openems` apt package installs
  into Ubuntu 20.04's `/usr/bin/python3` (py3.8) which cannot run rfx
  (requires-python ≥ 3.10). Switched to the Nvidia JAX image on
  Ubuntu 22.04 whose `/usr/bin/python3` is py3.10 and matches the
  `python3-openems` apt binding.

---

## Memory saving — NU vs uniform at matched physical accuracy

Measured with `sim.estimate_ad_memory(n_steps=2000)` (added in this
session as part of issue #30 preflight):

| Configuration | cells | forward | checkpointed AD | full AD |
|-|-:|-:|-:|-:|
| **NU** (dx=dy=1mm, graded z 0.25→1mm) | 553,288 | 40.5 MB | 161.9 MB | 30.5 GB |
| Uniform fine (0.25mm isotropic)       | 20,724,826 | 1324 MB | 5297 MB | 996 GB |
| Uniform coarse (1mm isotropic — substrate under-resolved) | 553,288 | 35.4 MB | 141.4 MB | 26.6 GB |

**Ratios (NU vs accuracy-matched 0.25mm uniform):**
- **37.5× cell-count reduction**
- **32.7× forward memory reduction**
- **32.7× checkpointed-AD memory reduction**

The earlier plan-note ceiling of 5–10× is conservative for
thin-substrate antenna workloads. A thin substrate forces either
(a) isotropic 0.25 mm (996 GB full-AD → OOM on any commodity GPU)
or (b) NU grading that preserves substrate resolution only where it
is needed.

Practical consequence: on a 24 GB GPU, gradient-based inverse design
of this patch **fits comfortably in the NU path** (162 MB with
checkpointing) and **does not fit at all** under isotropic fine uniform
(5.3 GB checkpointed; full-AD in the hundreds of GB). The 2.85× demo
number reported by `nonuniform_patch_demo.py` compares z-axis cells
only — the 3D AD comparison is an order of magnitude larger.

---

## GitHub issues closed

| # | Commit | Impact |
|---|-|-|
| **#33** — forward() on NU mesh for differentiable optimization | `e1da198` | NU grids now usable in `sim.optimize()` / `sim.topology_optimize()`. JAX-grad through NU scan verified (AD vs centered FD rel err < 2 %). |
| **#32** — `maximize_directivity` always-zero gradient | `9f25ea2` | Absolute-power objective (~1e-27, float32 noise) replaced with scale-invariant directivity ratio `U(θ,φ)/P_rad` integrated over the hemisphere, `stop_gradient` on denominator. Issue-reported fix: directivity gate regressed from flat `-4.4e-31` loss to 9400 % improvement, 7.6 dBi. |
| **#29** — forward() waveguide port broadcast error `(8,1,1) vs (7,42,13)` | `f4f7dd1` | `_forward_from_materials` was not forwarding `grid.cpml_axes` — CPML state was allocated for axes without padding. The run() path forwards them; now forward() does too. Unblocks gradient-based waveguide optimisation. |
| **#30** — inverse-design preflight | `598f755` | Four new preflight checks (tightened resolution thresholds, NTFF↔PEC overlap, NTFF near-field λ/4 gap, reverse-mode AD memory with VRAM comparison). Exposed as `sim.estimate_ad_memory(n_steps, ...)` for direct use. |

Remaining issue #31 (checkpointed NU + mixed precision + temporal
windowing for memory-bounded inverse design) is scoped but deferred
to a separate session.

---

## Files of record

- Commits on `nonuniform-completion`:
  `e1da198`, `9f25ea2`, `f4f7dd1`, `598f755`, plus example `21f5675`.
- Demo: `examples/nonuniform_patch_demo.py`.
- VESSL job spec (gitignored by repo policy; keep local):
  `examples/vessl_nu_patch_crossval.yaml`.
- Completed VESSL run: `369367233458` (Apr 15).
- Crossval reference: `examples/crossval/05_patch_antenna.py` (rfx +
  OpenEMS, both on GPU).
- Prior session handoff:
  `docs/research_notes/2026-04-15_nonuniform_completion_handoff.md`
  and `2026-04-15_nonuniform_completion_session_handoff.md`.

---

## Do not repeat

- Do **not** use `pytorch/pytorch:*-runtime` as the rfx+openEMS image —
  Ubuntu 20.04's `/usr/bin/python3` is py3.8 and rfx needs ≥ 3.10.
  Use `nvcr.io/nvidia/jax:24.10-py3` (Ubuntu 22.04, py3.10) and
  `apt install python3-openems`; invoke `/usr/bin/python3` so the
  apt-installed binding is visible.
- Do **not** embed bash heredocs (`<<'PY'`) inside VESSL `run: |-`
  blocks — the terminator is indented by the block scalar and bash
  never reaches it. Use `python3 -c "..."` inline instead.
- Do **not** quote "2.85× memory saving" without saying "z-axis cells
  only" — the realistic 3D AD comparison is ~33× for this geometry
  class. The earlier plan-note 5–10× range is also conservative.
- Do **not** add a second preflight entry point when the existing
  `sim.preflight()` already collects warnings/errors into a list —
  extend flags, don't fork the harness (pattern used by #30).

---

## Next steps (for the next session)

1. File follow-up GitHub issues for the NU-forward gaps (`pec_occupancy`,
   `checkpoint=`, Floquet port) noted in `e1da198` commit message.
2. Issue #31 — memory-efficient inverse design. Scope:
   - plumb `checkpoint=True` into `run_nonuniform_path` so the NU scan
     body is rematerialised by `jax.checkpoint`.
   - optional mixed-precision for the scan state (fp16 fields, fp32
     accumulators for DFT / probes).
   - temporal windowing for very long runs where the full time series
     is not needed (inverse design typically wants only the S-parameter
     frequency probe, not the full trace).
3. Open a PR from `nonuniform-completion` → `main` once #31 scope is
   decided; current branch has 12 commits ahead, all passing the
   touched tests.
