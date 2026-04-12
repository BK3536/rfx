# CLAUDE.md - rfx

## Purpose
This file gives Claude the repo-local operating standard for `research/rfx/`.

## Read first
- `docs/agent-memory/index.md`
- `docs/agent-memory/public-site-sync-checklist.md`
- `docs/agent-memory/task_recipes/` — canonical patterns for common tasks

## Feature discovery (before writing new simulation code)

**Hard rule**: before writing a raw Yee loop (`update_h` / `update_e`) or an FFT-of-probe-time-series analysis, grep `rfx/api.py` for `def add_*` and ask: why can't I use the `Simulation` API for this?

Grep map:

| Task area | Grep target | Source dirs |
|-----------|-------------|-------------|
| Sources (lumped, TFSF, waveguide, Floquet) | `grep "def add_.*source\|def add_waveguide_port\|def add_floquet_port" rfx/api.py` | `rfx/sources/` |
| Probes / measurements | `grep "def add_(probe\|dft_plane_probe\|flux_monitor\|ntff_box)" rfx/api.py` | `rfx/probes/` |
| Materials / dispersion | `grep "def add_material" rfx/api.py` + `rfx/materials/{debye,lorentz,nonlinear}.py` | `rfx/materials/` |
| Boundaries | `grep -l "cpml\|upml\|pec\|floquet" rfx/boundaries/` | `rfx/boundaries/` |
| Post-processing | `grep "def flux_spectrum\|def harminv\|def compute_" rfx/probes/ rfx/` | `rfx/probes/probes.py`, `rfx/harminv.py` |
| S-parameters | `grep "compute_waveguide_s_matrix\|extract_s\|waveguide_port" rfx/api.py` | `rfx/ports/` |

**Canonical examples**: `tests/test_*.py` and `examples/crossval/*.py`. Task recipes in `docs/agent-memory/task_recipes/` give the decision tree for each task type.

**Never** time-series + FFT of probe data for R(f), T(f) of dispersive slabs — time-window artifacts. Use `add_flux_monitor` + two-run reference subtraction instead. See `docs/agent-memory/task_recipes/rt_measurement.md`.

## Project overview
- JAX-native 3D FDTD electromagnetic simulator for RF/microwave
- v1.4.0 | 730+ tests | PyPI: `rfx-fdtd` | GitHub: `bk-squared/rfx` (PUBLIC, MIT)
- 7 cross-validation scripts (`examples/crossval/01-07`), all with external references (Meep/OpenEMS/analytic)
- Preflight validation: 12 checks (mesh, CPML, NTFF, PEC, port, normalize)

## Core rule
Treat `research/rfx` as the source-of-truth for:
- simulator code,
- technical documentation,
- future public `rfx` content.

Do not treat:
- `teaching/creative-engineering-design` as the home of `rfx` docs,
- `infra/remilab-sites-gitops` seed pages as the authoring source,
- generated workspaces / `dist/` outputs as canonical.

## Development discipline
**Correctness > feature sprawl.** Ordered priorities:
1. Accuracy validation, correctness fixes
2. Non-uniform mesh, gradient behavior
3. Validated examples
4. Do NOT work on: neural surrogates, new features, docs polish — until explicitly requested

## Engineering principles
1. **Physical absolute coordinates** — probe/port/source 위치는 `h/2` 등 물리 좌표. `dx*1.5` 같은 cell-relative 금지.
2. **Axis-aware formulas** — `d_parallel/(Z0·d_perp1·d_perp2)`. Cubic cell 가정 (`1/(Z0·dx)`) 금지.
3. **Duck typing for grid** — `getattr(grid, 'dy', dx)` 패턴으로 Grid/NonUniformGrid 통합.
4. **JIT safety** — cell sizes는 Python float로 추출 → NamedTuple. scan 내 grid attribute 접근 금지.
5. **Evidence before defaults** — CPML 등 파라미터 변경은 sweep 실측 후 결정.

## Validation rules
- Validate simulator **physics**, not optimizer convergence.
- PEC for closed cavities, CPML for open structures — 혼용 금지.
- Subgridding은 3D에서 미검증 — overclaim 금지.
- Conformal PEC는 nonuniform/subgrid path에 전파되지 않음.

## Known issues (2026-04-09)
- ~~CPML defaults~~ → **해결**: 8 layers, kappa=1.0
- ~~Far-field dS~~ → **해결**: per-face dS (numpy + JAX)
- ~~test_reciprocity_two_port~~ → **해결**: threshold 5%
- ~~test_floquet NaN~~ → **해결**: explicit dx + PEC thickness
- ~~test_validation_suite~~ → **해결**: stale file removed
- ~~Distributed Debye/Lorentz~~ → **해결**: shard_map 5D fix (30/30 PASS)
- ~~thin conductor / RIS GPU failures~~ → **해결**: API tuple + PEC routing
- Multi-GPU TFSF/waveguide ports → single-device fallback (설계 제약, 문서화 완료)
- ADI 3D: LOD scheme, CFL 2-5x stable (10x diverges). Resonance error ~42% at 5x CFL (LOD splitting). 2D는 50x CFL, <2% resonance error
- `test_gradient_through_dft_plane` — GPU float32 precision only
- ~~CPML guided-mode reflection ~-4 dB~~ → **해결**: D/B-equivalent UPML + subpixel smoothing. Material-independent PML loss (σ/ε₀), Meep-matched sigma scaling (n/2), per-component eps for subpixel. |rfx−Meep| = 0.056 (was 0.10). Self-T=0.995.
- FluxMonitor finite-size regions 구현 완료 — `add_flux_monitor(..., size=(4e-3, 1e-3))`. 단, standing wave가 큰 환경에서는 full-plane이 더 안정적.

## remilab.ai relationship
- `remilab.ai` runtime is assembled in `infra/remilab-sites-gitops`.
- For `rfx`, that infra repo should act as a deployment/snapshot layer.
- Long-term target model:
  - public route: `/rfx/`
  - source-of-truth: `research/rfx`
- deploy snapshot: `infra/remilab-sites-gitops/.../seed-pages/rfx`
- For route ownership and publish flow, follow
  `docs/agent-memory/public-site-sync-checklist.md`.

## Documentation split
- `docs/agent-memory/index.md` = durable project knowledge, session handovers
- `docs/research_notes/` = chronological notes, handoffs, experiments, planning
- repo guides / API docs / future public docs = stable reusable knowledge
- If a note is internal or provisional, do not present it as finished public
  documentation.

## Writing rules
- Be precise about whether something is:
  - validated,
  - diagnostic,
  - provisional,
  - planned.
- Keep numerical claims tied to concrete evidence.
- Keep website-ownership / publication claims tied to actual repo paths and
  build flow.

## Test and lint
- test: `pytest` (730+ tests)
- lint: `ruff check`
- preflight: `sim.preflight()` (12 checks)
- GPU validation: `vessl run create -f scripts/vessl_v140_validation_v3.yaml`

## Safe operating assumption
If asked to prepare `rfx` material for remilab.ai, author it in this repo
first and assume gitops sync later unless explicit deployment work is requested.
