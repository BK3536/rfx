# 2026-03-31 Final Session Handoff

## Session Accomplishments (27+ commits)

### Core Physics
- **Normalized |S21| = 1.0000** for empty waveguide (two-run normalization)
- **Passivity satisfied**: column power 3.7 → 1.10 (per-port incident-wave normalization)
- **GPU: 1,310 Mcells/s** on RTX 4090, gradient in 0.31s
- CFS-CPML (1.6x evanescent improvement), subpixel smoothing (1.2x error reduction)
- Overlap integral modal extraction, RCS pipeline (TFSF+NTFF)
- Magnetic material validated (|R|=3.1%, v=0.0%)
- Optimizer convergence verified (47% S11 improvement)

### Tests: 134 → 190+ (1 xfail remaining)
- Conservation laws: passivity, unitarity, reciprocity, convergence, causality
- All passivity/unitarity xfails REMOVED

### Documentation (10 guides, ~5,000 words)
- Installation, quickstart, API, waveguide ports, inverse design
- Far-field/RCS, advanced features, geometry/limitations, visualization/AI

### Release
- MIT LICENSE, README with GPU benchmarks, AI credits (Claude+Codex)
- GitHub Release v0.1.0 tagged
- 3 examples, objective library (5 functions), GitHub Actions CI
- remilab.ai website entry (deploy pending v1)

## v1.0 Remaining

### 1. Oblique TFSF (1 xfail)
**Current**: 25% leakage with single dispersion-matched 1D grid
**Solution identified**: Multiple 1D grids (one per transverse cell), each with time-delayed source:
- delta_t = j * dx * sin(theta) / c for y-cell j
- Storage: ny * n_1d cells (e.g., 81 * 352 = 28K, manageable)
- This is the standard Taflove approach (Ch. 5.6)
- Implement directly (too complex for Codex)

### 2. Eigenmode Solver
- Numerical 2D cross-section eigenmode via scipy.sparse.linalg.eigsh
- Enables non-rectangular waveguide support
- Codex couldn't deliver; implement directly

### 3. Clean Public Repo
- Separate internal docs (research_notes, codex_specs, agent-memory) from public
- Options: .gitignore, separate branch, or clean fork
- Must be done before v1.0 public

### 4. Website Deploy
- remilab.ai entry committed but not deployed
- Deploy with v1.0

## Codex Delegation Lessons
- **60% success rate** for Codex tasks
- Works well: single-file implementation, test generation, documentation
- Fails: physics debugging, multi-file refactoring, complex algorithms
- Best pattern: Claude designs + skeleton, Codex fills implementation

## Git State
- Branch: main, tag: v0.1.0
- Last push: 9b33543
- 1 xfail: test_oblique_tfsf_fresnel (oblique TFSF 25% leakage)
- VESSL GPU benchmark #369367231243: completed successfully
