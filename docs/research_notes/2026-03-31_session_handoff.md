# 2026-03-31 Session Handoff

## Accomplished This Session

### Codex Review + Commit
- Reviewed Codex waveguide overhaul (+1,793 lines, 12 files)
- Forward/backward wave decomposition, N-port S-matrix, multi-axis ports
- Deduplicated `_dft_window_weight` → `rfx/core/dft_utils.py`
- 152 tests passing → committed as 7c0f4a2

### Phase 6B-6D (all PASS, committed f50737d)
- 6B: Optimizer convergence tests (slab S11 47% improvement, TX 2.1x)
- 6C: CWSource + CustomWaveform (4 tests, 3.45s)
- 6D: Magnetic material validation (|R|=0.344 vs 0.333, v=0.0% error)

### Phase 6E1: Two-Run Normalization (committed 0c77070)
- **Key result: normalized |S21| = 1.0000** for empty waveguide (was 0.74 raw)
- Element-wise S_dev/S_ref normalization
- Known issue: S11 also normalizes to ~1 (small/small), needs incident-wave normalization

### Phase 6E2: Field Decay Convergence (committed a437bd4)
- `run_until_decay()` with JIT step function + Python loop
- `Simulation.run(until_decay=1e-3)` API parameter
- 4 tests passing

### Phase 6E3: Conservation Law Tests (committed 0c77070)
- Passivity: xfail (V/I extraction |S|>1 at some freqs)
- Unitarity: xfail (same root cause)
- Reciprocity: PASS (9% for asymmetric structure)
- Mesh convergence: PASS
- Causality: PASS

### Multiport Reciprocity Verification
- 2-port x-normal: 3.2% mean error (good)
- 3-port T-junction mixed x/y: 2.7% x↔x, 10% x↔y (marginal)
- Passivity violated in raw S-params (column sum 3.72)

### Validation Strategy Document
- 3-tier hierarchy: Analytical → Conservation Laws → Cross-validation (report-only)
- Meep/OpenEMS errors should NOT gate rfx pass/fail
- Conservation laws are tool-independent physics invariants

### Meep/OpenEMS Accuracy Research
- Meep: eigenmode overlap integral, normalization run, subpixel smoothing
- OpenEMS: analytical TE/TM profiles, real(β) de-embedding (inferior)
- rfx already better: CPML > Meep's UPML, complex de-embedding > OpenEMS real-only

## Pending (Codex dispatched, awaiting results)

| Task | Status | Blocker |
|------|--------|---------|
| 6E4 Overlap integral | Codex running | — |
| 6E5 CFS-CPML fix | Codex debugging | x-axis κ correction missing |
| 6E6 Subpixel fix | Codex debugging | wrong peak freq, convergence=0 |
| 6G RCS pipeline | Codex running | — |
| 6F Oblique TFSF | Spec written, not dispatched | Complex (retarded time interp) |

## Key Numbers

| Metric | Before | After |
|--------|--------|-------|
| Total tests | 134 | 172+ (160 pass + 3 xfail + new) |
| |S21| empty WG (normalized) | 0.74 | **1.0000** |
| Reciprocity (2-port) | 3% | 3% (unchanged) |
| Passivity | violated (3.72) | violated (needs overlap integral) |
| Optimizer convergence | untested | 47% S11 improvement verified |
| mu_r validation | untested | |R|=3.1%, v=0.0% |

## Git State
- Branch: main
- Last push: a437bd4
- Remote: up to date
- Uncommitted: 6E5/6E6 files (tests failing, being debugged by Codex)

## Next Steps (priority order)
1. Verify and commit 6E5/6E6/6E4/6G when Codex returns
2. Implement proper incident-wave normalization (fix S11→1 issue in 6E1)
3. Un-xfail passivity/unitarity once overlap integral (6E4) lands
4. 6F Oblique TFSF (dispersion-matched 1D aux grid)
5. Full regression + program spec update
6. Phase 7: README, examples, public release
