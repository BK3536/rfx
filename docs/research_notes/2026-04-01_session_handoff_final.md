# 2026-04-01 Session Handoff (Final)

## Session Accomplishments (35+ commits)

### Tests: 134 → 230+ (0 xfails)
### Source modules: 22 → 37+
### Test files: 26 → 45+

### Major Features Delivered

**Phase 6 (Simulator Quality)**
- Normalized |S21| = 1.0000 (two-run normalization)
- Passivity fix: column power 3.7 → 1.10 (incident-wave normalization)
- Oblique TFSF fix: leakage 27% → 8e-7 (dx_1d = dx)
- CFS-CPML: 1.6x evanescent improvement
- Subpixel smoothing: 1.2x error reduction
- Overlap integral modal extraction
- RCS pipeline (TFSF + NTFF)
- Field decay convergence criterion
- CW + custom waveform sources
- Magnetic material (mu_r) validation
- Optimizer convergence verified
- Conservation law tests (passivity, unitarity, reciprocity, convergence, causality)

**v1.0 Features**
- Eigenmode solver (analytical + numerical via scipy)
- Eigenmode ↔ waveguide port integration
- Gradient coverage tests (lossy, mu_r, CW, spatial)
- Gradient behavior documentation guide
- Objective library (5 functions)
- LICENSE, README (with GPU benchmarks), GitHub Actions CI
- 10 documentation guides + roadmap + subgridding survey
- 3 examples + GPU benchmark (1,310 Mcells/s RTX 4090)
- GitHub Release v0.1.0 tagged
- remilab.ai website entry

**v1.1 Features**
- Dey-Mittra conformal PEC boundaries
- Batch simulation (ParameterSweep + SimulationDataset)

**v2 Features**
- Kerr nonlinear material (JAX-differentiable)
- Holland thin-wire subcell model
- **SBP-SAT subgridding: 1D + 2D TM (Phase 1-2 complete)**

### Strategic Documents
- Differentiable solver roadmap (rfx = physics oracle, not ML model)
- Validation strategy (3-tier: analytical → conservation laws → cross-val)
- Subgridding survey (SBP-SAT recommended, no existing differentiable subgridding)
- SBP-SAT implementation plan (4 phases)

## Next Session: Phase 3 (3D SBP-SAT + scan integration)

### What's needed
1. Extend 2D TM subgridding to full 3D (6 field components, 6 interface faces)
2. Integrate with jax.lax.scan in simulation.py (compound carry)
3. Fine sub-steps via jax.lax.fori_loop inside scan body
4. High-level API: Simulation.add_refinement(Box(...), ratio=3)
5. Verify jax.grad flows through subgrid interface

### Key design decisions
- SAT penalty must be in field-space (not coefficient-space — causes NaN)
- Courant condition: dt limited by fine grid, dt_c = ratio * dt
- SAT uses downsample/upsample for coarse-fine field mapping

### References
- Cheng et al., arXiv:2202.10770 (staggered Yee, no field modification)
- Cheng et al., IEEE TAP 2023 (arbitrary grid ratios)
- Cheng et al., Semantic Scholar 2022 (3D theory)

## Git State
- Branch: main
- Last push: 6cba4ae
- Tag: v0.1.0 (will need v1.0 re-tag after Phase 3)
- Clean working tree
