# Session 3 Handoff: Cross-Validation Expansion to 23 Scripts (2026-04-08)

## What Was Done

### Fixed: Debye Water Crossval (#15)
- **Root cause chain** (5 iterations to fix):
  1. Point source in thin 3D domain → spherical spreading, no plane wave
  2. TFSF plane wave → probe coordinate error (physical coords vs grid index with padding)
  3. Fixed probe → Hanning window bias (incident at edge, reflected at center)
  4. Time-gated FFT → R still ~50% of expected
  5. **Final fix**: Water Box didn't fill CPML padding in y/z → edge diffraction. Oversized Box (`-yz_pad` to `dom_yz + yz_pad`) fills padding cells.
- **Method**: TFSF + reference subtraction + time-gated FFT
- **Result**: R(10 GHz) = 0.755 vs Fresnel 0.791, 9.8% relative RMS

### New Crossval Scripts (8 total this session)

| # | Script | Key Result |
|---|--------|------------|
| 15 | Debye Water (fixed) | R(10GHz)=0.755, 9.8% RMS vs Fresnel |
| 17 | CRLH Unit Cell | beta=0 at 1.87 GHz, S21 passband 7.2 GHz |
| 18 | NU Mesh Convergence | 8.3%→2.3% monotonic, order 0.74 |
| 19 | Dipole Above PEC | D=5.7 dBi, peak 64 deg, image theory corr 0.63 |
| 20 | Rectangular Waveguide Modes | TE101=6.72 GHz (4.2% err), 7 modes |
| 21 | Dielectric Resonator Q | 4.67 GHz, Q=985, 6 modes |
| 22 | 2-Port Reciprocity | S12=S21 (0.000%), |S21|=1.000, power=1.044 |
| 23 | Drude Metal Sphere | D=6.7 dBi, Drude+TFSF+NTFF pipeline OK |

### Test Suite
- **490 passed, 1 failed** (`test_openems_crossval` — OpenEMS binary not installed, not an rfx bug)

## Commits (all on main, pushed)
- `5deb29e` → `fc207c5` (rebased) — Debye fix, CRLH, NU convergence, dipole+PEC
- `781128f` — waveguide modes, resonator Q, reciprocity, Drude sphere

## Key Findings

### TFSF Measurement for Dispersive Materials
- TFSF `margin` in physical coords = `tfsf_margin * dx` (CPML padding is outside physical domain)
- Material Box must extend into CPML padding for transverse dimensions to prevent edge diffraction at y/z boundaries
- Time gating is essential: Hanning window over full signal biases short vs long pulse segments
- Reference subtraction (vacuum - water) gives clean reflected field for R(f) extraction

### CRLH Geometry
- Series gap at cell BOUNDARIES, shunt via at cell CENTERS (initially had via at gap position, shorting the capacitor)
- At dx=0.5mm, 1mm gap = 2 cells (barely resolved but sufficient for CRLH physics demonstration)

### Waveguide Port S-Matrix
- `compute_waveguide_s_matrix(normalize=True)` gives perfect reciprocity (S12=S21, 0.000% error)
- Known limitation: S11 ≠ S22 asymmetry in port-direction normalization (empty waveguide: S11=0.19 vs S22=0.72)
- `pec_faces` is NOT propagated through `compute_waveguide_s_matrix` — it only reads `grid.cpml_axes`
- Dielectric material extending into CPML y/z corrupts S-parameters — use PEC obstacles or keep material away from CPML

### Mesh/CPML Constraints
- CPML thickness (`cpml_layers * dx`) must be << waveguide cross-section for waveguide port extraction
- WR-90 (a=22.86mm): dx=1.5mm gives CPML=12mm, entirely filling y. Need dx≤1.0mm.

## Cross-Validation Status (23 scripts, all PASS)

| # | Script | Physics |
|---|--------|---------|
| 01-11 | Original suite | waveguide, RCS, Mie, Fresnel, Lorentz, dipole, Kerr, patch, NU |
| 12 | MSL Notch Filter | microstrip, non-uniform z, pec_faces, wire ports |
| 13 | CPML Reflection Sweep | CPML quality quantification |
| 14 | Cavity Q 3D | PEC cavity, Harminv, analytical match |
| 15 | Debye Water | TFSF + Debye ADE + reference subtraction |
| 16 | Helical Antenna | PolylineWire, NTFF, 3D far-field |
| 17 | CRLH Unit Cell | metamaterial TL, gap cap + via inductor, dispersion |
| 18 | NU Mesh Convergence | convergence order with non-uniform z-mesh |
| 19 | Dipole Above PEC | pec_faces, NTFF, image theory |
| 20 | Waveguide Modes | multi-mode Harminv extraction |
| 21 | Resonator Q | dielectric in PEC cavity, Q extraction |
| 22 | 2-Port Reciprocity | waveguide port S-matrix, S12=S21 |
| 23 | Drude Metal Sphere | Drude dispersion + TFSF + NTFF |

## Remaining Work

### Crossval Plan (feasible without new features)
- Horn antenna (large 3D, GPU recommended)
- Oblique Fresnel R(θ) (TFSF oblique + Floquet)
- GPR A-scan (pulsed, lossy soil)
- Coupled-line BPF (Tidy3D reference)

### Blocked
- Bent-patch antenna → needs cylindrical FDTD

### Known Issues to Fix
- `compute_waveguide_s_matrix`: S11≠S22 port-direction asymmetry
- `pec_faces` not propagated to `compute_waveguide_s_matrix` internal runs
- Dielectric in CPML region corrupts waveguide S-params

### Feature Gaps
- Cylindrical FDTD (bent-patch blocked)
- PolylineWire JIT optimization (>100 segments slow)
- ADI 3D Namiki maturation
