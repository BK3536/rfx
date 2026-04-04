# Validation

This page summarizes rfx validation evidence with actual simulation results.

## Accuracy Validation (5 cases)

Benchmarked against Balanis "Antenna Theory" and Pozar "Microwave Engineering":

| # | Structure | Reference | Error | Strength |
|---|-----------|-----------|-------|----------|
| 1 | Patch antenna resonance | Balanis Ch 14 | 1.97% | Solid |
| 2 | WR-90 TE10 cutoff | Analytical | 0.60% | Solid |
| 3 | Dielectric cavity TM110 | Analytical | 0.016% | Solid |
| 4 | Microstrip Z0 | Hammerstad-Jensen | 0.47% | Solid |
| 5 | Coupled-line filter | Pozar Ch 8 | 22.5% | Weak (formula limitation) |

### Patch antenna (Balanis Ch 14)
![Patch antenna validation](./images/01_patch_balanis.png)

### Waveguide TE10 cutoff
![Waveguide TE10 validation](./images/02_waveguide_te10.png)

### Dielectric cavity TM110
![Dielectric cavity validation](./images/03_cavity_tm110.png)

### Microstrip impedance
![Microstrip Z0 validation](./images/04_microstrip_z0.png)

### Coupled-line filter
![Coupled filter validation](./images/05_coupled_filter.png)

## GPU Validation (6 cases)

Advanced physics tests run on RTX 4090, cross-validated by Codex review.

| # | Test | Key Metric | Verdict |
|---|------|------------|---------|
| 1 | Topology opt pipeline | 4.36x energy gain | Pipeline smoke test |
| 2 | PEC cavity TE101 (Pozar Ch 6) | **0.41% error** | Solid physics |
| 3 | Series RLC ADE spectral effect | Measurable shift | ADE functional |
| 4 | Free-space 1/r^2 decay (CPML) | **exponent 2.448** | Defensible |
| 5 | Dielectric lens (CPML) | **13.06 dB** gain | Optimizer + CPML |
| 6 | Debye material fit (Kaatze 1989) | eps_inf 3.68% err | Self-consistency |

### Topology optimization pipeline
![Topology optimization](./images/01_patch_opt_validation.png)

### PEC cavity resonance via Harminv
![PEC cavity resonance](./images/02_filter_validation.png)

### Series RLC ADE spectral effect
![RLC ADE validation](./images/03_matching_validation.png)

### Free-space energy decay (CPML)
![Free-space decay](./images/04_coupling_validation.png)

### Dielectric lens focusing (CPML)
![Lens focusing](./images/05_lens_validation.png)

### Debye material characterization
![Debye fitting](./images/06_matfit_validation.png)

## Honesty notes

Following our validation discipline:

**Strongly benchmarked** (make confident claims):
- PEC cavity resonances (0.016% - 0.41%)
- Rectangular waveguide cutoff (0.60%)
- Dielectric-loaded cavity
- CPML free-space energy decay

**Functional / diagnostic** (working but not strong physics validation):
- Topology optimization pipeline (smoke test, not physics)
- RLC ADE (functional test, direction depends on mode coupling)
- Debye fitting (self-consistency, not robustness test)
- Lens focusing (optimizer converges, but criterion is lenient)

**Evolving**:
- Patch antenna feed/port workflows
- Coupled-line filter (25% tolerance reflects formula limitation)

## Recommended wording discipline

When documenting results:
- say **benchmark** or **comparison** when appropriate
- reserve stronger wording for cases with clear reproducible evidence
- distinguish analytical agreement, Meep/OpenEMS agreement, diagnostic results, and evolving examples
