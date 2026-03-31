# Codex Spec 6D: Magnetic Material (mu_r) Validation

## Goal
Validate that the existing `mu_r` parameter in the Yee H-update works correctly
for materials with relative permeability != 1.

## Context
- `rfx/core/yee.py` `update_h()` already divides by `mu_r` in the H-field update:
  `H += (dt / (MU_0 * mu_r)) * curl_E`
- `MaterialArrays` has a `mu_r` field (default all-ones)
- No existing test sets `mu_r != 1` or validates the physics

## Deliverable
`tests/test_magnetic.py` with two validation tests:

### Test 1: `test_magnetic_impedance_reflection`
Verify that a plane wave reflecting off a magnetic slab produces the correct
Fresnel reflection coefficient.

For a slab with (eps_r=1, mu_r=4):
- η = η₀ * √(μ_r/ε_r) = η₀ * 2
- R = (η₂ - η₁) / (η₂ + η₁) = (2η₀ - η₀) / (2η₀ + η₀) = 1/3
- |R|² = 1/9 ≈ 0.111

Setup:
- Domain (0.10, 0.006, 0.006), CPML on x, periodic y/z
- TFSF plane wave source (Ez polarization, +x direction)
- Magnetic slab: mu_r=4, eps_r=1, located at x_interface = nx//2
- Probe in scattered-field region (x < x_lo)
- Probe in total-field region for incident reference (vacuum run)
- n_steps: enough for reflection to arrive at probe (compute from geometry)

Measure:
- Spectral |R| = |FFT(scattered)| / |FFT(incident)| averaged over 3-7 GHz band

Assertions:
- Mean |R| in band matches 1/3 within 15% tolerance
- No NaN in fields

### Test 2: `test_magnetic_phase_velocity`
Verify that a pulse travels at v = c / √(μ_r * ε_r) through a magnetic medium.

Setup:
- Domain (0.10, 0.006, 0.006), CPML on x, periodic y/z
- TFSF plane wave, Ez polarization
- Two regions: vacuum (left half), magnetic medium mu_r=4, eps_r=1 (right half)
- Two probes inside the total-field region:
  - Probe A in vacuum at x_A
  - Probe B in magnetic medium at x_B (same distance from interface)

Measure:
- Arrival time at each probe (peak of |Ez(t)|)
- Velocity ratio: t_B / t_A should equal √(mu_r) = 2.0
  (since both probes are same distance from source, but medium B has v = c/2)

Actually simpler approach — just measure in uniform magnetic medium:
- Fill entire domain with mu_r=4, eps_r=1
- TFSF source, two probes at x1 and x2 inside total-field region
- Travel time Δt = (x2 - x1) * √(mu_r * eps_r) / c
- Measured Δt (from peak arrival) should match within 10%

Assertions:
- Measured velocity ratio v_measured / (c/√(mu_r)) within 10%
- No NaN

## Constraints
- Each test < 90 seconds on CPU
- Do NOT modify any existing source files
- Only create `tests/test_magnetic.py`
- Use existing TFSF infrastructure (`init_tfsf`, `apply_tfsf_e/h`, etc.)
- Use manual stepping loop (not the compiled runner) for direct control of
  material placement relative to TFSF box

## Verification
Run: `pytest -xvs tests/test_magnetic.py`
Both tests must pass.
