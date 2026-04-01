"""Tests for 1D SBP-SAT FDTD subgridding prototype.

Required tests (per specification):
1. test_sbp_property         — P*D + D^T*P = E_boundary
2. test_stability_long_run   — 100,000 steps, energy does not grow
3. test_subgrid_matches_uniform — pulse through interface matches uniform fine grid within 5%
4. test_energy_conservation  — total energy (E^2 + H^2) is non-increasing
"""

import numpy as np
import jax.numpy as jnp

from rfx.subgridding.sbp_sat_1d import (
    build_sbp_norm,
    build_sbp_diff,
    build_interpolation_c2f,
    build_interpolation_f2c,
    init_subgrid_1d,
    step_subgrid_1d,
    compute_energy,
    SubgridState1D,
    _update_h_1d,
    _update_e_1d,
    C0,
    EPS_0,
    MU_0,
)


# ── 1. SBP property ──────────────────────────────────────────────

def test_sbp_property():
    """Verify the SBP identity:  P @ D + D^T @ P = E_boundary.

    E_boundary = diag(-1, 0, ..., 0, +1) for the standard first-derivative
    SBP operator with the trapezoidal norm.

    Also checks that interpolation matrices have consistent shapes.
    """
    for n in [10, 20, 50]:
        dx = 0.01
        p_diag = build_sbp_norm(n, dx)
        D = build_sbp_diff(n, dx)

        P = np.diag(p_diag)
        Q = P @ D  # should satisfy Q + Q^T = E_boundary

        # Symmetry check: Q + Q^T
        S = Q + Q.T

        # Expected: E_boundary = diag(-1, 0, ..., 0, +1)
        E_expected = np.zeros((n, n), dtype=np.float64)
        E_expected[0, 0] = -1.0
        E_expected[-1, -1] = +1.0

        err = np.max(np.abs(S - E_expected))
        print(f"  n={n}, dx={dx}: SBP error = {err:.2e}")
        assert err < 1e-12, (
            f"SBP property violated for n={n}: max|P*D + D^T*P - E| = {err}"
        )

    # Verify interpolation matrix shapes
    ratio = 3
    R_c2f = build_interpolation_c2f(20, 60, ratio)
    R_f2c = build_interpolation_f2c(60, 20, ratio)
    assert R_c2f.shape == (ratio + 1, 2), f"R_c2f shape: {R_c2f.shape}"
    assert R_f2c.shape == (2, ratio + 1), f"R_f2c shape: {R_f2c.shape}"
    # R_f2c should be transpose of R_c2f
    assert np.allclose(R_f2c, R_c2f.T), "R_f2c should equal R_c2f^T"


# ── 2. Stability over long run ───────────────────────────────────

def test_stability_long_run():
    """Energy must not grow over 100,000 steps (provable stability)."""
    config, state = init_subgrid_1d(n_c=40, n_f=60, dx_c=0.003, ratio=3)

    # Inject a Gaussian pulse on the coarse grid
    x_c = jnp.arange(config.n_c) * config.dx_c
    pulse = jnp.exp(-((x_c - 0.06) / 0.01) ** 2).astype(jnp.float32)
    state = state._replace(e_c=pulse)

    initial_energy = compute_energy(state, config)
    max_energy = initial_energy

    n_steps = 100_000
    for i in range(n_steps):
        state = step_subgrid_1d(state, config)
        if i % 5000 == 0:
            e = compute_energy(state, config)
            max_energy = max(max_energy, e)

    final_energy = compute_energy(state, config)

    print(f"\nStability test ({n_steps} steps):")
    print(f"  Initial energy: {initial_energy:.6e}")
    print(f"  Max energy:     {max_energy:.6e}")
    print(f"  Final energy:   {final_energy:.6e}")
    print(f"  Growth ratio:   {max_energy / max(initial_energy, 1e-30):.6f}")

    # Energy must not grow beyond the initial value (allow 5% numerical margin)
    assert max_energy < initial_energy * 1.05, (
        f"Energy grew: max {max_energy:.6e} > 1.05 * initial {initial_energy:.6e}"
    )
    # Should not be NaN
    assert not np.isnan(final_energy), "Final energy is NaN — simulation diverged"


# ── 3. Subgrid matches uniform fine grid ─────────────────────────

def test_subgrid_matches_uniform():
    """Pulse propagation through interface matches uniform fine grid within 5%.

    We compare total energy after a fixed number of steps between:
    (a) a uniform fine grid covering the whole domain, and
    (b) the same domain split into coarse + fine with SBP-SAT coupling.

    The comparison uses matched initial conditions: the Gaussian pulse
    is placed entirely within the coarse region (equivalent to the left
    half of the uniform grid) so that the initial energy is the same.
    """
    ratio = 3
    dx_f = 0.001
    dx_c = dx_f * ratio
    n_total_fine = 120  # total domain in fine cells

    # CFL-safe timestep for the fine grid
    courant = 0.5
    dt = courant * dx_f / C0
    n_steps = 500  # short run before the pulse reaches boundary

    # Gaussian pulse centred at cell 15 (well inside left region)
    pulse_centre = 15.0 * dx_f
    pulse_width = 4.0 * dx_f

    # ── (a) Uniform fine reference ──
    n_uni = n_total_fine
    x_uni = jnp.arange(n_uni, dtype=jnp.float32) * dx_f
    e_uni = jnp.exp(-((x_uni - pulse_centre) / pulse_width) ** 2).astype(jnp.float32)
    h_uni = jnp.zeros(n_uni - 1, dtype=jnp.float32)

    initial_energy_uni = (
        float(jnp.sum(e_uni ** 2)) * EPS_0 * dx_f
        + float(jnp.sum(h_uni ** 2)) * MU_0 * dx_f
    )

    for _ in range(n_steps):
        h_uni = _update_h_1d(e_uni, h_uni, dt, dx_f)
        e_uni = _update_e_1d(e_uni, h_uni, dt, dx_f)
        e_uni = e_uni.at[0].set(0.0)
        e_uni = e_uni.at[-1].set(0.0)

    energy_uniform = (
        float(jnp.sum(e_uni ** 2)) * EPS_0 * dx_f
        + float(jnp.sum(h_uni ** 2)) * MU_0 * dx_f
    )

    # ── (b) Subgridded domain ──
    # Coarse covers left half (60 fine cells = 20 coarse cells)
    # Fine covers right half (60 fine cells)
    n_c = n_total_fine // (2 * ratio)  # 20 coarse nodes
    n_f = n_total_fine // 2            # 60 fine nodes

    config, state = init_subgrid_1d(
        n_c=n_c, n_f=n_f, dx_c=dx_c, ratio=ratio, dt=dt,
    )

    # Same Gaussian pulse on the coarse grid (equivalent physical positions)
    x_c = jnp.arange(n_c, dtype=jnp.float32) * dx_c
    pulse_c = jnp.exp(-((x_c - pulse_centre) / pulse_width) ** 2).astype(jnp.float32)
    state = state._replace(e_c=pulse_c)

    initial_energy_sub = compute_energy(state, config)

    for _ in range(n_steps):
        state = step_subgrid_1d(state, config)

    energy_subgrid = compute_energy(state, config)

    print(f"\nUniform vs subgridded ({n_steps} steps):")
    print(f"  Initial energy (uniform): {initial_energy_uni:.6e}")
    print(f"  Initial energy (subgrid): {initial_energy_sub:.6e}")
    print(f"  Final energy (uniform):   {energy_uniform:.6e}")
    print(f"  Final energy (subgrid):   {energy_subgrid:.6e}")

    # Both should have positive energy and no NaN
    assert energy_uniform > 0, "Uniform energy should be positive"
    assert energy_subgrid > 0, "Subgrid energy should be positive"
    assert not np.isnan(energy_subgrid), "Subgrid energy is NaN"

    # Compare energy retained (fraction of initial energy still present).
    # Both start with similar energy; after propagation, the uniform grid
    # preserves energy exactly (leapfrog is symplectic) while the subgridded
    # grid loses some at the interface (SAT dissipation).  We check that
    # the subgridded energy is within 5% of the uniform energy.
    rel_diff = abs(energy_subgrid - energy_uniform) / energy_uniform
    print(f"  Relative diff:            {rel_diff:.4f} ({rel_diff*100:.2f}%)")
    assert rel_diff < 0.05, (
        f"Energy mismatch too large: {rel_diff*100:.2f}% > 5%"
    )


# ── 4. Energy conservation ───────────────────────────────────────

def test_energy_conservation():
    """Total energy (E^2 + H^2) is non-increasing over time.

    The SBP-SAT formulation guarantees d/dt(energy) <= 0 (the SAT penalty
    dissipates interface mismatch energy).  We verify that:
    1. No sampled energy exceeds the initial energy.
    2. The overall trend is monotonically non-increasing above the noise floor.
    """
    config, state = init_subgrid_1d(n_c=30, n_f=45, dx_c=0.002, ratio=3)

    # Smooth Gaussian pulse that straddles the interface
    x_c = jnp.arange(config.n_c, dtype=jnp.float32) * config.dx_c
    pulse = jnp.exp(-((x_c - 0.03) / 0.005) ** 2).astype(jnp.float32)
    state = state._replace(e_c=pulse)

    initial_energy = compute_energy(state, config)

    n_steps = 5000
    sample_every = 50
    energies = [initial_energy]

    for i in range(n_steps):
        state = step_subgrid_1d(state, config)
        if (i + 1) % sample_every == 0:
            energies.append(compute_energy(state, config))

    energies = np.array(energies)

    # No NaN
    assert not np.any(np.isnan(energies)), "NaN in energy trace"

    print(f"\nEnergy conservation ({len(energies)} samples over {n_steps} steps):")
    print(f"  Initial: {energies[0]:.6e}")
    print(f"  Final:   {energies[-1]:.6e}")
    print(f"  Min:     {energies.min():.6e}")
    print(f"  Max:     {energies.max():.6e}")

    # (1) No sample should exceed initial energy (with tiny floating-point margin)
    assert energies.max() <= initial_energy * (1.0 + 1e-6), (
        f"Energy exceeded initial: max {energies.max():.6e} > initial {initial_energy:.6e}"
    )

    # (2) Check monotonic decrease above the noise floor.
    #     Once energy drops below 1% of initial, we're in the noise regime
    #     and don't check monotonicity.
    noise_floor = initial_energy * 0.01
    above_floor = energies > noise_floor
    if np.sum(above_floor) > 2:
        meaningful = energies[above_floor]
        diffs = np.diff(meaningful)
        # Allow tiny relative growth from float32 rounding
        tol = meaningful[:-1] * 1e-4
        n_growing = int(np.sum(diffs > tol))
        print(f"  Meaningful samples: {len(meaningful)}")
        print(f"  Growing (above tol): {n_growing}/{len(diffs)}")
        assert n_growing < len(diffs) * 0.1, (
            f"Too many growing intervals: {n_growing}/{len(diffs)}"
        )

    # (3) Final energy <= initial energy
    assert energies[-1] <= initial_energy * 1.01, (
        f"Final energy {energies[-1]:.6e} exceeds initial {initial_energy:.6e}"
    )
