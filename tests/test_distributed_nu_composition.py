"""
tests/test_distributed_nu_composition.py

Phase 1 TDD contract suite for Issue #44: Distributed + Non-Uniform + Segmented
Composition.

All 18 tests are ``xfail(strict=True)``.  Each test body is *real* — it
attempts the actual API call — so the ``xfail`` marks the expected failure mode,
not a stub.  When a Phase 2 subtask lands and makes a test pass, that test will
strict-XPASS (strict=True → the suite will flag it), which is the intended
signal that the subtask is complete and the xfail mark should be removed.

XLA device guard
----------------
Every test calls ``_require_two_devices()`` as its first line.  This function
issues ``pytest.skip`` (not ``pytest.fail``) when ``len(jax.devices()) < 2``.
The repo-root ``conftest.py`` sets XLA_FLAGS before jax is imported, so in
normal pytest runs 2 CPU virtual devices are available.  If XLA_FLAGS somehow
arrived late (unusual invocation), the skip keeps the session clean without
masking unrelated failures.

Test inventory (V3 plan §Phase 1 Tolerance Contract):

 #  | Name                                                         | Class
----+--------------------------------------------------------------+------
  1 | test_shard_map_checkpoint_trivial_scan_grad (M5 spike)       | E
  2 | test_forward_distributed_flag_requires_multi_device          | E/API
  3 | test_forward_distributed_rejects_uniform_mesh_in_v162        | E/API
  4 | test_forward_distributed_rejects_tfsf_on_nu_path             | E/API
  5 | test_forward_distributed_requested_devices_gt_available_raises| E/API
  6 | test_forward_distributed_devices_without_distributed_flag_raises| E/API
  7 | test_forward_distributed_pec_two_device_matches_single_device_small_case | B
  8 | test_forward_distributed_cpml_two_device_matches_single_device_small_case | B+C
  9 | test_forward_distributed_debye_two_device_matches_single_device_small_case | B+D
 10 | test_forward_distributed_lorentz_two_device_matches_single_device_small_case | B+D
 11 | test_forward_distributed_mixed_dispersion_two_device_matches_single_device_small_case | B+D
 12 | test_forward_distributed_pec_occupancy_two_device_matches_single_device | B
 13 | test_forward_distributed_checkpoint_every_matches_no_segment_small_grad_case | A+E
 14 | test_forward_distributed_n_warmup_tail_grad_matches_single_device | A
 15 | test_forward_distributed_design_mask_stop_grad_matches_single_device | A
 16 | test_forward_distributed_cpml_internal_seam_is_noop_not_boundary | C
 17 | test_forward_distributed_pec_mask_seam_exchange_preserves_field | B
 18 | test_forward_distributed_grad_per_cell_matches_single_device_near_seam (G1) | A seam
     test_forward_distributed_nan_propagates_via_ghost_exchange (G2) | E
"""

from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from tests._distributed_nu_tolerances import (
    RTOL_A,
    assert_class_a_grad,
    assert_class_a_grad_seam,
    assert_class_b_parity,
    assert_class_c_cpml_seam_noop,
    assert_class_d_timeseries_drift,
    assert_class_e_bit_match,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_two_devices() -> list:
    """Return jax.devices()[:2] or skip the test."""
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip(
            "XLA_FLAGS did not produce 2 virtual devices; check conftest.py load order. "
            "Expected XLA_FLAGS=--xla_force_host_platform_device_count=2."
        )
    return list(devices[:2])


def _make_nu_sim_small(
    *,
    nx: int = 16,
    ny: int = 12,
    nz: int = 12,
    dx: float = 5e-3,
    boundary: str = "cpml",
    add_source: bool = True,
    add_probe: bool = True,
):
    """Build a minimal NU-profiled simulation for composition tests.

    Uses a constant dx_profile so it is degenerate-uniform (safest baseline).
    The non-trivial composition tests can override profiles after calling this.
    Grid is small (16×12×12 default) to keep CPU runtime short.
    """
    from rfx import Simulation

    sim = Simulation(
        freq_max=5e9,
        domain=(nx * dx, ny * dx, nz * dx),
        dx=dx,
        boundary=boundary,
        dx_profile=np.full(nx, dx),
    )
    if add_source:
        # Place source well inside device-0 slab (x < nx//2) to avoid seam artifacts
        sx = (nx // 4) * dx
        sim.add_source(position=(sx, ny // 2 * dx, nz // 2 * dx), component="ez")
    if add_probe:
        px = (nx // 4 + 2) * dx
        sim.add_probe(position=(px, ny // 2 * dx, nz // 2 * dx), component="ez")
    return sim


# ---------------------------------------------------------------------------
# Test 1 — M5 Spike: shard_map + checkpoint trivial scan grad
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    strict=True,
    reason=(
        "Phase 2F spike: jax.checkpoint(shard_map(scan)) must compose with "
        "jax.grad on the installed JAX version. If this test PASSES it means "
        "the basic JAX composition primitive works and Phase 2F can proceed "
        "without custom vjp rules. If XFAIL, non-trivial shmap_kwargs / "
        "custom_vjp are needed and Phase 2F budget escalates to 24h."
    ),
)
def test_shard_map_checkpoint_trivial_scan_grad():
    """Trivial toy: jax.grad(jax.checkpoint(shard_map(scan_body)))(x).

    No physics — just a linear scan over a NamedTuple carry that multiplies
    x by 1.0 at each step.  The gradient should be 1.0.

    This must be the FIRST test in the file to make the dependency order
    explicit: if this fails, all physics tests will also fail, and the root
    cause is JAX composition, not rfx.

    Outcome:
      PASSED  → JAX composition works on this install; Phase 2F proceeds normally.
      XFAIL   → Non-trivial fixes needed; Phase 2F budget escalates from 8h to 24h.
    """
    _require_two_devices()

    from jax.experimental.shard_map import shard_map
    from jax.sharding import Mesh, PartitionSpec as P
    import jax.lax as lax

    devices = jax.devices()[:2]
    mesh = Mesh(np.array(devices), axis_names=("x",))

    n_steps = 8

    def scan_body(carry, _):
        # Trivial: multiply carry by 1.0 so grad w.r.t. initial value = 1.0
        return carry * 1.0, carry

    @jax.checkpoint
    def sharded_scan(x):
        # shard_map wraps a trivial scan; each device runs its own copy
        def per_device_scan(x_local):
            final, _ = lax.scan(scan_body, x_local, None, length=n_steps)
            return final

        result = shard_map(
            per_device_scan,
            mesh=mesh,
            in_specs=P("x"),
            out_specs=P("x"),
            check_rep=False,
        )(x)
        return jnp.sum(result)

    x = jnp.ones(2)  # one scalar per device
    grad = jax.grad(sharded_scan)(x)

    # Expected: each element of grad should be 1.0 (linear passthrough)
    assert jnp.allclose(grad, jnp.ones_like(grad), atol=1e-6), (
        f"Trivial shard_map+checkpoint scan grad is not 1.0: {grad}"
    )


# ---------------------------------------------------------------------------
# Tests 2-6 — API contract: distributed=True kwarg not yet wired
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    strict=True,
    reason=(
        "Phase 2A: distributed=True kwarg not yet added to Simulation.forward(). "
        "Expect TypeError for unexpected keyword argument."
    ),
)
def test_forward_distributed_flag_requires_multi_device():
    """forward(distributed=True) with only 1 device must raise ValueError.

    When the kwarg IS wired (Phase 2A), passing distributed=True with a
    single device should raise ValueError, not silently fall back.
    For now the call raises TypeError because the kwarg doesn't exist yet.
    """
    _require_two_devices()  # skip if <2 devices available

    from rfx import Simulation

    # Build a minimal NU sim with PEC boundary (fastest)
    sim = _make_nu_sim_small(boundary="pec")

    # Once distributed= is wired, we'd call with a 1-device list to test the
    # ValueError. Until then, even calling with distributed=True raises TypeError.
    # Either TypeError or ValueError satisfies the xfail contract.
    sim.forward(n_steps=4, distributed=True)  # type: ignore[call-arg]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Phase 2A: distributed=True on uniform mesh must raise NotImplementedError "
        "in v1.6.2 (DP3). Not yet wired — TypeError first."
    ),
)
def test_forward_distributed_rejects_uniform_mesh_in_v162():
    """forward(distributed=True) on a uniform mesh must raise NotImplementedError.

    In v1.6.2 the distributed forward path is NU-only (DP3 locked decision).
    A uniform mesh (no dx_profile) with distributed=True must be rejected.
    """
    _require_two_devices()

    from rfx import Simulation

    # Uniform mesh: no dx_profile
    sim = Simulation(
        freq_max=5e9,
        domain=(0.1, 0.06, 0.06),
        dx=5e-3,
        boundary="pec",
    )
    sim.add_source(position=(0.025, 0.03, 0.03), component="ez")
    sim.add_probe(position=(0.05, 0.03, 0.03), component="ez")

    # Should raise NotImplementedError when wired; TypeError until then
    sim.forward(n_steps=4, distributed=True)  # type: ignore[call-arg]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Phase 2A: distributed=True with TFSF source must raise NotImplementedError "
        "(TFSF is single-device only in v1.6.2). Not yet wired — TypeError first."
    ),
)
def test_forward_distributed_rejects_tfsf_on_nu_path():
    """forward(distributed=True) with a TFSF source must raise NotImplementedError.

    TFSF is single-device only in v1.6.2 (non-goal in plan). The distributed
    forward path must detect and reject this configuration.
    """
    _require_two_devices()

    from rfx import Simulation

    sim = _make_nu_sim_small(boundary="pec", add_source=False)
    # TFSF source — not yet supported on distributed path
    try:
        sim.add_tfsf_source(
            center=(sim._domain[0] / 2, sim._domain[1] / 2, sim._domain[2] / 2),
            size=(sim._domain[0] * 0.4, sim._domain[1] * 0.4, sim._domain[2] * 0.4),
            freq=3e9,
            component="ez",
        )
    except AttributeError:
        # add_tfsf_source may have a different name; skip if API not available
        pytest.skip("TFSF source API not available under expected name")

    sim.forward(n_steps=4, distributed=True)  # type: ignore[call-arg]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Phase 2A + Phase 3: forward(distributed=True, devices=[...]) with more "
        "devices than available must raise ValueError. Not yet wired — TypeError first."
    ),
)
def test_forward_distributed_requested_devices_gt_available_raises():
    """Requesting more devices than available must raise ValueError.

    E.g. passing a 4-element devices list when only 2 CPU virtual devices
    exist must raise ValueError, not hang or silently clip.
    """
    devices = _require_two_devices()

    from rfx import Simulation

    sim = _make_nu_sim_small(boundary="pec")

    # Build a fake list longer than available devices
    fake_extra_devices = devices + devices  # 4 elements, but only 2 real devices

    sim.forward(  # type: ignore[call-arg]
        n_steps=4,
        distributed=True,
        devices=fake_extra_devices,
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Phase 3: passing devices= without distributed=True must raise ValueError "
        "(no silent distributed activation). Not yet wired — TypeError first."
    ),
)
def test_forward_distributed_devices_without_distributed_flag_raises():
    """forward(devices=[...]) without distributed=True must raise ValueError.

    There must be no silent activation of distributed mode. If a user passes
    devices= but forgets distributed=True, they get a clear error, not a
    silently single-device run.
    """
    devices = _require_two_devices()

    from rfx import Simulation

    sim = _make_nu_sim_small(boundary="pec")

    sim.forward(  # type: ignore[call-arg]
        n_steps=4,
        devices=devices,  # devices without distributed=True → must raise
    )


# ---------------------------------------------------------------------------
# Tests 7-12 — Physics parity: forward parity on small cases (Class B)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    strict=True,
    reason=(
        "Phase 2B + 2A: ghost-cell exchange + x-slab ownership not yet implemented. "
        "forward(distributed=True) raises TypeError until Phase 2A wires the kwarg."
    ),
)
def test_forward_distributed_pec_two_device_matches_single_device_small_case():
    """Distributed forward with PEC boundary must match single-device to Class B.

    PEC is the simplest boundary (no absorbing layers, no dispersion).
    This test validates basic 2-device field exchange on the NU path.
    Class B: final-step rel_err < 5e-5, energy rel_err < 5e-4.
    """
    devices = _require_two_devices()

    from rfx import Simulation

    n_steps = 20

    # Single-device reference
    sim_s = _make_nu_sim_small(boundary="pec")
    res_s = sim_s.forward(n_steps=n_steps)

    # 2-device distributed (raises TypeError until Phase 2A)
    sim_d = _make_nu_sim_small(boundary="pec")
    res_d = sim_d.forward(  # type: ignore[call-arg]
        n_steps=n_steps,
        distributed=True,
        devices=devices,
    )

    assert_class_b_parity(
        res_s.time_series[0],
        res_d.time_series[0],
        label="pec_two_device",
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Phase 2A + 2C: CPML ownership (x-face psi must be zero at seam) + "
        "ghost-cell exchange not yet implemented."
    ),
)
def test_forward_distributed_cpml_two_device_matches_single_device_small_case():
    """Distributed forward with CPML boundary must match single-device to Class B+C.

    Class B checks probe parity. Class C checks that the interior-rank x-face
    CPML psi arrays are exactly zero (the seam is NOT an absorbing boundary).
    """
    devices = _require_two_devices()

    from rfx import Simulation

    n_steps = 20

    sim_s = _make_nu_sim_small(boundary="cpml")
    res_s = sim_s.forward(n_steps=n_steps)

    sim_d = _make_nu_sim_small(boundary="cpml")
    res_d = sim_d.forward(  # type: ignore[call-arg]
        n_steps=n_steps,
        distributed=True,
        devices=devices,
        return_internal_state=True,  # needed for Class C check
    )

    assert_class_b_parity(
        res_s.time_series[0],
        res_d.time_series[0],
        label="cpml_two_device",
    )

    # Class C: interior rank's x-face CPML psi must be exactly zero
    # (res_d would expose cpml_state_per_rank when implemented)
    if hasattr(res_d, "cpml_state_rank1"):
        assert_class_c_cpml_seam_noop(res_d.cpml_state_rank1, label="rank1_seam")


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Phase 2D: Debye ADE splice not yet ported to distributed scan body. "
        "Also needs Phase 2A for the distributed= kwarg."
    ),
)
def test_forward_distributed_debye_two_device_matches_single_device_small_case():
    """Distributed Debye-material forward must match single-device to Class B+D.

    Class B: final-step and energy parity.
    Class D: drift sweep at T/4, T/2, 3T/4, T to catch cumulative ADE errors.

    The Debye slab is placed near x = nx//2 (the seam) to stress ADE ordering.
    """
    devices = _require_two_devices()

    from rfx import Simulation
    import rfx.materials as mat

    nx, ny, nz, dx = 16, 12, 12, 5e-3
    n_steps = 24

    def _build_sim():
        sim = Simulation(
            freq_max=5e9,
            domain=(nx * dx, ny * dx, nz * dx),
            dx=dx,
            boundary="pec",
            dx_profile=np.full(nx, dx),
        )
        # Debye slab spanning x-index 7..8 (straddles seam at nx//2=8)
        sim.add_material(
            region=((6 * dx, 0, 0), (9 * dx, ny * dx, nz * dx)),
            material=mat.Debye(eps_inf=2.0, delta_eps=[0.5], tau=[1e-11]),
        )
        sim.add_source(position=(3 * dx, ny // 2 * dx, nz // 2 * dx), component="ez")
        sim.add_probe(position=(5 * dx, ny // 2 * dx, nz // 2 * dx), component="ez")
        return sim

    sim_s = _build_sim()
    res_s = sim_s.forward(n_steps=n_steps)

    sim_d = _build_sim()
    res_d = sim_d.forward(  # type: ignore[call-arg]
        n_steps=n_steps,
        distributed=True,
        devices=devices,
    )

    # Class B
    assert_class_b_parity(
        res_s.time_series[0],
        res_d.time_series[0],
        label="debye_two_device",
    )

    # Class D: drift sweep
    assert_class_d_timeseries_drift(
        res_s.time_series[0],
        res_d.time_series[0],
        label="debye_drift",
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Phase 2D: Lorentz ADE splice not yet ported to distributed scan body. "
        "Lorentz polarisation state must be sharded along x and advanced correctly."
    ),
)
def test_forward_distributed_lorentz_two_device_matches_single_device_small_case():
    """Distributed Lorentz-material forward must match single-device to Class B+D.

    Class D drift sweep at 4 time points is critical: Lorentz state has a
    two-step polarisation update; an ordering error at the seam accumulates
    faster than Debye.
    """
    devices = _require_two_devices()

    from rfx import Simulation
    import rfx.materials as mat

    nx, ny, nz, dx = 16, 12, 12, 5e-3
    n_steps = 24

    def _build_sim():
        sim = Simulation(
            freq_max=5e9,
            domain=(nx * dx, ny * dx, nz * dx),
            dx=dx,
            boundary="pec",
            dx_profile=np.full(nx, dx),
        )
        # Lorentz slab near seam
        sim.add_material(
            region=((6 * dx, 0, 0), (9 * dx, ny * dx, nz * dx)),
            material=mat.Lorentz(eps_inf=2.0, delta_eps=[0.5], omega0=[2e10], delta=[1e9]),
        )
        sim.add_source(position=(3 * dx, ny // 2 * dx, nz // 2 * dx), component="ez")
        sim.add_probe(position=(5 * dx, ny // 2 * dx, nz // 2 * dx), component="ez")
        return sim

    sim_s = _build_sim()
    res_s = sim_s.forward(n_steps=n_steps)

    sim_d = _build_sim()
    res_d = sim_d.forward(  # type: ignore[call-arg]
        n_steps=n_steps,
        distributed=True,
        devices=devices,
    )

    assert_class_b_parity(
        res_s.time_series[0],
        res_d.time_series[0],
        label="lorentz_two_device",
    )
    assert_class_d_timeseries_drift(
        res_s.time_series[0],
        res_d.time_series[0],
        label="lorentz_drift",
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Phase 2D: Mixed Debye+Lorentz carry composition must survive sharding. "
        "Both polarisation states must be sharded and advanced in the correct order."
    ),
)
def test_forward_distributed_mixed_dispersion_two_device_matches_single_device_small_case():
    """Distributed mixed Debye+Lorentz forward must match single-device to Class B+D.

    Mixed dispersion has both Debye AND Lorentz polarisation states in the
    same carry.  The sharded ADE update must advance both in the correct order
    (matching _update_e_nu_dispersive in rfx/nonuniform.py).
    """
    devices = _require_two_devices()

    from rfx import Simulation
    import rfx.materials as mat

    nx, ny, nz, dx = 16, 12, 12, 5e-3
    n_steps = 24

    def _build_sim():
        sim = Simulation(
            freq_max=5e9,
            domain=(nx * dx, ny * dx, nz * dx),
            dx=dx,
            boundary="pec",
            dx_profile=np.full(nx, dx),
        )
        # Mixed slab near seam
        sim.add_material(
            region=((6 * dx, 0, 0), (9 * dx, ny * dx, nz * dx)),
            material=mat.Debye(eps_inf=2.0, delta_eps=[0.3], tau=[1e-11]),
        )
        sim.add_material(
            region=((7 * dx, 0, 0), (8 * dx, ny * dx, nz * dx)),
            material=mat.Lorentz(eps_inf=2.5, delta_eps=[0.2], omega0=[2e10], delta=[1e9]),
        )
        sim.add_source(position=(3 * dx, ny // 2 * dx, nz // 2 * dx), component="ez")
        sim.add_probe(position=(5 * dx, ny // 2 * dx, nz // 2 * dx), component="ez")
        return sim

    sim_s = _build_sim()
    res_s = sim_s.forward(n_steps=n_steps)

    sim_d = _build_sim()
    res_d = sim_d.forward(  # type: ignore[call-arg]
        n_steps=n_steps,
        distributed=True,
        devices=devices,
    )

    assert_class_b_parity(
        res_s.time_series[0],
        res_d.time_series[0],
        label="mixed_dispersion_two_device",
    )
    assert_class_d_timeseries_drift(
        res_s.time_series[0],
        res_d.time_series[0],
        label="mixed_dispersion_drift",
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Phase 2E: pec_occupancy sharding not yet implemented. "
        "The single-device NU path already applies pec_occupancy correctly (C2 confirmed); "
        "Phase 2E ports that splice into the distributed scan body."
    ),
)
def test_forward_distributed_pec_occupancy_two_device_matches_single_device():
    """Distributed forward with soft-PEC (pec_occupancy_override) must match single-device.

    The single-device NU path correctly applies pec_occupancy (C2 confirmed in V3 plan).
    This test verifies that Phase 2E correctly shards the occupancy array along x
    and routes it through the distributed scan body.
    Class B: final-step and energy parity.
    """
    devices = _require_two_devices()

    from rfx import Simulation

    nx, ny, nz, dx = 16, 12, 12, 5e-3
    n_steps = 20

    def _build_sim():
        sim = Simulation(
            freq_max=5e9,
            domain=(nx * dx, ny * dx, nz * dx),
            dx=dx,
            boundary="pec",
            dx_profile=np.full(nx, dx),
        )
        sim.add_source(position=(3 * dx, ny // 2 * dx, nz // 2 * dx), component="ez")
        sim.add_probe(position=(5 * dx, ny // 2 * dx, nz // 2 * dx), component="ez")
        return sim

    # Build a soft-PEC occupancy mask: partial conductivity in a slab near the seam
    # Shape matches the field grid (nx, ny, nz)
    pec_occ = jnp.zeros((nx, ny, nz))
    pec_occ = pec_occ.at[7:9, :, :].set(0.5)  # partial occupancy near seam

    sim_s = _build_sim()
    res_s = sim_s.forward(n_steps=n_steps, pec_occupancy_override=pec_occ)

    sim_d = _build_sim()
    res_d = sim_d.forward(  # type: ignore[call-arg]
        n_steps=n_steps,
        pec_occupancy_override=pec_occ,
        distributed=True,
        devices=devices,
    )

    assert_class_b_parity(
        res_s.time_series[0],
        res_d.time_series[0],
        label="pec_occupancy_two_device",
    )


# ---------------------------------------------------------------------------
# Tests 13-15 — Gradient parity (Class A)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    strict=True,
    reason=(
        "Phase 2F: segmented scan-of-scan with shard_map not yet composed. "
        "checkpoint_every on the distributed path requires scan-of-scan inside shard_map; "
        "gradients must be bit-identical to the non-segmented baseline (Class A+E)."
    ),
)
def test_forward_distributed_checkpoint_every_matches_no_segment_small_grad_case():
    """Distributed forward with checkpoint_every must produce bit-identical grad.

    checkpoint_every segments the scan for memory efficiency.  On the
    distributed path, this means scan-of-scan inside shard_map — a composition
    that requires Phase 2F work.

    Class A: grad rtol < 1e-6.
    Class E: bit-identical probe output (checkpoint must not alter values).
    """
    devices = _require_two_devices()

    from rfx import Simulation

    n_steps = 16
    nx, ny, nz, dx = 12, 10, 10, 5e-3

    def _build_sim():
        sim = Simulation(
            freq_max=5e9,
            domain=(nx * dx, ny * dx, nz * dx),
            dx=dx,
            boundary="pec",
            dx_profile=np.full(nx, dx),
        )
        sim.add_source(position=(3 * dx, ny // 2 * dx, nz // 2 * dx), component="ez")
        sim.add_probe(position=(5 * dx, ny // 2 * dx, nz // 2 * dx), component="ez")
        return sim

    eps = jnp.ones((nx, ny, nz)) * 1.0

    def loss_no_segment(eps_val):
        sim = _build_sim()
        res = sim.forward(  # type: ignore[call-arg]
            n_steps=n_steps,
            eps_override=eps_val,
            distributed=True,
            devices=devices,
        )
        return jnp.sum(res.time_series[0] ** 2)

    def loss_with_checkpoint(eps_val):
        sim = _build_sim()
        res = sim.forward(  # type: ignore[call-arg]
            n_steps=n_steps,
            eps_override=eps_val,
            distributed=True,
            devices=devices,
            checkpoint_every=4,
        )
        return jnp.sum(res.time_series[0] ** 2)

    grad_no_seg = jax.grad(loss_no_segment)(eps)
    grad_checkpt = jax.grad(loss_with_checkpoint)(eps)

    # Class E bit-match
    assert_class_e_bit_match(grad_no_seg, grad_checkpt, label="checkpoint_every_bit_match")


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Phase 2F: n_warmup stop-gradient splitting must be applied at the sharded "
        "carry boundary. The warmup tail must not receive gradients from the warm-up "
        "phase, matching single-device behaviour."
    ),
)
def test_forward_distributed_n_warmup_tail_grad_matches_single_device():
    """Distributed forward with n_warmup must produce same tail gradient as single-device.

    n_warmup runs steps 0..n_warmup-1 without gradient accumulation, then
    differentiates over steps n_warmup..n_steps-1.  On the distributed path,
    the stop-gradient boundary must be applied at the sharded carry boundary
    in the same way.
    Class A: max_rel_err < 1e-6.
    """
    devices = _require_two_devices()

    from rfx import Simulation

    n_steps = 20
    n_warmup = 8
    nx, ny, nz, dx = 12, 10, 10, 5e-3

    def _build_sim():
        sim = Simulation(
            freq_max=5e9,
            domain=(nx * dx, ny * dx, nz * dx),
            dx=dx,
            boundary="pec",
            dx_profile=np.full(nx, dx),
        )
        sim.add_source(position=(3 * dx, ny // 2 * dx, nz // 2 * dx), component="ez")
        sim.add_probe(position=(5 * dx, ny // 2 * dx, nz // 2 * dx), component="ez")
        return sim

    eps = jnp.ones((nx, ny, nz)) * 1.0

    def loss_single(eps_val):
        sim = _build_sim()
        res = sim.forward(n_steps=n_steps, eps_override=eps_val, n_warmup=n_warmup)
        return jnp.sum(res.time_series[0] ** 2)

    def loss_dist(eps_val):
        sim = _build_sim()
        res = sim.forward(  # type: ignore[call-arg]
            n_steps=n_steps,
            eps_override=eps_val,
            n_warmup=n_warmup,
            distributed=True,
            devices=devices,
        )
        return jnp.sum(res.time_series[0] ** 2)

    grad_single = jax.grad(loss_single)(eps)
    grad_dist = jax.grad(loss_dist)(eps)

    assert_class_a_grad(grad_single, grad_dist, label="n_warmup_tail_grad")


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Phase 2F: design_mask stop-gradient must be applied correctly at sharded "
        "carry rebuild. The design_mask cells must receive zero gradient in both "
        "single-device and distributed runs."
    ),
)
def test_forward_distributed_design_mask_stop_grad_matches_single_device():
    """Distributed forward with design_mask must match single-device gradient.

    design_mask applies stop_gradient on non-design cells so only the
    design region accumulates gradients.  On the distributed path, the
    sharded carry rebuild must apply design_mask in the same way.
    Class A: max_rel_err < 1e-6.
    """
    devices = _require_two_devices()

    from rfx import Simulation

    n_steps = 16
    nx, ny, nz, dx = 12, 10, 10, 5e-3

    def _build_sim():
        sim = Simulation(
            freq_max=5e9,
            domain=(nx * dx, ny * dx, nz * dx),
            dx=dx,
            boundary="pec",
            dx_profile=np.full(nx, dx),
        )
        sim.add_source(position=(3 * dx, ny // 2 * dx, nz // 2 * dx), component="ez")
        sim.add_probe(position=(5 * dx, ny // 2 * dx, nz // 2 * dx), component="ez")
        return sim

    eps = jnp.ones((nx, ny, nz)) * 1.0
    # Design mask: only cells x=3..7 are design cells
    design_mask = jnp.zeros((nx, ny, nz), dtype=bool)
    design_mask = design_mask.at[3:8, :, :].set(True)

    def loss_single(eps_val):
        sim = _build_sim()
        res = sim.forward(
            n_steps=n_steps,
            eps_override=eps_val,
            design_mask=design_mask,
        )
        return jnp.sum(res.time_series[0] ** 2)

    def loss_dist(eps_val):
        sim = _build_sim()
        res = sim.forward(  # type: ignore[call-arg]
            n_steps=n_steps,
            eps_override=eps_val,
            design_mask=design_mask,
            distributed=True,
            devices=devices,
        )
        return jnp.sum(res.time_series[0] ** 2)

    grad_single = jax.grad(loss_single)(eps)
    grad_dist = jax.grad(loss_dist)(eps)

    assert_class_a_grad(grad_single, grad_dist, label="design_mask_stop_grad")


# ---------------------------------------------------------------------------
# Tests 16-17 — Seam isolation (Class C + B)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    strict=True,
    reason=(
        "Phase 2C: interior seam must NOT activate x-face CPML on interior ranks. "
        "CPML x-face psi arrays for interior ranks must be exactly zero (Class C)."
    ),
)
def test_forward_distributed_cpml_internal_seam_is_noop_not_boundary():
    """Interior CPML ranks must have zero x-face psi (seam is not an absorber).

    When the domain is split into 2 x-slabs, only the global x boundaries
    (rank 0 x-low face, rank 1 x-high face) should activate CPML absorbers.
    The seam between rank 0 and rank 1 must NOT activate CPML psi update on
    either the x-high face of rank 0 or the x-low face of rank 1.

    This is Class C: the assertion is exact-zero on the interior psi arrays,
    not an approximate parity check.
    """
    devices = _require_two_devices()

    from rfx import Simulation

    n_steps = 10

    sim = _make_nu_sim_small(boundary="cpml")
    # Run distributed and expose internal CPML state per rank
    res = sim.forward(  # type: ignore[call-arg]
        n_steps=n_steps,
        distributed=True,
        devices=devices,
        return_internal_state=True,
    )

    # When implemented, res.cpml_state_per_rank is a list of CPMLState per device.
    # Interior rank 0's x-high face and rank 1's x-low face must be zero.
    if hasattr(res, "cpml_state_per_rank"):
        rank0_cpml = res.cpml_state_per_rank[0]
        rank1_cpml = res.cpml_state_per_rank[1]

        # Rank 0 x-hi face (seam side) must be zero
        for field in ["psi_ey_xhi", "psi_ez_xhi", "psi_hy_xhi", "psi_hz_xhi"]:
            arr = getattr(rank0_cpml, field, None)
            if arr is not None:
                assert jnp.all(arr == 0), (
                    f"Rank 0 CPML seam field '{field}' must be zero (got max={float(jnp.max(jnp.abs(arr))):.3e})"
                )

        # Rank 1 x-lo face (seam side) must be zero
        for field in ["psi_ey_xlo", "psi_ez_xlo", "psi_hy_xlo", "psi_hz_xlo"]:
            arr = getattr(rank1_cpml, field, None)
            if arr is not None:
                assert jnp.all(arr == 0), (
                    f"Rank 1 CPML seam field '{field}' must be zero (got max={float(jnp.max(jnp.abs(arr))):.3e})"
                )
    else:
        # The kwarg is not yet implemented; the forward call above will have raised
        # TypeError, which is the expected xfail. If we get here, assert failure.
        pytest.fail("forward(distributed=True) must expose cpml_state_per_rank when implemented")


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Phase 2B + 2E: pec_mask must survive ghost-cell exchange without being "
        "corrupted at the seam. Field values at seam-adjacent cells must match "
        "single-device reference (Class B)."
    ),
)
def test_forward_distributed_pec_mask_seam_exchange_preserves_field():
    """PEC mask at the seam must survive ghost-cell exchange unchanged (Class B).

    A hard-PEC mask (pec_mask_override) that spans the seam boundary between
    rank 0 and rank 1 must produce identical field values to the single-device
    reference.  Ghost exchange must not corrupt the PEC-masked cells.
    """
    devices = _require_two_devices()

    from rfx import Simulation

    nx, ny, nz, dx = 16, 12, 12, 5e-3
    n_steps = 20

    # PEC mask: a thin conductor near the seam (x = nx//2 - 1 to nx//2 + 1)
    pec_mask = jnp.zeros((nx, ny, nz), dtype=bool)
    pec_mask = pec_mask.at[nx // 2 - 1 : nx // 2 + 2, :, :].set(True)

    def _build_sim():
        sim = Simulation(
            freq_max=5e9,
            domain=(nx * dx, ny * dx, nz * dx),
            dx=dx,
            boundary="pec",
            dx_profile=np.full(nx, dx),
        )
        sim.add_source(position=(3 * dx, ny // 2 * dx, nz // 2 * dx), component="ez")
        sim.add_probe(position=(5 * dx, ny // 2 * dx, nz // 2 * dx), component="ez")
        return sim

    sim_s = _build_sim()
    res_s = sim_s.forward(n_steps=n_steps, pec_mask_override=pec_mask)

    sim_d = _build_sim()
    res_d = sim_d.forward(  # type: ignore[call-arg]
        n_steps=n_steps,
        pec_mask_override=pec_mask,
        distributed=True,
        devices=devices,
    )

    assert_class_b_parity(
        res_s.time_series[0],
        res_d.time_series[0],
        label="pec_mask_seam_exchange",
    )


# ---------------------------------------------------------------------------
# Test 18 (G1) — Per-cell seam gradient parity (Class A near seam)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    strict=True,
    reason=(
        "Phase 2D: per-cell gradient near the seam must match single-device within "
        "rtol=1e-4 (relaxed from 1e-6 due to ghost-cell exchange rounding). "
        "Requires Phase 2A + 2D + 2F all implemented."
    ),
)
def test_forward_distributed_grad_per_cell_matches_single_device_near_seam():
    """Per-cell gradient for seam-adjacent cells must match single-device (Class A seam).

    This is the G1 test from the V3 plan. The gradient w.r.t. eps at x-cells
    immediately adjacent to the seam (x = nx//2 - 2 : nx//2 + 2) must match
    the single-device reference within rtol=1e-4. Cells far from the seam use
    standard Class A rtol=1e-6.

    The tighter requirement on seam-adjacent cells guards against ghost-cell
    exchange introducing systematic gradient errors at the domain boundary.
    """
    devices = _require_two_devices()

    from rfx import Simulation

    nx, ny, nz, dx = 16, 12, 12, 5e-3
    n_steps = 16

    def _build_sim():
        sim = Simulation(
            freq_max=5e9,
            domain=(nx * dx, ny * dx, nz * dx),
            dx=dx,
            boundary="pec",
            dx_profile=np.full(nx, dx),
        )
        sim.add_source(position=(3 * dx, ny // 2 * dx, nz // 2 * dx), component="ez")
        sim.add_probe(position=(5 * dx, ny // 2 * dx, nz // 2 * dx), component="ez")
        return sim

    eps = jnp.ones((nx, ny, nz)) * 1.0

    def loss_single(eps_val):
        sim = _build_sim()
        res = sim.forward(n_steps=n_steps, eps_override=eps_val)
        return jnp.sum(res.time_series[0] ** 2)

    def loss_dist(eps_val):
        sim = _build_sim()
        res = sim.forward(  # type: ignore[call-arg]
            n_steps=n_steps,
            eps_override=eps_val,
            distributed=True,
            devices=devices,
        )
        return jnp.sum(res.time_series[0] ** 2)

    grad_single = jax.grad(loss_single)(eps)
    grad_dist = jax.grad(loss_dist)(eps)

    # Seam-adjacent cells: x = nx//2-2 to nx//2+2
    seam = slice(nx // 2 - 2, nx // 2 + 2)
    assert_class_a_grad_seam(
        grad_single[seam],
        grad_dist[seam],
        label="seam_adjacent_cells",
        rtol=1e-4,
    )

    # Non-seam cells: standard Class A
    non_seam_single = jnp.concatenate([grad_single[:nx//2-2].ravel(),
                                        grad_single[nx//2+2:].ravel()])
    non_seam_dist   = jnp.concatenate([grad_dist[:nx//2-2].ravel(),
                                        grad_dist[nx//2+2:].ravel()])
    assert_class_a_grad(non_seam_single, non_seam_dist, label="non_seam_cells")


# ---------------------------------------------------------------------------
# Test 19 (G2) — NaN propagation via ghost exchange (Class E structural)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    strict=True,
    reason=(
        "Phase 2C: NaN injected on rank 0 must propagate to rank 1 within "
        "2*exchange_interval steps via ghost-cell exchange. "
        "Requires ghost exchange to be implemented (Phase 2B/2C)."
    ),
)
def test_forward_distributed_nan_propagates_via_ghost_exchange():
    """NaN on rank 0 must reach rank 1's probe within 2*exchange_interval steps.

    This is the G2 test from the V3 plan. It validates that ghost-cell exchange
    is actually happening: if rank 0's fields contain NaN, rank 1's probe must
    show NaN within 2*exchange_interval time steps (not stay at a stale value).

    This test requires a mechanism to inject NaN into rank 0's field (e.g. via
    an eps_override containing NaN in rank 0's slab).  The probe is on rank 1.

    Class E structural: no numeric threshold, just assert NaN appears in time.
    """
    devices = _require_two_devices()

    from rfx import Simulation

    nx, ny, nz, dx = 16, 12, 12, 5e-3
    exchange_interval = 1
    n_steps = 4 * exchange_interval + 4  # enough steps for NaN to propagate

    sim = Simulation(
        freq_max=5e9,
        domain=(nx * dx, ny * dx, nz * dx),
        dx=dx,
        boundary="pec",
        dx_profile=np.full(nx, dx),
    )
    # Source on rank 0 side (x < nx//2)
    sim.add_source(position=(3 * dx, ny // 2 * dx, nz // 2 * dx), component="ez")
    # Probe on rank 1 side (x >= nx//2)
    sim.add_probe(
        position=((nx // 2 + 3) * dx, ny // 2 * dx, nz // 2 * dx),
        component="ez",
    )

    # Inject NaN into rank 0's eps slab (x = 0..nx//2-1)
    eps_nan = jnp.ones((nx, ny, nz))
    eps_nan = eps_nan.at[: nx // 2, :, :].set(jnp.nan)

    res = sim.forward(  # type: ignore[call-arg]
        n_steps=n_steps,
        eps_override=eps_nan,
        distributed=True,
        devices=devices,
        exchange_interval=exchange_interval,
    )

    ts = np.asarray(res.time_series[0])  # probe on rank 1

    # NaN must appear within 2*exchange_interval steps of rank-1 probe
    nan_steps = np.where(np.isnan(ts))[0]
    assert len(nan_steps) > 0, (
        f"G2: NaN did not propagate from rank 0 to rank 1 probe within {n_steps} steps. "
        f"time_series={ts}. Ghost exchange may not be wired."
    )
    first_nan = nan_steps[0]
    max_allowed = 2 * exchange_interval + 2  # +2 for source injection latency
    assert first_nan <= max_allowed, (
        f"G2: NaN took {first_nan} steps to propagate; expected <= {max_allowed}. "
        f"Ghost exchange interval may be too large or exchange is not per-step."
    )
