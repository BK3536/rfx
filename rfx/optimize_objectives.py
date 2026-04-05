"""Pre-built objective functions for ``rfx.optimize.optimize()``.

Each factory returns a callable ``objective(result) -> scalar`` that
is compatible with :func:`rfx.optimize.optimize` and differentiable
through JAX.

**S-parameter vs time-domain objectives**

The frequency-domain objectives (``minimize_s11``, ``maximize_s21``,
``target_impedance``, ``maximize_bandwidth``) require ``result.s_params``
to be populated.  This happens automatically when running a simulation
via ``Simulation.run()`` with ports, but the lightweight forward pass
used inside ``optimize()`` and ``topology_optimize()`` does **not**
compute S-parameters (it would break JAX traceability).

For gradient-based optimization, use the **time-domain proxy**
objectives instead:

- ``minimize_reflected_energy`` -- proxy for minimizing S11
- ``maximize_transmitted_energy`` -- proxy for maximizing S21

These operate directly on ``result.time_series`` and are fully
JAX-differentiable through the FDTD scan loop.

Typical usage
-------------
>>> from rfx import Simulation, optimize, DesignRegion
>>> from rfx.optimize_objectives import minimize_reflected_energy
>>> sim = Simulation(freq_max=10e9, domain=(0.05, 0.05, 0.025))
>>> # ... add geometry, ports, probes ...
>>> obj = minimize_reflected_energy(port_probe_idx=0)
>>> result = optimize(sim, region, obj, n_iters=50)
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _db_to_linear(db: float) -> float:
    """Convert dB power to linear scale: 10^(dB/10)."""
    return 10.0 ** (db / 10.0)


def _find_freq_indices(result_freqs: jnp.ndarray, target_freqs: jnp.ndarray) -> jnp.ndarray:
    """Find nearest indices in *result_freqs* for each entry in *target_freqs*.

    Uses pure JAX operations so the index lookup is differentiable-friendly
    (the indices themselves are integer, but the downstream gather is
    stop-gradient safe because we only use them for indexing).
    """
    # (n_target, n_result) absolute differences
    diffs = jnp.abs(target_freqs[:, None] - result_freqs[None, :])
    return jnp.argmin(diffs, axis=1)


# ---------------------------------------------------------------------------
# Public objective factories
# ---------------------------------------------------------------------------

def minimize_s11(
    freqs: jnp.ndarray | np.ndarray,
    target_db: float = -10.0,
) -> Callable:
    """Minimize |S11|² over specified frequencies.

    Returns a callable ``objective(result) -> scalar`` that computes the
    mean |S11|² across the nearest frequency bins.  If the mean S11 is
    already below *target_db* the loss is clamped to zero.

    Parameters
    ----------
    freqs : array-like, shape (n,)
        Target frequencies (Hz) over which to minimize S11.
    target_db : float
        Threshold in dB.  When mean |S11|² is below this, loss = 0.

    Returns
    -------
    callable(Result) -> scalar (JAX-differentiable)
    """
    target_freqs = jnp.asarray(freqs, dtype=jnp.float32)
    threshold_linear = _db_to_linear(target_db)  # e.g. -10 dB -> 0.1

    def objective(result) -> jnp.ndarray:
        s_params = result.s_params  # (n_ports, n_ports, n_freqs) complex
        if s_params is None:
            raise ValueError(
                "minimize_s11 requires result.s_params but got None. "
                "The optimize() / topology_optimize() forward pass does not "
                "compute S-parameters. Use minimize_reflected_energy() as a "
                "time-domain proxy objective for gradient-based optimization."
            )
        result_freqs = result.freqs  # (n_freqs,)

        s11 = s_params[0, 0, :]  # (n_freqs,) complex
        mag_sq = jnp.abs(s11) ** 2  # |S11|² linear

        indices = _find_freq_indices(
            jnp.asarray(result_freqs, dtype=jnp.float32), target_freqs,
        )
        selected = mag_sq[indices]  # (n_target,)
        mean_mag_sq = jnp.mean(selected)

        # Clamp: if already below target, loss = 0
        return jnp.maximum(mean_mag_sq - threshold_linear, 0.0)

    return objective


def maximize_s21(
    freqs: jnp.ndarray | np.ndarray,
) -> Callable:
    """Maximize |S21|² (transmission) over specified frequencies.

    Returns ``-mean|S21|²`` so that *minimizing* this objective
    maximizes transmission.

    Parameters
    ----------
    freqs : array-like, shape (n,)
        Target frequencies (Hz).

    Returns
    -------
    callable(Result) -> scalar (JAX-differentiable)
    """
    target_freqs = jnp.asarray(freqs, dtype=jnp.float32)

    def objective(result) -> jnp.ndarray:
        s_params = result.s_params  # (n_ports, n_ports, n_freqs) complex
        if s_params is None:
            raise ValueError(
                "maximize_s21 requires result.s_params but got None. "
                "The optimize() / topology_optimize() forward pass does not "
                "compute S-parameters. Use maximize_transmitted_energy() as a "
                "time-domain proxy objective for gradient-based optimization."
            )
        result_freqs = result.freqs

        s21 = s_params[1, 0, :]  # (n_freqs,) complex
        mag_sq = jnp.abs(s21) ** 2

        indices = _find_freq_indices(
            jnp.asarray(result_freqs, dtype=jnp.float32), target_freqs,
        )
        selected = mag_sq[indices]
        return -jnp.mean(selected)

    return objective


def target_impedance(
    freq: float,
    z_target: float = 50.0,
) -> Callable:
    """Minimize |Z_in − z_target|² at a specific frequency.

    Computes input impedance from S11: ``Z_in = Z0 * (1 + S11) / (1 − S11)``
    where Z0 = 50 Ω (standard reference impedance).

    Parameters
    ----------
    freq : float
        Target frequency (Hz).
    z_target : float
        Desired input impedance (Ω).

    Returns
    -------
    callable(Result) -> scalar (JAX-differentiable)
    """
    freq_arr = jnp.asarray([freq], dtype=jnp.float32)
    z0 = 50.0  # reference impedance for S-parameter normalization

    def objective(result) -> jnp.ndarray:
        s_params = result.s_params
        if s_params is None:
            raise ValueError(
                "target_impedance requires result.s_params but got None. "
                "The optimize() / topology_optimize() forward pass does not "
                "compute S-parameters. Use a time-domain proxy objective "
                "for gradient-based optimization."
            )
        result_freqs = result.freqs

        s11 = s_params[0, 0, :]  # (n_freqs,) complex

        idx = _find_freq_indices(
            jnp.asarray(result_freqs, dtype=jnp.float32), freq_arr,
        )
        s11_at_f = s11[idx[0]]  # scalar complex

        # Z_in = Z0 * (1 + S11) / (1 - S11), with safety clamp
        denom = 1.0 - s11_at_f
        # Avoid division by zero: add tiny imaginary part
        denom = jnp.where(jnp.abs(denom) < 1e-12, 1e-12 + 0j, denom)
        z_in = z0 * (1.0 + s11_at_f) / denom

        return jnp.abs(z_in - z_target) ** 2

    return objective


def maximize_bandwidth(
    f_center: float,
    f_bw: float,
    s11_threshold: float = -10.0,
) -> Callable:
    """Maximize the bandwidth where |S11| < threshold.

    Evaluates S11 over a frequency band ``[f_center − f_bw/2, f_center + f_bw/2]``
    and returns a soft loss that is lower when more frequency bins satisfy
    ``|S11|_dB < s11_threshold``.

    The loss is computed as the mean of ``max(|S11|²_dB − threshold, 0)``
    across the band, so it drives the optimizer to push all bins below
    threshold simultaneously, effectively maximizing matched bandwidth.

    Parameters
    ----------
    f_center : float
        Center frequency (Hz).
    f_bw : float
        Bandwidth span (Hz).
    s11_threshold : float
        Threshold in dB (default −10 dB).

    Returns
    -------
    callable(Result) -> scalar (JAX-differentiable)
    """
    f_lo = f_center - f_bw / 2.0
    f_hi = f_center + f_bw / 2.0

    def objective(result) -> jnp.ndarray:
        s_params = result.s_params
        if s_params is None:
            raise ValueError(
                "maximize_bandwidth requires result.s_params but got None. "
                "The optimize() / topology_optimize() forward pass does not "
                "compute S-parameters. Use minimize_reflected_energy() as a "
                "time-domain proxy objective for gradient-based optimization."
            )
        result_freqs = jnp.asarray(result.freqs, dtype=jnp.float32)

        s11 = s_params[0, 0, :]  # (n_freqs,) complex
        mag_sq = jnp.abs(s11) ** 2

        # Select frequency bins within the target band
        mask = (result_freqs >= f_lo) & (result_freqs <= f_hi)
        # Convert to dB: 10*log10(|S11|²), with floor to avoid log(0)
        mag_sq_safe = jnp.maximum(mag_sq, 1e-30)
        s11_db = 10.0 * jnp.log10(mag_sq_safe)

        # Hinge loss: penalize bins above threshold
        excess = jnp.maximum(s11_db - s11_threshold, 0.0)

        # Weighted mean over in-band frequencies (out-of-band contribute 0)
        n_in_band = jnp.maximum(jnp.sum(mask), 1.0)
        return jnp.sum(excess * mask) / n_in_band

    return objective


def maximize_directivity(
    theta_target: float,
    phi_target: float,
) -> Callable:
    """Maximize far-field power in a target direction.

    Computes the total radiated power density
    ``|E_θ|² + |E_φ|²`` at the target (θ, φ) direction, averaged
    over all frequencies in the far-field result, and returns
    its negation (so minimizing drives power upward).

    The ``result`` must carry ``ntff_data``, ``ntff_box``, and ``grid``
    (i.e., the simulation must include an NTFF box and preserve the
    post-processing context needed to evaluate it).

    Parameters
    ----------
    theta_target : float
        Polar angle in radians [0, π].
    phi_target : float
        Azimuthal angle in radians [0, 2π].

    Returns
    -------
    callable(Result) -> scalar (JAX-differentiable)

    Notes
    -----
    This objective uses NumPy-based far-field computation internally.
    It is differentiable through the NTFF DFT accumulation (which is
    JAX-traced), but the angular projection itself is not
    re-differentiated.  For gradient-based optimization, the NTFF
    accumulation provides the necessary gradient path.
    """
    theta_arr = np.array([theta_target])
    phi_arr = np.array([phi_target])

    def objective(result) -> jnp.ndarray:
        from rfx.farfield import compute_far_field

        ntff_data = result.ntff_data
        ntff_box = result.ntff_box

        if ntff_data is None or ntff_box is None:
            raise ValueError(
                "maximize_directivity requires a simulation with an NTFF box. "
                "Use sim.add_ntff_box(...) before running."
            )

        grid = getattr(result, "grid", None)
        if grid is None:
            grid = getattr(getattr(result, "state", None), "grid", None)
        if grid is None:
            raise ValueError(
                "maximize_directivity requires result.grid so the far-field "
                "can be evaluated from NTFF data."
            )

        ff = compute_far_field(ntff_data, ntff_box, grid, theta_arr, phi_arr)

        # Power density at target direction, summed over frequencies
        # E_theta, E_phi: (n_freqs, 1, 1)
        power = jnp.abs(ff.E_theta[:, 0, 0]) ** 2 + jnp.abs(ff.E_phi[:, 0, 0]) ** 2
        mean_power = jnp.mean(power)

        # Negate: minimizing this = maximizing directivity
        return -mean_power

    return objective


# ---------------------------------------------------------------------------
# Time-domain proxy objectives (for use with optimize / topology_optimize)
# ---------------------------------------------------------------------------

def minimize_reflected_energy(
    port_probe_idx: int = 0,
    *,
    late_fraction: float = 0.5,
) -> Callable:
    """Time-domain S11 proxy: minimize late-time reflected energy at port.

    Computes the ratio of energy in the second half of the probe time
    series (dominated by reflections) to energy in the first half
    (dominated by the incident pulse).  Minimizing this ratio drives
    the optimizer toward better impedance matching.

    This objective works with ``optimize()`` and ``topology_optimize()``
    because it uses only ``result.time_series`` (no S-parameters needed).

    Parameters
    ----------
    port_probe_idx : int
        Index into ``result.time_series`` columns identifying the probe
        co-located with the excitation port (default 0).
    late_fraction : float
        Fraction of the time series considered "late" (default 0.5).
        A value of 0.5 means the second half is treated as reflection.

    Returns
    -------
    callable(Result) -> scalar (JAX-differentiable)
    """
    def objective(result) -> jnp.ndarray:
        ts = result.time_series[:, port_probe_idx]
        n = ts.shape[0]
        split = int(n * (1.0 - late_fraction))
        early_energy = jnp.sum(ts[:split] ** 2) + 1e-30
        late_energy = jnp.sum(ts[split:] ** 2)
        return late_energy / early_energy

    return objective


def maximize_transmitted_energy(
    output_probe_idx: int = -1,
) -> Callable:
    """Time-domain S21 proxy: maximize energy at an output probe.

    Returns the negated total squared energy at the output probe, so
    that *minimizing* this objective maximizes transmission.

    This objective works with ``optimize()`` and ``topology_optimize()``
    because it uses only ``result.time_series`` (no S-parameters needed).

    Parameters
    ----------
    output_probe_idx : int
        Index into ``result.time_series`` columns for the output probe.
        Default -1 (last probe).

    Returns
    -------
    callable(Result) -> scalar (JAX-differentiable)
    """
    def objective(result) -> jnp.ndarray:
        ts = result.time_series[:, output_probe_idx]
        return -jnp.sum(ts ** 2)

    return objective


# ---------------------------------------------------------------------------
# Near-field probe-array beam steering (no NTFF required)
# ---------------------------------------------------------------------------

def steer_probe_array(
    target_probe_idx: int,
    suppress_probe_idx: int = 0,
    *,
    late_fraction: float = 0.5,
) -> Callable:
    """Steer radiation toward *target_probe_idx* and away from *suppress_probe_idx*.

    Uses a near-field probe array as a differentiable surrogate for
    far-field beam steering.  Place probes at different spatial positions
    around the antenna (e.g., above-left and above-right), then this
    objective maximizes the power ratio between target and suppressed
    probe.

    This avoids NTFF DFT entirely and works reliably in float32.

    Setup example::

        # Probes at different angles in the near field
        sim.add_probe((x_center - 0.02, y_center, z_above), "ez")  # probe 0: left
        sim.add_probe((x_center + 0.02, y_center, z_above), "ez")  # probe 1: right

        # Steer toward probe 1 (right), suppress probe 0 (left)
        obj = steer_probe_array(target_probe_idx=1, suppress_probe_idx=0)

    Parameters
    ----------
    target_probe_idx : int
        Column index in ``result.time_series`` to maximize.
    suppress_probe_idx : int
        Column index to suppress (default 0).
    late_fraction : float
        Fraction of time series to use (late portion, after source decays).

    Returns
    -------
    callable(Result) -> scalar (JAX-differentiable)
    """
    def objective(result) -> jnp.ndarray:
        ts = result.time_series
        n = ts.shape[0]
        start = int(n * (1.0 - late_fraction))
        target_energy = jnp.sum(ts[start:, target_probe_idx] ** 2)
        suppress_energy = jnp.sum(ts[start:, suppress_probe_idx] ** 2)
        # Maximize ratio: minimize -(target / (suppress + eps))
        return -(target_energy / (suppress_energy + 1e-12))

    return objective
