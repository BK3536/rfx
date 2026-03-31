"""Shared DFT utilities used by probes and waveguide ports."""

from __future__ import annotations

import jax.numpy as jnp


def dft_window_weight(step: int, total_steps: int, window: str, alpha: float) -> jnp.ndarray:
    """Streaming DFT window weight for a given timestep.

    Parameters
    ----------
    step : int
        Current timestep index.
    total_steps : int
        Total number of timesteps in the simulation.
    window : str
        Window type: ``"rect"``, ``"hann"``, or ``"tukey"``.
    alpha : float
        Shape parameter for the Tukey window (ignored for others).

    Returns
    -------
    jnp.ndarray
        Scalar weight in [0, 1].
    """
    if total_steps <= 1 or window == "rect":
        return jnp.asarray(1.0, dtype=jnp.float32)

    n = jnp.asarray(step, dtype=jnp.float32)
    N = jnp.asarray(total_steps - 1, dtype=jnp.float32)
    x = n / jnp.maximum(N, 1.0)

    if window == "hann":
        return 0.5 * (1.0 - jnp.cos(2.0 * jnp.pi * x))

    if window == "tukey":
        a = float(alpha)
        if a <= 0.0:
            return jnp.asarray(1.0, dtype=jnp.float32)
        if a >= 1.0:
            return 0.5 * (1.0 - jnp.cos(2.0 * jnp.pi * x))
        left = 0.5 * (1.0 + jnp.cos(jnp.pi * (2.0 * x / a - 1.0)))
        right = 0.5 * (1.0 + jnp.cos(jnp.pi * (2.0 * x / a - 2.0 / a + 1.0)))
        return jnp.where(
            x < a / 2.0,
            left,
            jnp.where(x <= 1.0 - a / 2.0, 1.0, right),
        )

    raise ValueError(f"dft_window must be 'rect', 'hann', or 'tukey', got {window!r}")
