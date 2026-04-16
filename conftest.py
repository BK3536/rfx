"""
Repository-root conftest.py for rfx.

XLA_FLAGS isolation
-------------------
This file sets XLA_FLAGS *before* any ``import jax`` statement is executed.
pytest loads conftest.py at collection time, before test modules are imported,
so this is the correct place to inject the flag.

Without this, the distributed-NU composition tests in
``tests/test_distributed_nu_composition.py`` fail in a subtle way: if jax
was already imported in another module before XLA_FLAGS was set, JAX will
have already enumerated physical devices (typically 1 CPU) and the flag has
no effect.  When that happens, ``len(jax.devices()) < 2`` causes those tests
to skip (by design), but the absence of 2 virtual devices is the sign that
the env-var injection arrived too late.  Setting the var here, at the very top
of the first conftest that is loaded, ensures the flag is in the environment
before JAX's C++ backend initialises.

The ``setdefault`` call is intentional: if the user already has
``XLA_FLAGS`` set (e.g. for a GPU run), this does not overwrite it.
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import warnings  # noqa: E402

import jax  # noqa: E402
import pytest  # noqa: E402


def pytest_configure(config):
    """Warn (but do not abort) when fewer than 2 JAX devices are visible.

    The distributed-NU composition tests require 2 virtual CPU devices.
    If the device count is 1 it means either:
      - XLA_FLAGS arrived too late (another conftest or test file imported jax first), or
      - the user is running in an environment that explicitly suppressed virtual devices.
    Individual composition tests guard against this with an in-body
    ``pytest.skip`` so the session continues without them.
    """
    n = len(jax.devices())
    if n < 2:
        warnings.warn(
            f"[conftest] Only {n} JAX device(s) visible. "
            "The distributed-NU composition tests will be skipped. "
            "Expected XLA_FLAGS=--xla_force_host_platform_device_count=2 "
            "to produce 2 virtual CPU devices. "
            "Check that nothing imported jax before this conftest was loaded.",
            stacklevel=1,
        )


@pytest.fixture(scope="session")
def two_devices():
    """Return the first 2 JAX devices, or skip the test if unavailable."""
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip(
            "Need >=2 JAX devices. "
            "Set XLA_FLAGS=--xla_force_host_platform_device_count=2 "
            "before importing jax (conftest.py does this automatically)."
        )
    return devices[:2]
