# Codex Spec 6C: CW and Custom Waveform Sources

## Goal
Add continuous-wave (CW) and user-defined waveform sources so the simulator
supports steady-state analysis and arbitrary excitation.

## Context
- `rfx/sources/sources.py` has `GaussianPulse` with `__call__(t) -> float`.
- `rfx/simulation.py` `make_source()` takes a pulse object and calls `pulse(t)`.
- The compiled `run()` via `jax.lax.scan` expects `src_waveform` as a
  precomputed `(n_steps,)` array built in `make_source()`.

## Deliverable

### 1. `CWSource` class in `rfx/sources/sources.py`
```python
class CWSource:
    """Continuous-wave sinusoidal source with smooth ramp-up.

    Parameters
    ----------
    f0 : float
        Frequency in Hz.
    amplitude : float
        Peak amplitude.
    ramp_steps : int
        Number of timesteps for smooth cosine taper onset (0 = instant).
    """
    def __init__(self, f0, amplitude=1.0, ramp_steps=50): ...
    def __call__(self, t): ...
```

The ramp should be: `amplitude * sin(2π·f0·t) * min(1, 0.5*(1 - cos(π*step/ramp_steps)))`.
After `ramp_steps`, the envelope is constant 1.0.

### 2. `CustomWaveform` class in `rfx/sources/sources.py`
```python
class CustomWaveform:
    """User-defined waveform.

    Parameters
    ----------
    func : callable
        Function `f(t: float) -> float` returning the source amplitude at time t.
    """
    def __init__(self, func): ...
    def __call__(self, t): ...
```

Simply wraps and calls `func(t)`.

### 3. Export both from `rfx/__init__.py`
Add `CWSource` and `CustomWaveform` to the public API exports.

### 4. Tests in `tests/test_custom_waveforms.py`

**Test 1: `test_cw_source_reaches_steady_state`**
- Create a small domain (0.03, 0.01, 0.01) with CPML
- CW source at f0=5e9, run for 300 steps
- Measure Ez at a probe; after ramp_steps, amplitude should be approximately constant
- Assert: std(envelope) / mean(envelope) < 0.15 for the last 100 steps

**Test 2: `test_cw_source_dft_peak_at_f0`**
- Same setup, run with a DFT probe at f0 and f0*2
- Assert: |DFT(f0)| >> |DFT(2*f0)| (at least 10x ratio)

**Test 3: `test_custom_waveform_matches_gaussian`**
- Create a `CustomWaveform(lambda t: GaussianPulse(f0=3e9)(t))`
- Run side-by-side with a regular `GaussianPulse(f0=3e9)`
- Assert: time series are identical (allclose)

**Test 4: `test_cw_source_differentiable`**
- Use CW source in a `run()` with `checkpoint=True`
- Compute `jax.grad` of sum(time_series^2) w.r.t. eps_r
- Assert: gradient is non-zero

## Constraints
- Each test < 60 seconds on CPU
- Do NOT modify `make_source()` or `run()` — the existing interface already
  supports any callable with `__call__(t)`. Just verify it works.
- Only create new files + add exports to `__init__.py`

## Verification
Run: `pytest -xvs tests/test_custom_waveforms.py`
All 4 tests must pass.
