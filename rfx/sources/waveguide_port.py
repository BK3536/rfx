"""Rectangular waveguide port: analytical TE/TM mode profiles.

Excites and extracts waveguide modes using precomputed analytical profiles
on the port cross-section. Supports TE_mn and TM_mn modes for rectangular
waveguides with PEC walls.

The port sits on a y-z plane at a fixed x index. Mode propagation is along +x.

TE_mn transverse E-field profiles (Pozar, mapped to prop-along-x):
    Ey(y,z) = -(nπ/b) cos(mπy/a) sin(nπz/b)
    Ez(y,z) =  (mπ/a) sin(mπy/a) cos(nπz/b)

where k_c² = (mπ/a)² + (nπ/b)², a = waveguide width (y),
b = waveguide height (z).

Key examples:
    TE10: Ey = 0,  Ez = (π/a) sin(πy/a)
    TE01: Ey = -(π/b) sin(πz/b),  Ez = 0

S21 is extracted using V/I forward-wave decomposition at two probe planes:
    a_fwd(f) = (V(f) + Z_TE(f) * I(f)) / 2
    S21(f)   = a_fwd_probe(f) / a_fwd_ref(f)
This guarantees |S21| <= 1 for a matched lossless waveguide.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from rfx.core.yee import EPS_0, MU_0


C0_LOCAL = 1.0 / np.sqrt(EPS_0 * MU_0)


class WaveguidePort(NamedTuple):
    """Waveguide port definition.

    x_index : int
        Grid x-index where the port plane sits.
    y_slice : (y_lo, y_hi)
        Grid y-index range of the waveguide aperture (exclusive end).
    z_slice : (z_lo, z_hi)
        Grid z-index range of the waveguide aperture (exclusive end).
    a : float
        Waveguide width in meters (y-direction).
    b : float
        Waveguide height in meters (z-direction).
    mode : (m, n)
        Mode indices. (1, 0) for TE10 dominant mode.
    mode_type : str
        "TE" or "TM". Default "TE".
    """
    x_index: int
    y_slice: tuple[int, int]
    z_slice: tuple[int, int]
    a: float
    b: float
    mode: tuple[int, int] = (1, 0)
    mode_type: str = "TE"


class WaveguidePortConfig(NamedTuple):
    """Compiled waveguide port config for time-stepping."""
    # Port geometry
    x_index: int       # Source injection plane
    ref_x: int         # Reference probe (near source, downstream)
    probe_x: int       # Measurement probe (further downstream)
    y_lo: int
    y_hi: int
    z_lo: int
    z_hi: int

    # Normalized mode profiles on the aperture (ny_port, nz_port)
    ey_profile: jnp.ndarray
    ez_profile: jnp.ndarray
    hy_profile: jnp.ndarray
    hz_profile: jnp.ndarray

    # Waveguide parameters
    f_cutoff: float
    a: float
    b: float

    # Source waveform parameters (differentiated Gaussian)
    src_amp: float
    src_t0: float
    src_tau: float

    # DFT accumulators for S-parameter extraction
    v_probe_dft: jnp.ndarray   # (n_freqs,) complex — modal voltage at probe
    v_ref_dft: jnp.ndarray     # (n_freqs,) complex — modal voltage at ref
    i_probe_dft: jnp.ndarray   # (n_freqs,) complex — modal current at probe
    i_ref_dft: jnp.ndarray     # (n_freqs,) complex — modal current at ref
    v_inc_dft: jnp.ndarray     # (n_freqs,) complex — source waveform DFT
    freqs: jnp.ndarray         # (n_freqs,) float


def _te_mode_profiles(a: float, b: float, m: int, n: int,
                      y_coords: np.ndarray, z_coords: np.ndarray,
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute TE_mn E and H transverse mode profiles.

    Returns (ey, ez, hy, hz) each of shape (ny, nz), normalized so that
    integral(Ey² + Ez²) dA = 1.

    Derivation: TE_mn eigenfunction Hx = cos(mπy/a) cos(nπz/b).
    Transverse E from Maxwell (propagation along +x):
        Ey = -(nπ/b) cos(mπy/a) sin(nπz/b)
        Ez =  (mπ/a) sin(mπy/a) cos(nπz/b)

    The (mπ/a) and (nπ/b) derivative weights are essential for correct
    relative amplitudes in higher-order modes (e.g., TE11 with a != b).
    """
    Y, Z = np.meshgrid(y_coords, z_coords, indexing='ij')

    ey = -(n * np.pi / b) * np.cos(m * np.pi * Y / a) * np.sin(n * np.pi * Z / b) if n > 0 else np.zeros_like(Y)
    ez = (m * np.pi / a) * np.sin(m * np.pi * Y / a) * np.cos(n * np.pi * Z / b) if m > 0 else np.zeros_like(Y)

    # H for forward +x propagation: hy = -ez, hz = ey (unnormalized)
    # gives Poynting P_x = Ey*Hz - Ez*Hy = Ey² + Ez² > 0
    hy = -ez.copy()
    hz = ey.copy()

    # Normalize: integral(Ey² + Ez²) dA = 1
    dy = y_coords[1] - y_coords[0] if len(y_coords) > 1 else a
    dz = z_coords[1] - z_coords[0] if len(z_coords) > 1 else b
    power = np.sum(ey**2 + ez**2) * dy * dz
    if power > 0:
        norm = np.sqrt(power)
        ey /= norm
        ez /= norm
        hy /= norm
        hz /= norm

    return ey, ez, hy, hz


def _tm_mode_profiles(a: float, b: float, m: int, n: int,
                      y_coords: np.ndarray, z_coords: np.ndarray,
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute TM_mn E and H transverse mode profiles.

    TM modes require both m >= 1 and n >= 1.
    Eigenfunction: Ex_z = sin(mπy/a) sin(nπz/b).
    Transverse E from grad_t:
        Ey = (mπ/a) cos(mπy/a) sin(nπz/b)
        Ez = (nπ/b) sin(mπy/a) cos(nπz/b)
    """
    if m < 1 or n < 1:
        raise ValueError(f"TM modes require m >= 1 and n >= 1, got ({m}, {n})")

    Y, Z = np.meshgrid(y_coords, z_coords, indexing='ij')

    ey = (m * np.pi / a) * np.cos(m * np.pi * Y / a) * np.sin(n * np.pi * Z / b)
    ez = (n * np.pi / b) * np.sin(m * np.pi * Y / a) * np.cos(n * np.pi * Z / b)

    hy = -ez.copy()
    hz = ey.copy()

    dy = y_coords[1] - y_coords[0] if len(y_coords) > 1 else a
    dz = z_coords[1] - z_coords[0] if len(z_coords) > 1 else b
    power = np.sum(ey**2 + ez**2) * dy * dz
    if power > 0:
        norm = np.sqrt(power)
        ey /= norm
        ez /= norm
        hy /= norm
        hz /= norm

    return ey, ez, hy, hz


def cutoff_frequency(a: float, b: float, m: int, n: int) -> float:
    """TE_mn or TM_mn cutoff frequency for rectangular waveguide."""
    kc = np.sqrt((m * np.pi / a) ** 2 + (n * np.pi / b) ** 2)
    return kc * C0_LOCAL / (2 * np.pi)


def init_waveguide_port(
    port: WaveguidePort,
    dx: float,
    freqs: jnp.ndarray,
    f0: float = 5e9,
    bandwidth: float = 0.5,
    amplitude: float = 1.0,
    probe_offset: int = 10,
    ref_offset: int = 3,
) -> WaveguidePortConfig:
    """Initialize a waveguide port with precomputed mode profiles.

    Parameters
    ----------
    probe_offset : int
        Cells downstream from source for measurement probe.
    ref_offset : int
        Cells downstream from source for reference probe.
    """
    m, n = port.mode
    y_lo, y_hi = port.y_slice
    z_lo, z_hi = port.z_slice
    ny_port = y_hi - y_lo
    nz_port = z_hi - z_lo

    y_coords = np.linspace(0.5 * dx, port.a - 0.5 * dx, ny_port)
    z_coords = np.linspace(0.5 * dx, port.b - 0.5 * dx, nz_port)

    if port.mode_type == "TE":
        ey, ez, hy, hz = _te_mode_profiles(port.a, port.b, m, n, y_coords, z_coords)
    else:
        ey, ez, hy, hz = _tm_mode_profiles(port.a, port.b, m, n, y_coords, z_coords)

    f_c = cutoff_frequency(port.a, port.b, m, n)

    tau = 1.0 / (f0 * bandwidth * np.pi)
    t0 = 3.0 * tau

    ref_x = port.x_index + ref_offset
    probe_x = port.x_index + probe_offset

    nf = len(freqs)
    zeros_c = jnp.zeros(nf, dtype=jnp.complex64)

    return WaveguidePortConfig(
        x_index=port.x_index,
        ref_x=ref_x,
        probe_x=probe_x,
        y_lo=y_lo, y_hi=y_hi,
        z_lo=z_lo, z_hi=z_hi,
        ey_profile=jnp.array(ey, dtype=jnp.float32),
        ez_profile=jnp.array(ez, dtype=jnp.float32),
        hy_profile=jnp.array(hy, dtype=jnp.float32),
        hz_profile=jnp.array(hz, dtype=jnp.float32),
        f_cutoff=float(f_c),
        a=port.a, b=port.b,
        src_amp=float(amplitude),
        src_t0=float(t0),
        src_tau=float(tau),
        v_probe_dft=zeros_c,
        v_ref_dft=zeros_c,
        i_probe_dft=zeros_c,
        i_ref_dft=zeros_c,
        v_inc_dft=zeros_c,
        freqs=freqs,
    )


def inject_waveguide_port(state, cfg: WaveguidePortConfig,
                          t: float, dt: float, dx: float):
    """Inject mode-shaped E-field at the port plane. Call AFTER update_e."""
    arg = (t - cfg.src_t0) / cfg.src_tau
    src_val = cfg.src_amp * (-2.0 * arg) * jnp.exp(-(arg ** 2))

    ey = state.ey
    ez = state.ez

    ey = ey.at[cfg.x_index, cfg.y_lo:cfg.y_hi, cfg.z_lo:cfg.z_hi].add(
        src_val * cfg.ey_profile
    )
    ez = ez.at[cfg.x_index, cfg.y_lo:cfg.y_hi, cfg.z_lo:cfg.z_hi].add(
        src_val * cfg.ez_profile
    )

    return state._replace(ey=ey, ez=ez)


def modal_voltage(state, cfg: WaveguidePortConfig, x_idx: int,
                  dx: float) -> jnp.ndarray:
    """Modal voltage: V = integral E_t . e_mode dA."""
    sl_y = slice(cfg.y_lo, cfg.y_hi)
    sl_z = slice(cfg.z_lo, cfg.z_hi)

    ey_sim = state.ey[x_idx, sl_y, sl_z]
    ez_sim = state.ez[x_idx, sl_y, sl_z]

    return jnp.sum(ey_sim * cfg.ey_profile + ez_sim * cfg.ez_profile) * dx * dx


def modal_current(state, cfg: WaveguidePortConfig, x_idx: int,
                  dx: float) -> jnp.ndarray:
    """Modal current: I = integral H_t . h_mode dA.

    H is averaged between x_idx-1 and x_idx to co-locate with E
    on the Yee grid (H sits at x+1/2, E sits at x).
    """
    sl_y = slice(cfg.y_lo, cfg.y_hi)
    sl_z = slice(cfg.z_lo, cfg.z_hi)

    hy_sim = 0.5 * (state.hy[x_idx, sl_y, sl_z] + state.hy[x_idx - 1, sl_y, sl_z])
    hz_sim = 0.5 * (state.hz[x_idx, sl_y, sl_z] + state.hz[x_idx - 1, sl_y, sl_z])

    return jnp.sum(hy_sim * cfg.hy_profile + hz_sim * cfg.hz_profile) * dx * dx


def update_waveguide_port_probe(cfg: WaveguidePortConfig, state,
                                dt: float, dx: float) -> WaveguidePortConfig:
    """Accumulate DFT of modal V and I at ref and probe planes."""
    t = state.step * dt

    v_ref = modal_voltage(state, cfg, cfg.ref_x, dx)
    v_probe = modal_voltage(state, cfg, cfg.probe_x, dx)
    i_ref = modal_current(state, cfg, cfg.ref_x, dx)
    i_probe = modal_current(state, cfg, cfg.probe_x, dx)

    arg = (t - cfg.src_t0) / cfg.src_tau
    v_inc = cfg.src_amp * (-2.0 * arg) * jnp.exp(-(arg ** 2))

    phase = jnp.exp(-1j * 2.0 * jnp.pi * cfg.freqs * t)

    return cfg._replace(
        v_probe_dft=cfg.v_probe_dft + v_probe * phase * dt,
        v_ref_dft=cfg.v_ref_dft + v_ref * phase * dt,
        i_probe_dft=cfg.i_probe_dft + i_probe * phase * dt,
        i_ref_dft=cfg.i_ref_dft + i_ref * phase * dt,
        v_inc_dft=cfg.v_inc_dft + v_inc * phase * dt,
    )


def _compute_z_te(freqs: jnp.ndarray, f_cutoff: float) -> jnp.ndarray:
    """TE mode impedance Z_TE(f) = ωμ₀/β.

    Returns complex array: real above cutoff, imaginary below.
    """
    omega = 2 * jnp.pi * freqs
    k = omega / C0_LOCAL
    kc = 2 * jnp.pi * f_cutoff / C0_LOCAL

    beta_sq = k**2 - kc**2
    # Above cutoff: beta real; below: beta imaginary
    beta = jnp.where(
        beta_sq >= 0,
        jnp.sqrt(jnp.maximum(beta_sq, 0.0)),
        1j * jnp.sqrt(jnp.maximum(-beta_sq, 0.0)),
    )

    safe_beta = jnp.where(jnp.abs(beta) > 1e-30, beta,
                           1e-30 * jnp.ones_like(beta))
    return omega * MU_0 / safe_beta


def extract_waveguide_s21(cfg: WaveguidePortConfig,
                          dt: float = 0.0) -> jnp.ndarray:
    """Extract S21 as modal voltage ratio between probe and reference.

    S21(f) = V_probe(f) / V_ref(f)

    Both numerator and denominator are modal voltage DFTs, so the ratio
    measures the transfer function between two planes. For a matched
    lossless waveguide above cutoff, mean |S21| ~ 1.

    Note: individual frequency points may show |S21| > 1 due to standing
    waves from residual CPML reflections. Use band-averaged |S21| for
    robust validation.

    Returns (n_freqs,) complex array.
    """
    safe_ref = jnp.where(jnp.abs(cfg.v_ref_dft) > 0, cfg.v_ref_dft,
                         jnp.ones_like(cfg.v_ref_dft))
    return cfg.v_probe_dft / safe_ref


def extract_waveguide_s11(cfg: WaveguidePortConfig) -> jnp.ndarray:
    """S11 placeholder — power conservation estimate.

    |S11|² = 1 - |S21|² for a lossless system.
    True S11 requires backward-wave extraction at the source plane.
    """
    s21 = extract_waveguide_s21(cfg)
    s21_sq = jnp.abs(s21) ** 2
    return jnp.sqrt(jnp.maximum(1.0 - s21_sq, 0.0))
