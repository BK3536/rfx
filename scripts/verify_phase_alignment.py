#!/usr/bin/env python3
"""Verification: apply the measured (slope, intercept) from
``phase_offset_beta_sweep.py`` to rfx S21 and confirm the corrected
result matches Meep within the fit RMS.

Measured fit on the WR-90 slab (2026-04-22):
  slope     = -5.87e-3  m
  intercept = -0.9995  rad  (≈ -57.27°)

Hypothesis: if the only differences between rfx and Meep S21 phase are
(a) a reference-plane misalignment of `slope` metres and (b) a constant
convention phase of `intercept` radians, then:

  S21_rfx_corrected(f) = S21_rfx(f) * exp(-j·[slope·β(f) + intercept])

should agree with S21_Meep to within the fit's RMS (≈ 2.3°).  If the
residual stays at that RMS, the hypothesis is confirmed: the phase
offset is fully a convention/reference-plane effect, not a latent
extractor bug.

Separately: apply the same correction to (a) empty, (b) pec_short —
if the correction is REALLY a convention shift (not slab-specific),
the same (slope, intercept) should also align those cases.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "0")

import numpy as np


C0 = 2.998e8

# Measured on 2026-04-22 WR-90 slab, cpml_layers=20 on both sides,
# reference_plane=0.050/0.150.  See scripts/phase_offset_beta_sweep.py.
MEASURED_SLOPE_M = -5.87e-3
MEASURED_INTERCEPT_RAD = -0.9995


def _load_cv11():
    cv11_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "examples", "crossval", "11_waveguide_port_wr90.py",
    )
    spec = importlib.util.spec_from_file_location("cv11", cv11_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_meep():
    path = ("/root/workspace/byungkwan-workspace/research/microwave-energy/"
            "results/rfx_crossval_wr90_meep/wr90_meep_reference.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _meep_complex(block):
    return np.array([complex(r, i) for r, i in block], dtype=np.complex128)


def _beta_from_freqs(f_hz, fc):
    omega = 2 * np.pi * f_hz
    k0 = omega / C0
    return np.sqrt(np.maximum(k0**2 - (2*np.pi*fc/C0)**2, 0.0))


def _phase_diff_deg(z1, z2):
    d = np.angle(z1) - np.angle(z2)
    d = np.mod(d + np.pi, 2 * np.pi) - np.pi
    return np.degrees(d)


def _apply_correction(s, beta, slope_m=MEASURED_SLOPE_M, intercept_rad=MEASURED_INTERCEPT_RAD):
    """Apply the measured alignment to a complex S-parameter array."""
    correction = np.exp(-1j * (slope_m * beta + intercept_rad))
    return s * correction


def main():
    cv11 = _load_cv11()
    meep = _load_meep()
    if meep is None:
        print("Meep JSON not found.", file=sys.stderr)
        return 2
    r_keys = sorted(k for k in meep.keys() if k.startswith("r") and k != "meta")
    rk = r_keys[-1]

    fc = cv11.F_CUTOFF_TE10

    # Case 1: slab (the fit was derived on this case)
    print("=== Case 1: Dielectric slab (ε_r=2.0, L=10mm) — fit-source ===")
    f_hz, s11_rfx, s21_rfx = cv11.run_rfx_slab(2.0, 0.010)
    beta = _beta_from_freqs(f_hz, fc)
    s21_meep = _meep_complex(meep[rk]["slab"]["s21"])

    # Baseline: rfx vs Meep without correction
    raw_diff = _phase_diff_deg(s21_rfx, s21_meep)
    raw_rms = float(np.sqrt(np.mean(raw_diff**2)))
    raw_max = float(np.max(np.abs(raw_diff)))

    # Corrected rfx vs Meep
    s21_rfx_corrected = _apply_correction(s21_rfx, beta)
    corrected_diff = _phase_diff_deg(s21_rfx_corrected, s21_meep)
    corrected_rms = float(np.sqrt(np.mean(corrected_diff**2)))
    corrected_max = float(np.max(np.abs(corrected_diff)))

    # Magnitude should be untouched
    mag_before = np.abs(s21_rfx).mean()
    mag_after = np.abs(s21_rfx_corrected).mean()

    print(f"  RMS phase diff   before correction : {raw_rms:.2f}°")
    print(f"  RMS phase diff   after  correction : {corrected_rms:.2f}°")
    print(f"  max phase diff   before / after    : {raw_max:.2f}° / {corrected_max:.2f}°")
    print(f"  |S21|_rfx mean   before / after    : {mag_before:.4f} / {mag_after:.4f}  (should be identical)")
    print()

    # Case 2: PEC short — different device, same geometry & reference planes.
    # If the correction reflects a genuine geometry/convention shift, it
    # should help this case too. If it's slab-specific, residual stays.
    print("=== Case 2: PEC short — same (slope, intercept) applied ===")
    f_hz_ps, s11_ps_rfx, _ = cv11.run_rfx_pec_short()
    beta_ps = _beta_from_freqs(f_hz_ps, fc)
    s11_ps_meep = _meep_complex(meep[rk]["pec_short"]["s11"])

    raw_diff_ps = _phase_diff_deg(s11_ps_rfx, s11_ps_meep)
    raw_rms_ps = float(np.sqrt(np.mean(raw_diff_ps**2)))

    # For S11 the convention adjustment is different: S11 = b_port1 / a_port1,
    # both at port-1 reference plane, so the reference-plane shift cancels
    # UNLESS the shift represents a Meep-internal asymmetry between its α⁺
    # (forward, used for a_inc) and α⁻ (backward, used for b_port1) references.
    # Apply the same slope·β + intercept and see.
    s11_ps_rfx_corrected = _apply_correction(s11_ps_rfx, beta_ps)
    corrected_diff_ps = _phase_diff_deg(s11_ps_rfx_corrected, s11_ps_meep)
    corrected_rms_ps = float(np.sqrt(np.mean(corrected_diff_ps**2)))

    print(f"  RMS phase diff   before correction : {raw_rms_ps:.2f}°")
    print(f"  RMS phase diff   after  correction : {corrected_rms_ps:.2f}°")
    print()

    # Verdict
    print("=== Verdict ===")
    if corrected_rms < 5.0:
        print(f"  Slab: correction reduces RMS from {raw_rms:.1f}° to "
              f"{corrected_rms:.1f}° — HYPOTHESIS CONFIRMED for the slab case.")
    else:
        print(f"  Slab: correction reduces RMS from {raw_rms:.1f}° to "
              f"{corrected_rms:.1f}° — residual large; hypothesis partial.")
    if corrected_rms_ps < raw_rms_ps * 0.5:
        print(f"  PEC short: correction also helps (RMS {raw_rms_ps:.1f}° → "
              f"{corrected_rms_ps:.1f}°). Convention shift is universal across geometries.")
    else:
        print(f"  PEC short: correction does NOT transfer (RMS {raw_rms_ps:.1f}° → "
              f"{corrected_rms_ps:.1f}°). The slab fit captured a slab-specific shift only.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
