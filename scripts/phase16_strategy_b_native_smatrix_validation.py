"""Phase XVI Strategy B native full S-matrix validation artifact.

This script validates the bounded native Strategy B full RF-port S-matrix
promotion for one-excited/one-passive lumped-port fixtures.  It deliberately
keeps standard S-matrix extraction as a separately labeled reference and never
uses it as native Strategy B output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rfx import GaussianPulse, Simulation  # noqa: E402
from rfx.probes.probes import extract_s_matrix  # noqa: E402
from rfx.sources.sources import LumpedPort  # noqa: E402

MAX_ABS_ERROR_THRESHOLD = 5e-3
MAX_REL_ERROR_THRESHOLD = 5e-2
RECIPROCITY_THRESHOLD = 5e-2
PASSIVITY_COLUMN_POWER_THRESHOLD = 1.5
REFERENCE_TRUST_THRESHOLD = 1e-4
INCIDENT_DENOMINATOR_THRESHOLD = 1e-12


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    return value


def _make_two_port_sim(boundary: str) -> tuple[Simulation, GaussianPulse, list[tuple[float, float, float]]]:
    pulse = GaussianPulse(f0=3e9, bandwidth=0.8)
    sim = Simulation(
        freq_max=5e9,
        domain=(0.010, 0.009, 0.009),
        dx=0.001,
        boundary=boundary,
        cpml_layers=0 if boundary == "pec" else 2,
    )
    active = (0.003, 0.004, 0.004)
    passive = (0.004, 0.004, 0.004)
    sim.add_port(active, "ez", impedance=50.0, waveform=pulse)
    sim.add_port(passive, "ez", impedance=50.0, excite=False)
    sim.add_probe((0.006, 0.004, 0.004), "ez")
    return sim, pulse, [active, passive]


def _standard_reference(
    sim: Simulation,
    pulse: GaussianPulse,
    ordered_positions: list[tuple[float, float, float]],
    freqs: jnp.ndarray,
    *,
    n_steps: int,
) -> np.ndarray:
    grid = sim._build_grid()
    base_materials = sim._assemble_materials(grid)[0]
    # Phase XVI requires apples-to-apples reference construction: the originally
    # passive public port is excited with the same deterministic active-derived
    # waveform used by the native sidecar column, not with a None passive source.
    ports = [LumpedPort(position, "ez", 50.0, pulse) for position in ordered_positions]
    return np.asarray(
        extract_s_matrix(
            grid,
            base_materials,
            ports,
            freqs,
            n_steps=n_steps,
            boundary=sim._boundary,
        )
    )


def _case_record(boundary: str, *, n_steps: int, freqs: jnp.ndarray) -> dict[str, Any]:
    sim, pulse, ordered = _make_two_port_sim(boundary)
    inputs = sim.build_hybrid_phase1_inputs(n_steps=n_steps, s_param_freqs=freqs)
    report = sim.inspect_hybrid_strategy_b_phase6_from_inputs(inputs, checkpoint_every=max(1, n_steps // 4))
    native = sim.forward_hybrid_phase1_from_inputs(
        inputs,
        strategy="b",
        checkpoint_every=max(1, n_steps // 4),
    )
    reference = _standard_reference(sim, pulse, ordered, freqs, n_steps=n_steps)
    native_s = np.asarray(native.s_params)
    freqs_out = np.asarray(native.freqs)
    abs_err = np.abs(native_s - reference)
    trusted = np.abs(reference) > REFERENCE_TRUST_THRESHOLD
    max_rel_error = float(np.max(abs_err[trusted] / np.abs(reference[trusted]))) if np.any(trusted) else None
    reciprocity = float(
        np.mean(np.abs(native_s[0, 1, :] - native_s[1, 0, :]))
        / (np.mean((np.abs(native_s[0, 1, :]) + np.abs(native_s[1, 0, :])) / 2.0) + 1e-30)
    )
    column_power = np.sum(np.abs(native_s) ** 2, axis=0)
    max_column_power = float(np.max(column_power))
    max_abs_error = float(np.max(abs_err))
    offdiag_max = float(max(np.max(np.abs(native_s[1, 0, :])), np.max(np.abs(native_s[0, 1, :]))))
    gates = {
        "supported": bool(report.supported),
        "native_full_smatrix_present_valid": native.s_params is not None and native.freqs is not None,
        "native_full_smatrix_shape_valid": list(native_s.shape) == [2, 2, len(freqs)],
        "native_full_smatrix_finite_valid": bool(np.isfinite(native_s.real).all() and np.isfinite(native_s.imag).all()),
        "native_vs_standard_smatrix_correlation_valid": (
            max_abs_error <= MAX_ABS_ERROR_THRESHOLD
            and max_rel_error is not None
            and max_rel_error <= MAX_REL_ERROR_THRESHOLD
        ),
        "two_port_reciprocity_valid": reciprocity <= RECIPROCITY_THRESHOLD,
        "two_port_passivity_envelope_valid": max_column_power <= PASSIVITY_COLUMN_POWER_THRESHOLD,
        "offdiagonal_not_all_zero_valid": offdiag_max > REFERENCE_TRUST_THRESHOLD,
        "freqs_match_valid": bool(np.allclose(freqs_out, np.asarray(freqs), rtol=0.0, atol=0.0)),
    }
    return {
        "boundary": boundary,
        "n_steps": n_steps,
        "freqs_hz": np.asarray(freqs),
        "observable_source": "strategy_b_native_sparams",
        "support_reason_text": report.reason_text,
        "phase1_forward_result_s_params_shape": list(native_s.shape),
        "phase1_forward_result_freqs_shape": list(freqs_out.shape),
        "reference": {
            "source": "standard_extract_s_matrix_separately_labeled_reference",
            "standard_smatrix_shape": list(reference.shape),
            "passive_column_waveform_source": "active_port_waveform",
        },
        "metrics": {
            "max_abs_error": max_abs_error,
            "max_rel_error": max_rel_error,
            "reciprocity_relative_error": reciprocity,
            "max_column_power": max_column_power,
            "offdiagonal_max": offdiag_max,
            "thresholds": {
                "max_abs_error": MAX_ABS_ERROR_THRESHOLD,
                "max_rel_error": MAX_REL_ERROR_THRESHOLD,
                "reciprocity": RECIPROCITY_THRESHOLD,
                "passivity_column_power": PASSIVITY_COLUMN_POWER_THRESHOLD,
                "reference_trust": REFERENCE_TRUST_THRESHOLD,
                "incident_denominator": INCIDENT_DENOMINATOR_THRESHOLD,
            },
        },
        "gates": gates,
    }


def _unsupported_scope_record() -> dict[str, Any]:
    pulse = GaussianPulse(f0=3e9, bandwidth=0.8)
    sim = Simulation(freq_max=5e9, domain=(0.010, 0.009, 0.009), dx=0.001, boundary="pec", cpml_layers=0)
    sim.add_port((0.003, 0.004, 0.004), "ez", impedance=50.0, waveform=pulse, direction="+x")
    sim.add_port((0.004, 0.004, 0.004), "ez", impedance=50.0, excite=False)
    sim.add_probe((0.006, 0.004, 0.004), "ez")
    inputs = sim.build_hybrid_phase1_inputs(n_steps=16, s_param_freqs=jnp.array([3.0e9], dtype=jnp.float32))
    report = sim.inspect_hybrid_strategy_b_phase6_from_inputs(inputs, checkpoint_every=8)
    return {
        "case": "explicit_direction_lumped_two_port",
        "supported": bool(report.supported),
        "reason_text": report.reason_text,
        "fail_closed_valid": (not report.supported) and "explicit port direction" in report.reason_text,
    }


def run_validation(*, n_steps: int = 64) -> dict[str, Any]:
    freqs = jnp.array([2.5e9, 3.0e9, 3.5e9], dtype=jnp.float32)
    cases = {
        boundary: _case_record(boundary, n_steps=n_steps, freqs=freqs)
        for boundary in ("pec", "cpml")
    }
    unsupported = _unsupported_scope_record()
    gates = {
        "native_full_smatrix_present_valid": all(
            case["gates"]["native_full_smatrix_present_valid"] for case in cases.values()
        ),
        "native_full_smatrix_shape_valid": all(
            case["gates"]["native_full_smatrix_shape_valid"] for case in cases.values()
        ),
        "native_full_smatrix_truthfulness_valid": all(
            case["observable_source"] == "strategy_b_native_sparams"
            and case["reference"]["source"] == "standard_extract_s_matrix_separately_labeled_reference"
            for case in cases.values()
        ),
        "native_vs_standard_smatrix_correlation_valid": all(
            case["gates"]["native_vs_standard_smatrix_correlation_valid"] for case in cases.values()
        ),
        "two_port_reciprocity_valid": all(
            case["gates"]["two_port_reciprocity_valid"] for case in cases.values()
        ),
        "two_port_passivity_envelope_valid": all(
            case["gates"]["two_port_passivity_envelope_valid"] for case in cases.values()
        ),
        "unsupported_scope_fail_closed_valid": unsupported["fail_closed_valid"],
    }
    all_gates = all(gates.values())
    return {
        "schema_version": "phase16_strategy_b_native_full_smatrix_v1",
        "overall_status": "phase16_native_full_smatrix_validated" if all_gates else "phase16_native_full_smatrix_failed",
        "strategy_b_seam": {
            "native_s_params_supported": all_gates,
            "observable_source": "strategy_b_native_sparams",
            "phase1_forward_result_s_params_shape": cases["pec"]["phase1_forward_result_s_params_shape"],
            "phase1_forward_result_freqs_shape": cases["pec"]["phase1_forward_result_freqs_shape"],
        },
        "cases": cases,
        "unsupported_scope": unsupported,
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-steps", type=int, default=64)
    parser.add_argument("--indent", type=int, default=None)
    args = parser.parse_args()
    result = run_validation(n_steps=args.n_steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_jsonable(result), indent=args.indent, sort_keys=True))
    print(json.dumps(_jsonable({"output": str(args.output), "overall_status": result["overall_status"], "gates": result["gates"]}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
