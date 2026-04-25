"""Phase VI Strategy B workflow benchmark/evidence harness.

This companion harness is intentionally Phase VI-owned.  It preserves the
locked Phase III Gate 0 benchmark contract by emitting a separate schema for
landed runtime families and their correctness/memory evidence status.

Run:
    python scripts/phase6_strategy_b_workflow_benchmark.py --indent 2
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from rfx import GaussianPulse, Simulation  # noqa: E402
from rfx.grid import C0  # noqa: E402
from rfx.optimize import DesignRegion, optimize  # noqa: E402
from rfx.topology import TopologyDesignRegion, topology_optimize  # noqa: E402

SCHEMA_VERSION = 1
DEFAULT_BUDGET_GB = 24.0
DEFAULT_CHECKPOINT_EVERY = 1000

REQUIRED_RESULT_FIELDS = (
    "case_id",
    "phase",
    "objective_family",
    "runtime_support_status",
    "grid_shape",
    "cell_count",
    "n_steps",
    "checkpoint_every",
    "boundary",
    "runtime_s",
    "estimator_runtime_s",
    "estimated_strategy_a_memory_gb",
    "estimated_strategy_b_memory_gb",
    "correctness_metric",
    "correctness_status",
    "evidence",
    "pass",
    "reason",
)


@dataclass(frozen=True)
class Phase6Workload:
    case_id: str
    objective_family: str
    builder: str
    boundary: str
    domain_m: tuple[float, float, float]
    dx_m: float
    freq_max_hz: float
    cpml_layers: int
    n_steps: int


WORKLOADS = (
    Phase6Workload(
        case_id="source_probe_cpml_strategy_b",
        objective_family="source_probe",
        builder="source_probe",
        boundary="cpml",
        domain_m=(0.030, 0.030, 0.030),
        dx_m=0.5e-3,
        freq_max_hz=10e9,
        cpml_layers=8,
        n_steps=10_000,
    ),
    Phase6Workload(
        case_id="cpml_topology_probe_energy_strategy_b",
        objective_family="cpml_topology_probe_energy",
        builder="topology_cpml",
        boundary="cpml",
        domain_m=(0.040, 0.040, 0.040),
        dx_m=0.6e-3,
        freq_max_hz=10e9,
        cpml_layers=8,
        n_steps=10_000,
    ),
    Phase6Workload(
        case_id="one_passive_port_proxy_strategy_b",
        objective_family="one_excited_one_passive_port_proxy",
        builder="port_proxy",
        boundary="pec",
        domain_m=(0.018, 0.018, 0.018),
        dx_m=0.35e-3,
        freq_max_hz=8e9,
        cpml_layers=0,
        n_steps=8_000,
    ),
)


def _center(domain: tuple[float, float, float]) -> tuple[float, float, float]:
    return (domain[0] / 2.0, domain[1] / 2.0, domain[2] / 2.0)


def _make_sim(spec: Phase6Workload) -> Simulation:
    sim = Simulation(
        freq_max=spec.freq_max_hz,
        domain=spec.domain_m,
        boundary=spec.boundary,
        cpml_layers=spec.cpml_layers,
        dx=spec.dx_m,
    )
    x_mid, y_mid, z_mid = _center(spec.domain_m)
    if spec.builder in {"source_probe", "topology_cpml"}:
        sim.add_source((x_mid, y_mid, max(2 * spec.dx_m, z_mid / 3.0)), "ez")
        sim.add_probe((x_mid, y_mid, z_mid), "ez")
    elif spec.builder == "port_proxy":
        sim.add_port(
            (spec.domain_m[0] * 0.33, y_mid, z_mid),
            "ez",
            impedance=50.0,
            waveform=GaussianPulse(f0=spec.freq_max_hz / 2.0, bandwidth=0.5),
        )
        sim.add_port(
            (spec.domain_m[0] * 0.67, y_mid, z_mid),
            "ez",
            impedance=50.0,
            excite=False,
        )
        sim.add_probe((x_mid, y_mid, z_mid), "ez")
    else:  # pragma: no cover - static workload table protects this.
        raise ValueError(f"unknown Phase VI workload builder {spec.builder!r}")
    return sim


def _make_tiny_source_probe_sim(*, boundary: str = "cpml") -> Simulation:
    sim = Simulation(freq_max=5e9, domain=(0.015, 0.015, 0.015), boundary=boundary)
    sim.add_source(
        (0.005, 0.0075, 0.0075),
        "ez",
        waveform=GaussianPulse(f0=3e9, bandwidth=0.5),
    )
    sim.add_probe((0.010, 0.0075, 0.0075), "ez")
    return sim


def _tiny_topology_objective(result):
    return -jnp.sum(result.time_series ** 2)


def _tiny_probe_objective(result):
    return -jnp.sum(result.time_series ** 2)


def _run_source_probe_correctness(checkpoint_every: int) -> float:
    sim = _make_tiny_source_probe_sim(boundary="cpml")
    inputs = sim.build_hybrid_phase1_inputs(n_steps=8)
    strategy_a = sim.forward_hybrid_phase1_from_inputs(inputs)
    strategy_b = sim.forward_hybrid_phase1_from_inputs(
        inputs,
        strategy="b",
        checkpoint_every=max(1, min(3, checkpoint_every)),
    )
    return float(jnp.max(jnp.abs(strategy_b.time_series - strategy_a.time_series)))


def _run_topology_correctness(checkpoint_every: int) -> float:
    sim = _make_tiny_source_probe_sim(boundary="cpml")
    sim.add_material("phase6_bench_diel", eps_r=4.0, sigma=0.0)
    region = TopologyDesignRegion(
        corner_lo=(0.009, 0.003, 0.003),
        corner_hi=(0.012, 0.006, 0.006),
        material_bg="air",
        material_fg="phase6_bench_diel",
        beta_projection=1.0,
    )
    pure = topology_optimize(
        sim,
        region,
        _tiny_topology_objective,
        n_iterations=1,
        learning_rate=0.05,
        n_steps=8,
        beta_schedule=[(0, 1.0)],
        verbose=False,
        adjoint_mode="pure_ad",
    )
    strategy_b = topology_optimize(
        sim,
        region,
        _tiny_topology_objective,
        n_iterations=1,
        learning_rate=0.05,
        n_steps=8,
        beta_schedule=[(0, 1.0)],
        verbose=False,
        adjoint_mode="hybrid",
        strategy="b",
        checkpoint_every=max(1, min(3, checkpoint_every)),
    )
    return float(np.max(np.abs(np.asarray(strategy_b.history) - np.asarray(pure.history))))


def _run_port_proxy_correctness(checkpoint_every: int) -> float:
    sim = Simulation(freq_max=5e9, domain=(0.015, 0.015, 0.015), boundary="pec")
    sim.add_port(
        (0.005, 0.0075, 0.0075),
        "ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=3e9, bandwidth=0.5),
    )
    sim.add_port((0.010, 0.0075, 0.0075), "ez", impedance=50.0, excite=False)
    sim.add_probe((0.012, 0.0075, 0.0075), "ez")
    region = DesignRegion(
        corner_lo=(0.009, 0.003, 0.003),
        corner_hi=(0.012, 0.006, 0.006),
        eps_range=(1.0, 4.4),
    )
    pure = optimize(
        sim,
        region,
        _tiny_probe_objective,
        n_iters=1,
        lr=0.01,
        n_steps=8,
        verbose=False,
        adjoint_mode="pure_ad",
    )
    strategy_b = optimize(
        sim,
        region,
        _tiny_probe_objective,
        n_iters=1,
        lr=0.01,
        n_steps=8,
        verbose=False,
        adjoint_mode="hybrid",
        strategy="b",
        checkpoint_every=max(1, min(3, checkpoint_every)),
    )
    return float(
        np.max(np.abs(np.asarray(strategy_b.loss_history) - np.asarray(pure.loss_history)))
    )


def _run_correctness(spec: Phase6Workload, checkpoint_every: int) -> tuple[float, float]:
    started = time.perf_counter()
    if spec.builder == "source_probe":
        metric = _run_source_probe_correctness(checkpoint_every)
    elif spec.builder == "topology_cpml":
        metric = _run_topology_correctness(checkpoint_every)
    elif spec.builder == "port_proxy":
        metric = _run_port_proxy_correctness(checkpoint_every)
    else:  # pragma: no cover - static workload table protects this.
        raise ValueError(f"unknown Phase VI workload builder {spec.builder!r}")
    return metric, time.perf_counter() - started


def _grid_shape(sim: Simulation) -> tuple[int, int, int]:
    dx = sim._dx or (C0 / sim._freq_max / 20.0)
    return tuple(
        int(math.ceil(extent / dx)) + 1 + 2 * sim._cpml_layers
        for extent in sim._domain
    )


def evaluate_workload(
    spec: Phase6Workload,
    *,
    budget_gb: float = DEFAULT_BUDGET_GB,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
) -> dict[str, Any]:
    estimator_started = time.perf_counter()
    sim = _make_sim(spec)
    est = sim.estimate_ad_memory(
        n_steps=spec.n_steps,
        available_memory_gb=budget_gb,
        checkpoint_every=checkpoint_every,
    )
    estimator_runtime_s = time.perf_counter() - estimator_started
    correctness_metric, runtime_s = _run_correctness(spec, checkpoint_every)
    shape = _grid_shape(sim)
    strategy_a_gb = float(est.ad_full_gb)
    strategy_b_gb = float(est.ad_segmented_gb)
    correctness_passed = correctness_metric <= 1e-5
    memory_passed = strategy_b_gb <= budget_gb
    passed = memory_passed and correctness_passed
    return {
        "case_id": spec.case_id,
        "phase": "phase_vi_strategy_b_runtime",
        "objective_family": spec.objective_family,
        "runtime_support_status": "implemented",
        "grid_shape": list(shape),
        "cell_count": math.prod(shape),
        "n_steps": spec.n_steps,
        "checkpoint_every": checkpoint_every,
        "boundary": spec.boundary,
        "runtime_s": round(runtime_s, 6),
        "estimator_runtime_s": round(estimator_runtime_s, 6),
        "estimated_strategy_a_memory_gb": round(strategy_a_gb, 6),
        "estimated_strategy_b_memory_gb": round(strategy_b_gb, 6),
        "correctness_metric": round(correctness_metric, 12),
        "correctness_status": "passed" if correctness_passed else "failed",
        "evidence": {
            "strategy_a_memory_model": "full replay trace estimate",
            "strategy_b_memory_model": "segmented checkpoint estimate",
            "tiny_fixture_runtime_note": "tiny fixtures may be slower; this row tracks bounded checkpoint growth",
        },
        "pass": passed,
        "reason": (
            f"Strategy B estimate fits {budget_gb:.1f}GB target budget"
            if memory_passed
            else f"Strategy B estimate exceeds {budget_gb:.1f}GB target budget"
        ),
    }


def build_phase6_report(
    *,
    budget_gb: float = DEFAULT_BUDGET_GB,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
) -> dict[str, Any]:
    rows = [
        evaluate_workload(
            spec,
            budget_gb=budget_gb,
            checkpoint_every=checkpoint_every,
        )
        for spec in WORKLOADS
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_contract": "phase_vi_strategy_b_workflow_runtime",
        "contract_status": "phase_vi_runtime_evidence",
        "preserves_phase_iii_gate0_contract": True,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "target_memory_gb": budget_gb,
        "checkpoint_every": checkpoint_every,
        "required_result_fields": list(REQUIRED_RESULT_FIELDS),
        "results": rows,
        "summary": {
            "result_rows": len(rows),
            "runtime_supported_cases": sum(
                row["runtime_support_status"] == "implemented" for row in rows
            ),
            "rows_with_required_fields": sum(
                all(field in row for field in REQUIRED_RESULT_FIELDS) for row in rows
            ),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-gb", type=float, default=DEFAULT_BUDGET_GB)
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path. The report is always printed to stdout.",
    )
    parser.add_argument("--indent", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = build_phase6_report(
        budget_gb=args.budget_gb,
        checkpoint_every=args.checkpoint_every,
    )
    text = json.dumps(report, indent=args.indent, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
