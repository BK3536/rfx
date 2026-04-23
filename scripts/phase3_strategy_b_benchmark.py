"""Phase III Strategy B benchmark contract harness.

Gate 0 for Strategy B is intentionally a contract lock, not an
algorithm implementation.  The harness emits a machine-readable ladder
that fixes the workload order, required schema fields, estimator
integration, and pass/fail memory-budget signal that later Strategy B
runs must preserve when real replay/runtime measurements are added.

Run:
    python scripts/phase3_strategy_b_benchmark.py --indent 2
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

warnings.filterwarnings(
    "ignore",
    message="Unable to import Axes3D.*",
    category=UserWarning,
)

from rfx import GaussianPulse, Simulation  # noqa: E402
from rfx.grid import C0  # noqa: E402

SCHEMA_VERSION = 1
DEFAULT_BUDGET_GB = 24.0
DEFAULT_CHECKPOINT_EVERY = 1000

REQUIRED_RESULT_FIELDS = (
    "case_id",
    "ladder_role",
    "objective_family",
    "strategy_name",
    "grid_shape",
    "cell_count",
    "n_steps",
    "estimated_memory_gb",
    "runtime_s",
    "correctness_metric",
    "correctness_status",
    "pass",
    "reason",
)

STRATEGIES = (
    "pure_ad_full",
    "strategy_a_replay_trace",
    "strategy_b_segmented_checkpoint",
)


@dataclass(frozen=True)
class WorkloadSpec:
    """One benchmark-ladder workload fixed by the Gate 0 contract."""

    case_id: str
    ladder_role: str
    objective_family: str
    description: str
    builder: str
    freq_max_hz: float
    domain_m: tuple[float, float, float]
    dx_m: float
    boundary: str
    cpml_layers: int
    n_steps: int
    enabled_by_default: bool = True


DEFAULT_WORKLOADS = (
    WorkloadSpec(
        case_id="source_probe_optimize_patch",
        ladder_role="primary",
        objective_family="source_probe_optimize",
        description=(
            "Clean source/probe optimize scale ladder for first Strategy B "
            "correctness oracle."
        ),
        builder="source_probe",
        freq_max_hz=10e9,
        domain_m=(0.030, 0.030, 0.030),
        dx_m=0.5e-3,
        boundary="cpml",
        cpml_layers=8,
        n_steps=10_000,
    ),
    WorkloadSpec(
        case_id="cpml_topology_probe_energy",
        ladder_role="secondary",
        objective_family="cpml_topology_probe_energy",
        description=(
            "CPML zero-sigma dielectric topology ladder; validate only after "
            "source/probe is stable."
        ),
        builder="topology_cpml",
        freq_max_hz=10e9,
        domain_m=(0.040, 0.040, 0.040),
        dx_m=0.6e-3,
        boundary="cpml",
        cpml_layers=8,
        n_steps=10_000,
    ),
    WorkloadSpec(
        case_id="one_passive_port_proxy",
        ladder_role="tertiary",
        objective_family="one_excited_one_passive_port_proxy",
        description=(
            "One excited plus one passive lumped-port proxy ladder; no generic "
            "multi-port widening."
        ),
        builder="port_proxy",
        freq_max_hz=8e9,
        domain_m=(0.018, 0.018, 0.018),
        dx_m=0.35e-3,
        boundary="pec",
        cpml_layers=0,
        n_steps=8_000,
    ),
    WorkloadSpec(
        case_id="ntff_directivity_smoke",
        ladder_role="optional",
        objective_family="ntff_directivity_smoke",
        description=(
            "Optional NTFF/directivity smoke; keep disabled until base Strategy "
            "B correctness is stable."
        ),
        builder="ntff_source_probe",
        freq_max_hz=10e9,
        domain_m=(0.030, 0.030, 0.030),
        dx_m=0.6e-3,
        boundary="cpml",
        cpml_layers=8,
        n_steps=8_000,
        enabled_by_default=False,
    ),
)


def _center(domain: tuple[float, float, float]) -> tuple[float, float, float]:
    return (domain[0] / 2.0, domain[1] / 2.0, domain[2] / 2.0)


def _make_sim(spec: WorkloadSpec) -> Simulation:
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
    elif spec.builder == "ntff_source_probe":
        sim.add_source((x_mid, y_mid, max(2 * spec.dx_m, z_mid / 3.0)), "ez")
        sim.add_probe((x_mid, y_mid, z_mid), "ez")
        margin = 4 * spec.dx_m
        sim.add_ntff_box(
            corner_lo=(margin, margin, margin),
            corner_hi=(
                spec.domain_m[0] - margin,
                spec.domain_m[1] - margin,
                spec.domain_m[2] - margin,
            ),
            n_freqs=8,
        )
    else:  # pragma: no cover - protected by static DEFAULT_WORKLOADS.
        raise ValueError(f"unknown workload builder {spec.builder!r}")

    return sim


def _grid_shape(sim: Simulation) -> tuple[int, int, int]:
    dx = sim._dx or (C0 / sim._freq_max / 20.0)

    def axis_cells(extent: float, profile: object | None) -> int:
        if profile is not None:
            return len(profile) + 1 + 2 * sim._cpml_layers  # type: ignore[arg-type]
        return int(math.ceil(extent / dx)) + 1 + 2 * sim._cpml_layers

    return (
        axis_cells(sim._domain[0], sim._dx_profile),
        axis_cells(sim._domain[1], sim._dy_profile),
        axis_cells(sim._domain[2], sim._dz_profile),
    )


def _memory_by_strategy(est: Any) -> dict[str, float]:
    if est.ad_segmented_gb is None:  # pragma: no cover - caller always passes checkpoint_every.
        raise ValueError("segmented Strategy B estimate requires checkpoint_every")
    return {
        "pure_ad_full": est.ad_full_gb,
        # Current Strategy A stores/replays a full per-step trace for the
        # supported custom-VJP family.  Gate 0 therefore treats it as the
        # O(n_steps) baseline that Strategy B must beat; the older
        # ``ad_checkpointed_gb`` estimator is carried as supplementary
        # metadata only because its docstring marks it as optimistic for FDTD.
        "strategy_a_replay_trace": est.ad_full_gb,
        "strategy_b_segmented_checkpoint": est.ad_segmented_gb,
    }


def _fit_reason(strategy: str, memory_gb: float, budget_gb: float) -> tuple[bool, str]:
    if memory_gb <= budget_gb:
        return True, f"{strategy} estimate fits {budget_gb:.1f}GB target budget"
    return False, f"{strategy} estimate exceeds {budget_gb:.1f}GB target budget"


def evaluate_workload(
    spec: WorkloadSpec,
    *,
    budget_gb: float = DEFAULT_BUDGET_GB,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
) -> list[dict[str, Any]]:
    """Evaluate one workload against the Gate 0 memory schema."""
    started = time.perf_counter()
    sim = _make_sim(spec)
    est = sim.estimate_ad_memory(
        n_steps=spec.n_steps,
        available_memory_gb=budget_gb,
        checkpoint_every=checkpoint_every,
    )
    elapsed = time.perf_counter() - started
    shape = _grid_shape(sim)
    cell_count = math.prod(shape)
    memory_by_strategy = _memory_by_strategy(est)

    rows: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        memory_gb = memory_by_strategy[strategy]
        passed, reason = _fit_reason(strategy, memory_gb, budget_gb)
        rows.append(
            {
                "case_id": spec.case_id,
                "ladder_role": spec.ladder_role,
                "objective_family": spec.objective_family,
                "strategy_name": strategy,
                "grid_shape": list(shape),
                "cell_count": cell_count,
                "n_steps": spec.n_steps,
                "estimated_memory_gb": round(float(memory_gb), 6),
                "runtime_s": round(float(elapsed), 6),
                "correctness_metric": None,
                "correctness_status": "not_evaluated_gate0_contract_only",
                "pass": passed,
                "reason": reason,
                "checkpoint_every": (
                    checkpoint_every
                    if strategy == "strategy_b_segmented_checkpoint"
                    else None
                ),
                "forward_memory_gb": round(float(est.forward_gb), 6),
                "legacy_checkpointed_memory_gb": round(float(est.ad_checkpointed_gb), 6),
                "ntff_dft_memory_gb": round(float(est.ntff_dft_gb), 6),
                "warning": est.warning,
                "description": spec.description,
            }
        )
    return rows


def build_contract_report(
    *,
    budget_gb: float = DEFAULT_BUDGET_GB,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    include_optional_ntff: bool = False,
) -> dict[str, Any]:
    """Build the complete Phase III Gate 0 benchmark contract report."""
    enabled_specs = [
        spec
        for spec in DEFAULT_WORKLOADS
        if spec.enabled_by_default or include_optional_ntff
    ]
    disabled_specs = [
        spec
        for spec in DEFAULT_WORKLOADS
        if not spec.enabled_by_default and not include_optional_ntff
    ]
    results: list[dict[str, Any]] = []
    for spec in enabled_specs:
        results.extend(
            evaluate_workload(
                spec,
                budget_gb=budget_gb,
                checkpoint_every=checkpoint_every,
            )
        )

    roles = sorted({row["ladder_role"] for row in results})
    strategies = sorted({row["strategy_name"] for row in results})
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_contract": "phase_iii_strategy_b_gate0",
        "contract_status": "gate0_schema_locked",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "target_memory_gb": budget_gb,
        "checkpoint_every": checkpoint_every,
        "required_result_fields": list(REQUIRED_RESULT_FIELDS),
        "benchmark_order": [
            {
                "case_id": spec.case_id,
                "ladder_role": spec.ladder_role,
                "objective_family": spec.objective_family,
                "enabled": spec in enabled_specs,
                "description": spec.description,
            }
            for spec in DEFAULT_WORKLOADS
        ],
        "results": results,
        "disabled_optional_cases": [spec.case_id for spec in disabled_specs],
        "summary": {
            "result_rows": len(results),
            "enabled_cases": len(enabled_specs),
            "roles": roles,
            "strategies": strategies,
            "rows_with_required_fields": sum(
                all(field in row for field in REQUIRED_RESULT_FIELDS) for row in results
            ),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-gb", type=float, default=DEFAULT_BUDGET_GB)
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    parser.add_argument(
        "--include-optional-ntff",
        action="store_true",
        help="Include the optional NTFF/directivity smoke case in the emitted report.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path. The report is always printed to stdout.",
    )
    parser.add_argument("--indent", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = build_contract_report(
        budget_gb=args.budget_gb,
        checkpoint_every=args.checkpoint_every,
        include_optional_ntff=args.include_optional_ntff,
    )
    text = json.dumps(report, indent=args.indent, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
