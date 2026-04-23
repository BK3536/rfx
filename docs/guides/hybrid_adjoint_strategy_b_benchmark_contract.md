# Hybrid adjoint Phase III Strategy B benchmark contract

Status: Gate 0 contract locked  
Date: 2026-04-23

Phase III starts with evidence, not a new replay algorithm.  The benchmark
contract is the version-controlled gate that later Strategy B work must satisfy
before any claim of memory-scaled hybrid adjoint support.

## Command

```bash
python scripts/phase3_strategy_b_benchmark.py --indent 2
```

The command is CPU-safe and requires no external credentials. It emits JSON to
stdout and can also write the same JSON with `--output <path>`.

## Required result fields

Each result row must include:

- `case_id`
- `ladder_role`
- `objective_family`
- `strategy_name`
- `grid_shape`
- `cell_count`
- `n_steps`
- `estimated_memory_gb`
- `runtime_s`
- `correctness_metric`
- `correctness_status`
- `pass`
- `reason`

Gate 0 only evaluates estimator integration, so correctness fields are present
but marked `not_evaluated_gate0_contract_only`. Later gates must replace that
placeholder with parity metrics for landed Strategy B families.

## Ladder order

1. **Primary — source/probe optimize**: clean source/probe workload for the first
   correctness oracle.
2. **Secondary — CPML topology**: CPML zero-sigma dielectric topology lane after
   source/probe correctness is stable.
3. **Tertiary — one-excited + one-passive port proxy**: practical RF proxy lane;
   this does not widen support to generic multi-port workflows.
4. **Optional — NTFF/directivity smoke**: disabled by default and enabled only with
   `--include-optional-ntff` after base Strategy B correctness is stable.

## Strategy rows

Every enabled workload emits rows for:

- `pure_ad_full`
- `strategy_a_replay_trace`
- `strategy_b_segmented_checkpoint`

`strategy_a_replay_trace` is the O(n_steps) custom-VJP trace baseline that
Strategy B is intended to beat. The script also carries the legacy checkpointed
estimator as supplementary metadata, but it is not treated as the Strategy A
acceptance baseline because the estimator itself documents that value as
optimistic for FDTD.

`strategy_b_segmented_checkpoint` is an estimator contract for the planned
checkpointed reconstruction path. It is not a runtime Strategy B implementation.

## Gate 1–2 source/probe prototype

The first runtime prototype is deliberately narrower than the benchmark ladder:

```python
result = sim.forward_hybrid_phase1(
    n_steps=8,
    fallback="raise",
    strategy="b",
    checkpoint_every=3,
)
```

Support can be inspected before execution:

```python
report = sim.inspect_hybrid_strategy_b_phase3(n_steps=8, checkpoint_every=3)
```

Current Strategy B runtime scope:

- PEC and CPML source/probe boundaries
- lossless, nondispersive materials only
- `add_source()` + probe time-series objectives only
- no NTFF, topology density replay, port, conductivity, PEC-mask, or
  PEC-occupancy replay
- no fallback-to-pure-AD for unsupported Strategy B requests; unsupported
  Strategy B selections raise instead of silently changing algorithms

The implementation stores segment checkpoint states and reconstructs per-step
PEC/CPML source/probe states during the custom-VJP backward pass. This proves
the source/probe checkpoint-reconstruction path without claiming topology,
port-proxy, or NTFF support.

## Non-goals

- No Strategy B default policy.
- No broad workflow support expansion.
- No generic multi-port, sigma, waveguide, Floquet, or NTFF promotion from this
  contract alone.
