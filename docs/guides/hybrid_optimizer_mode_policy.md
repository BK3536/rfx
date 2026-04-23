# Hybrid Optimizer Mode Policy

Status: Stage 1 public policy  
Date: 2026-04-24

This guide defines the current **public optimizer mode policy** for:

- `optimize()`
- `topology_optimize()`

It is intentionally narrower than the broader solver/reference-lane support
matrix. It describes only the public `adjoint_mode` contract that is already
landed, support-inspected, and verified.

## Stage 1 public policy

| Mode | Current status | Meaning | When to use | Unsupported behavior |
|---|---|---|---|---|
| `pure_ad` | supported default | existing full pure-AD path | default choice; safest baseline | n/a |
| `auto` | experimental-supported / bounded recommended path | inspect support first, use `hybrid` only for landed bounded families, otherwise fall back to `pure_ad` | preferred opt-in path for the bounded supported hybrid families below | fallback to `pure_ad` |
| `hybrid` | experimental-supported / strict opt-in | require support inspection to pass | when you want explicit hybrid use and explicit failure on unsupported cases | raise with support reason |

## Landed bounded families covered by Stage 1

### `optimize(..., adjoint_mode="auto" | "hybrid")`

Current landed hybrid-supported subset:

- time-domain / probe-proxy optimize flows
- exactly one excited lumped port
- optionally exactly one passive lumped port
- design region disjoint from all lumped-port load cells
- no additional passive ports
- no multi-excited-port layouts
- no wire / waveguide / Floquet / coaxial-specific replay

### `topology_optimize(..., adjoint_mode="auto" | "hybrid")`

Current landed hybrid-supported subset:

- source/probe topology fixtures only
- PEC or CPML boundary
- zero sigma everywhere
- nondispersive materials
- dielectric-only topology (`pec_occupancy_design is None`)
- no ports
- no pre-existing PEC / `pec_mask`
- uniform, non-periodic fixtures

## What Stage 1 does **not** do

Stage 1 does **not**:

- change the public default away from `pure_ad`
- treat `auto` as generic hybrid readiness
- widen physics or replay families beyond the landed bounded subsets above
- describe unsupported cases as recommended or production-default

## Strategy B exclusion

Strategy B is **not** part of the Stage 1 public optimizer policy matrix.

Today, Strategy B exists only as a bounded `Simulation` seam / inspection
surface for source/probe PEC/CPML scale-up work. It is a prototype lane with a
separate contract and separate deferred-family fences. In Stage 1:

- Strategy B gets **no default status**
- Strategy B gets **no recommended status**
- Strategy B gets **no implied-safe or equivalent public status**
- Strategy B may appear only in an appendix/exclusion note or a direct link to
  its own contract

For the current Strategy B runtime boundary, see
`docs/guides/hybrid_adjoint_strategy_b_benchmark_contract.md`.

## Migration guidance

### If you want the safest path
Stay on `pure_ad`.

### If your workflow matches a landed bounded family
Try `auto` first. It is the Stage 1 recommended opt-in path because it inspects
support before choosing `hybrid` and falls back to `pure_ad` otherwise.

### If you want strict enforcement
Use `hybrid`. Unsupported cases should raise with an explicit support reason
instead of silently taking a different route.

### If you are evaluating Strategy B
Treat it as a separate prototype seam. Do not assume it is covered by the
public optimizer policy described here.
