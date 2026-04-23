# rfx Support Matrix

Status legend:
- **supported** — part of the current claims-bearing surface
- **shadow** — retained and tested, but not claims-bearing yet
- **experimental** — research-grade / partial / promotion pending
- **unsupported** — should fail clearly rather than degrade silently

## Current claims-bearing reference lane

**Lane:** uniform Cartesian Yee RF workflows

### Scope
- boundaries: `pec`, `cpml`, `upml`
- sources: point/current sources, lumped/wire ports, waveguide ports
- observables: time-series probes, flux monitors, S-parameters, Harminv resonances, NTFF/far-field where benchmarked
- materials: isotropic linear, conductive, and validated dispersive subsets
- workflows: cavity, waveguide, patch antenna, simple scattering, resonance, de-embedding, selected differentiable proxy objectives

## Lane summary

| Lane | Status | Current role | Notes |
|---|---|---|---|
| Uniform Yee RF lane | **supported** | claims-bearing reference lane | primary correctness surface |
| Nonuniform graded-z | **shadow** | preserved thin-substrate lane | no silent fallback; no new promotions until re-qualified |
| SBP-SAT / subgridding | experimental | research lane | not part of the claims-bearing surface |
| ADI | experimental | research lane | separate accuracy/stability envelope |
| Distributed | experimental | scaling lane | not part of correctness-bearing baseline |
| Floquet/Bloch | experimental | periodic/phased-array lane | promotion pending explicit benchmark ladder |

## Reference-lane support table

| Dimension | Supported subset |
|---|---|
| grid / runner | uniform Cartesian Yee |
| materials | isotropic linear, conductive, validated Debye/Lorentz/Drude subsets |
| absorbers | PEC, CPML, bounded UPML |
| sources | point/current, lumped port, wire port, waveguide port |
| observables | probes, flux, calibrated S-parameters, Harminv resonance, benchmarked NTFF |
| optimization-facing observables | validated proxy objectives only until explicitly promoted |

## Nonuniform graded-z shadow lane

Current retained subset:
- graded-z thin-substrate workflows with probes and current-style excitation
- smoke/convergence coverage in `tests/test_nonuniform_api.py` and `tests/test_nonuniform_convergence.py`
- selected dispersive-material smoke coverage

Current policy:
- preserved for continuity and qualification work
- **not** part of the claims-bearing reference lane
- no silent fallback / no silent feature dropping
- no new promotions until contract + benchmark ladder exist

### Explicit unsupported combinations in the nonuniform lane

| Combination | Status | Expected behavior |
|---|---|---|
| Floquet + nonuniform | unsupported | hard-fail |
| NTFF + nonuniform | unsupported | hard-fail |
| DFT planes + nonuniform | unsupported | hard-fail |
| TFSF + nonuniform | unsupported | hard-fail |
| Waveguide ports + nonuniform | unsupported | hard-fail |
| Lumped RLC + nonuniform | unsupported | hard-fail |

## Promotion rule

A lane or feature can be promoted to **supported** only when all of the following exist:
1. support-matrix entry
2. explicit source/observable contract where relevant
3. unit + integration tests
4. benchmark / convergence evidence
5. docs/examples/API wording aligned to the promoted scope

## Hybrid optimizer mode policy (Stage 1)

The hybrid-adjoint public optimizer policy is intentionally narrower than the
claims-bearing RF reference lane above. Stage 1 covers only the public
`adjoint_mode` contract on `optimize()` and `topology_optimize()`.

| Public optimizer mode | Status | Stage 1 policy |
|---|---|---|
| `pure_ad` | supported default | public default for `optimize()` and `topology_optimize()` |
| `auto` | experimental-supported / bounded opt-in | inspect support first; select `hybrid` only on landed bounded families; otherwise fall back to `pure_ad` |
| `hybrid` | experimental-supported / strict opt-in | require support inspection to pass; raise on unsupported cases |

See `docs/guides/hybrid_optimizer_mode_policy.md` for the fuller policy matrix,
migration guidance, and explicit Strategy B exclusion.

Stage 1 keeps **Strategy B outside this public optimizer policy matrix**. It
may appear only as a deferred/prototype exclusion note linked to its own
contract; it receives no default, recommended, implied-safe, or equivalent
status here.
