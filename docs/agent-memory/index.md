# rfx Agent Memory — Durable Knowledge Index

## Read first

- `public-site-sync-checklist.md`

## Session Handover (2026-04-10)

### GPU Validation Results (VESSL #369367232843, RTX 4090)

| Phase | Result |
|-------|--------|
| Fast CI (non-slow) | **704 passed, 9 failed**, 1 skipped (76 min) |
| Slow tests (CPML, ADI, SBP-SAT, PML) | **38 passed, 0 failed** (12 min) |
| New Feature Validation | **5 PASS, 0 FAIL** |

#### 9 Fast CI Failures

| Test | Error | Category |
|------|-------|----------|
| `test_gradient_simple::test_gradient_eps_shifts_energy` | NameError | **Bug — missing import** |
| `test_gradient_simple::test_gradient_finite_difference_match` | NameError | **Bug — missing import** |
| `test_optimize::test_forward_returns_minimal_result_contract` | Unknown | **Needs traceback** |
| `test_physics::test_fresnel_normal_incidence` | AssertionError | **Regression — UPML/subpixel 관련 가능** |
| `test_ris::test_ris_sweep_capacitance` | ValueError: Floquet unsupported | Expected (experimental lane) |
| `test_ris::test_ris_sweep_angle` | ValueError: Floquet unsupported | Expected (experimental lane) |
| `test_sbp_sat_alpha::test_init_subgrid_3d_default_tau` | Unknown | **Needs traceback** |
| `test_topology::test_pec_foreground_gradient_is_finite_and_nonzero` | Unknown | **Needs traceback** |
| `test_verification::test_oblique_tfsf_fresnel` | AssertionError | **Regression — UPML/subpixel 관련 가능** |

**Verdict**: 2 expected Floquet failures + 7 real failures to investigate.
Gradient NameError는 quick fix 가능성 높음. Fresnel/TFSF 물리 assertion은 최근 UPML/subpixel 변경과 관련 가능.

---

### Repo Cleanup (2026-04-10)

6 commits pushed to main:

1. `bad2446` — fix: pyproject.toml explicit package discovery (forge_example 제외)
2. `3f7091a` — chore: 56 stale example files 삭제 (01-09, 40_*, 50_*, crossval 01-23)
3. `ac1b8c2` — feat: 6 new crossval diagnostics
4. `2d39cf7` — docs: support matrix + reference-lane contract
5. `130798f` — docs: lane annotations (README, public docs, agent overview)
6. `1d81190` — chore: gitignore forge_example/

---

### Crossval Examples 현황 (9 scripts)

| File | Description | Status |
|------|-------------|--------|
| `01_field_progression_review.py` | Field progression review: rfx vs Meep | Committed |
| `01_meep_waveguide_bend.py` | Waveguide bend transmittance (Meep Basics) | Committed |
| `02_deep_field_diagnostic.py` | Deep field diagnostic: rfx vs Meep bend | New, committed |
| `03_grid_aligned_comparison.py` | Grid-aligned field comparison | New, committed |
| `04_courant_test.py` | Courant number S=0.5 test | New, committed |
| `05_meep_ring_resonator.py` | Ring resonator: rfx vs Meep | Committed |
| `24_gpr_ascan.py` | Ground Penetrating Radar A-Scan | New, committed |
| `25_horn_antenna.py` | Open-ended rectangular waveguide horn | New, committed |
| `26_coupled_line_bpf.py` | Coupled microstrip line BPF | New, committed |

**Note**: 01-05는 Meep cross-validation 계열 (waveguide bend 진단 집중). 24-26은 독립 RF 시나리오.
GPU 실행 결과 미확인 — 다음 세션에서 crossval GPU run 필요.

---

### 남은 작업 (우선순위 순)

#### P0 — GPU Test Failures (7 real)
- [ ] `test_gradient_simple` NameError × 2 — 원인 파악 + fix
- [ ] `test_optimize::test_forward_returns_minimal_result_contract` — traceback 필요
- [ ] `test_physics::test_fresnel_normal_incidence` — UPML/subpixel regression?
- [ ] `test_verification::test_oblique_tfsf_fresnel` — UPML/subpixel regression?
- [ ] `test_sbp_sat_alpha::test_init_subgrid_3d_default_tau` — traceback 필요
- [ ] `test_topology::test_pec_foreground_gradient_is_finite_and_nonzero` — traceback 필요

#### P1 — Known Issues
- [ ] `test_floquet::test_unit_cell_with_floquet` — pre-existing NaN (Floquet+NU 비호환)
- [ ] Far-field dS per-face for non-uniform z — audit item #9

#### P2 — Quick Wins
- [ ] PyPI version bump (v1.4.0 → v1.5.0?)
- [ ] Crossval GPU validation run (9 scripts)

#### P3 — Advanced (deferred)
- [ ] Auto-subgrid (AMR indicator → subgrid)
- [ ] Level-set topology optimization
- [ ] Neural surrogate pipeline (사용자 요청 시)

---

### 핵심 설계 원칙

1. **Physical absolute coordinates**: Probe/port/source 위치는 항상 물리 절대 좌표
2. **Axis-aware formulas**: `d_parallel/(Z0·d_perp1·d_perp2)` — cubic cell 가정 금지
3. **Duck typing for grid types**: `getattr(grid, 'dy', dx)` 패턴
4. **JIT safety**: Cell sizes는 Python float로 추출 → NamedTuple
5. **Evidence before defaults**: 파라미터 변경은 sweep 실측 후 결정

---

### Support Lane Model (2026-04-10 도입)

| Lane | Status | Docs |
|------|--------|------|
| Uniform Cartesian Yee | **Reference** (claims-bearing) | `docs/guides/reference_lane_contract.md` |
| Non-uniform graded-z | Shadow | `docs/guides/support_matrix.md` |
| Distributed multi-GPU | Experimental | `docs/guides/support_matrix.md` |
| SBP-SAT subgridding | Experimental | `docs/guides/support_matrix.md` |
| Floquet/Bloch | Experimental | `docs/guides/support_matrix.md` |

---

### Previous Session (2026-04-07) — Archived

<details>
<summary>Session 2 summary (click to expand)</summary>

#### Completed
1. Non-uniform runner 15% error 해결 (center probe → edge probe)
2. Port sigma fix (anisotropic cell): `d_parallel/(Z0·d_perp1·d_perp2)`
3. CPML axis-aware refactor: `CPMLAxisParams` 4-profile
4. Port/Source sigma 통합: shared helper
5. Geometry rasterization 통합: `rfx/geometry/rasterize.py`
6. Physical absolute coordinates 통일
7. PR #24 close

#### VESSL Runs
- 369367232429: CPML reflectivity sweep — ALL PASS
- 369367232419: Physics validation — edge probe 일치

#### Axis-Dependent Audit (12건)
- #1-8, 10-12: Fixed
- #9: Far-field dS — Pending

</details>
