# Hybrid Adjoint Worktree Closure Handoff

Status: reviewer-facing closure summary
Date: 2026-04-24
Branch: `plan/hybrid-adjoint-custom-vjp`

## What this branch completed

This branch completed the currently approved hybrid-adjoint roadmap slices up
through the current **Phase IV completion-candidate state**.

### Phase II — Practical Workflow Coverage
Completed as a bounded workflow-widening slice:
- CPML zero-sigma dielectric source/probe topology hybrid support
- one-excited + optional one-passive lumped-port proxy optimize hybrid support
- support-contract / fallback / cookbook alignment

### Phase III — Strategy B / Scale-up
Completed as a bounded source/probe scale-up slice:
- Strategy B benchmark contract
- bounded PEC/CPML source-probe Strategy B replay path
- explicit fencing for unsupported Strategy B families

### Phase IV — Productization / Default Policy
Completed in branch scope as a policy/productization pass:
- `pure_ad` remains the public default
- `auto` is the bounded support-inspected opt-in path
- `hybrid` remains strict opt-in
- Strategy B remains outside the public optimizer policy matrix
- public/internal docs were aligned to that policy

## Current public optimizer policy

For `optimize()` and `topology_optimize()`:
- `adjoint_mode="pure_ad"` — public default
- `adjoint_mode="auto"` — bounded opt-in path; inspect support first, use hybrid only on landed supported families, otherwise fall back to `pure_ad`
- `adjoint_mode="hybrid"` — strict opt-in; unsupported cases should raise

## What is intentionally still out of scope

This branch does **not** claim:
- a `pure_ad -> auto` public default flip
- Strategy B promotion into the public optimizer policy matrix
- broader Strategy B topology / port / NTFF support
- new physics-family widening beyond the already landed bounded subsets
- a new macro phase beyond the current Phase IV closure work

## Why there is no Phase V yet

There is currently **no defined Phase V** in the roadmap.

The latest roadmap state says:
- Phase II and Phase III are completed reference artifacts
- Phase IV is a completion-candidate reference artifact in this worktree
- no immediate new macro phase is required unless new evidence materially changes the workflow, policy, or replay boundary

In other words, the remaining gap after Phase IV was branch closure / reviewer
handoff quality, not another feature phase.

## Internal vs tracked artifacts

The detailed phase ledgers and execution records under `.omx/` are internal
planning/execution artifacts. They are useful for maintainers but are not the
reviewer-facing source of truth because `.omx/` is gitignored in this repo.

This document exists to translate the final branch state into a tracked summary
that reviewers can read directly from the branch.

## Docs-site verification status

I searched the repository for a local public-docs build surface (for example:
`package.json`, `astro.config.*`, `mkdocs.yml`, or an explicit docs build
command) and did **not** find one in this worktree.

The repository does contain public docs source files (`docs/public/*.mdx`) and
an architecture note that treats public-docs publishing as a separate deploy
surface. Based on the current repo contents, docs-site validation should be
considered **external/pipeline-owned**, not a locally runnable build step in
this worktree.

I also ran:

```bash
python scripts/check_public_docs_sync.py --format text
```

That source-drift check reported content drift for the public pages changed in
this branch (`support-boundaries.mdx`, `changelog.mdx`, `inverse-design.md`,
and `migration.md`). That is expected at this stage: the source-repo worktree
is updated, while the downstream gitops deploy snapshot has not yet been
re-exported and rebuilt.

So the remaining docs-site work is **not** a missing feature in this worktree;
it is the separate gitops export/build follow-up owned by the public-docs
deployment workflow.

## Reviewer handoff summary

From tracked files alone, the branch now answers:
- what was completed: Phases II–IV branch scope above
- what remains out of scope: default flip, Strategy B public promotion, broader support widening
- why there is no Phase V yet: no new macro boundary is justified by current evidence
