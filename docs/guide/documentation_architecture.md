# Documentation Architecture

This page defines the durable documentation structure for `rfx`.

## Source-of-truth boundaries

| Area | Purpose | Authoring rule |
|---|---|---|
| `docs/guide/` | Stable user-facing guides, tutorials, and maintainer docs | Author here first |
| `docs/agent/` | Stable public AI-agent workflows for `rfx` | Author here first |
| `docs/api/` | Generated API reference | Never hand-edit generated HTML |
| `docs/research_notes/` | Chronological notes, audits, handoffs, planning | Internal / historical only |
| `docs/codex_specs/`, `docs/superpowers/` | Verification specs and planning artifacts | Internal support docs, not public entry points |

`infra/remilab-sites-gitops/.../seed-pages/rfx` is the **deployment snapshot**, not the
canonical authoring home.

## Public topic hierarchy

The public information architecture should stay grouped by user task, not by
file age or implementation history.

### 1. Getting Started
- Installation
- Quick Start
- First Patch / first end-to-end success path

### 2. Modeling & Setup
- API Reference
- Materials & Geometry
- Sources & Ports
- Probes & S-Parameters
- Non-Uniform Mesh
- Waveguide Ports
- Floquet Ports

### 3. Analysis & Validation
- Validation
- Convergence Study
- Far-Field & RCS
- Antenna Metrics
- Visualization & Analysis
- Solver Comparison

### 4. Design & Optimization
- Inverse Design
- Topology Optimisation
- Parametric Sweeps
- Material Fitting
- RF backend / production workflow guidance
- Full design tutorials such as patch and microstrip filter

### 5. Advanced & Research Methods
- Advanced Features overview
- Conformal PEC
- Subgridding
- Gradient behavior / numerical caveats

### 6. AI Agent Guide
- Agent overview
- Automatic simulation configuration
- Prompt templates
- Automated design workflows

### 7. Project & Maintainer
- Migration
- Changelog
- Contributing
- Documentation architecture / ownership

## Naming and route rules

- Public route slugs use **kebab-case**.
- Do not keep the same concept alive in both underscore and kebab-case forms.
- Use `.md` unless JSX / component embedding is actually needed; use `.mdx` only
  when there is a concrete rendering reason.
- One page should answer one job-to-be-done:
  - concept/reference,
  - tutorial/workflow,
  - maintainer policy.
- If a page is experimental, say so explicitly in the title, intro, or status
  note.

## Maintenance workflow

1. **Author in `research/rfx` first.**
2. **Update the matching doc class**:
   - user docs → `docs/guide/`
   - AI-agent docs → `docs/agent/`
   - internal history → `docs/research_notes/`
3. **Sync the public snapshot** into `infra/remilab-sites-gitops` using the same
   slug and topic bucket.
4. **Update navigation and landing pages** whenever a page is added, renamed, or
   retired.
5. **Keep claims aligned with evidence**:
   - new feature → docs + example/tests + changelog
   - experimental feature → docs must say experimental
   - generated API output → regenerate, do not hand-edit

## Review triggers

Review the public hierarchy whenever one of these happens:

- a new user-facing solver feature lands,
- a public route is added or renamed,
- README capabilities materially change,
- remilab.ai and repo docs diverge on titles, counts, or supported workflows.

## Current migration priority

The highest-priority cleanup is to eliminate source/snapshot drift:

1. keep `research/rfx` as the authoring source,
2. migrate remaining public-only pages from gitops back into this repo,
3. retire duplicate page pairs that differ only by filename style,
4. keep remilab.ai sidebar groups aligned with the topic hierarchy above.
