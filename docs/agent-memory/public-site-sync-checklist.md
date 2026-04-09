# Public Site Sync Checklist

Use this when a change should appear on `https://remilab.ai/rfx/`.

## Ownership

- Source-of-truth for `rfx` public docs: **this repo**
- Deployment/snapshot hub: `infra/remilab-sites-gitops`
- Runtime route: `remilab.ai/rfx/` on shared `starlight-public`

## Checklist

1. Author docs here first.
2. Do **not** hand-edit:
   - `remilab-sites-gitops` runtime workspaces
   - `dist/`
   - deploy-host copied docs
3. Run relevant local checks in this repo.
4. Export/sync the public docs snapshot into `infra/remilab-sites-gitops`.
5. Verify gitops snapshot drift is resolved.
6. Trigger/build/deploy from gitops.
7. Verify:
   - `https://remilab.ai/rfx/` returns 200
   - expected new docs content is present

## What is *not* the fix here

- Fixing compose/runtime drift directly in this repo
- Treating gitops snapshot files as the long-term authoring home
- Editing `teaching/creative-engineering-design` for `rfx` docs

## Last reviewed
- 2026-04-09
