# rfx Guide Redirect

`docs/guide/` is no longer the canonical authoring tree for public `rfx`
documentation.

## Use these paths instead

- `docs/public/guide/` — canonical public guide source
- `docs/agent/` — canonical public AI-agent docs
- `docs/guides/public_docs_architecture.md` — ownership, sync, and CI rules

## Public entrypoints

- live docs: [remilab.ai/rfx](https://remilab.ai/rfx/)
- public guide source index: [`../public/guide/index.md`](../public/guide/index.md)
- public landing source: [`../public/index.mdx`](../public/index.mdx)

## Why this file still exists

It is kept as a single redirect-style entrypoint so older local references to
`docs/guide/` fail gracefully while the canonical source lives in
`docs/public/guide/`.
