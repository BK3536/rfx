# 2026-03-30: Branch/junction-normal waveguide port design note

## Current state
The branch now supports:
- single-port waveguide calibration with stabilized S21
- direction-aware two-end-port scattering
- boundary-aperture multiport on the x-end planes

What it does **not** support yet is changing the port normal axis away from x.
That is the remaining gap for orthogonal branch/junction ports.

## Minimal sound next step
The smallest correctness-first extension is **axis-normal local-frame support**,
not arbitrary port normals.

That means a port should eventually carry:
- `normal_axis in {x, y, z}`
- inward/outward sign
- two transverse aperture ranges on the plane orthogonal to that normal
- a local `(u, v, n)` frame used consistently for:
  - mode profiles
  - source injection
  - modal voltage/current extraction
  - incident/outgoing wave definitions

## Why this must be a dedicated work package
The current low-level implementation still assumes:
- a fixed x-index plane
- y/z apertures only
- global forward/backward waves mapped from ±x

So supporting branch/junction normals is not a one-line extension. It is a
local-frame refactor across the port model and its measurement/injection logic.

## Explicit recommendation
Do **not** claim branch/junction-normal support until:
1. the port geometry is axis-normal aware
2. local tangential field mapping is generalized
3. at least one non-x-normal boundary case has dedicated regression coverage

## Suggested implementation order
1. add `normal_axis` and local-frame mapping to `WaveguidePort`
2. generalize local modal field profiles to `(u, v)` instead of hardcoded y/z
3. generalize modal V/I extraction to any supported normal axis
4. generalize scattering assembly without changing one-driven-port-at-a-time
5. add one orthogonal branch/junction test case
