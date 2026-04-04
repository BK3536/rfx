# Non-Uniform Mesh

One of the biggest recent `rfx` upgrades is support for **graded z meshes**
through `dz_profile`.

## Why it matters

For printed RF structures:

- lateral dimensions may be tens of millimetres,
- substrate thickness may be only ~1 mm,
- forcing one uniform fine cell size everywhere wastes memory badly.

`rfx` addresses this with:

- uniform `dx`, `dy`
- non-uniform `dz`

This is especially useful for:

- microstrip lines
- patch antennas
- thin dielectric stacks
- PCB-like layered structures

## `dz_profile`

`dz_profile` is a 1-D array of physical z-cell sizes:

```python
import numpy as np
from rfx import Simulation

dz_profile = np.concatenate([
    np.full(6, 0.0002667),  # fine cells through substrate
    np.full(20, 0.0015),    # coarser air cells
])

sim = Simulation(
    freq_max=4e9,
    domain=(0.08, 0.06, 0.0),   # z comes from dz_profile
    boundary="cpml",
    dx=5e-4,
    dz_profile=dz_profile,
)
```

## Auto-configured non-uniform z

`auto_configure()` can enable non-uniform z automatically when it detects thin
z-features:

```python
from rfx import Box, auto_configure

geometry = [
    (Box((0, 0, 0), (0.06, 0.06, 0.0016)), "fr4"),
    (Box((0, 0, 0), (0.06, 0.06, 0.0)), "pec"),
]

materials = {
    "fr4": {"eps_r": 4.4, "sigma": 0.025},
    "pec": {"eps_r": 1.0, "sigma": 1e10},
}

cfg = auto_configure(
    geometry,
    freq_range=(1e9, 4e9),
    materials=materials,
    accuracy="standard",
)

print(cfg.summary())
```

When enabled, `cfg.to_sim_kwargs()` includes `dz_profile`.

## Best current use

Use non-uniform mesh when:

- substrate or dielectric layer thickness is the dominant resolution bottleneck
- z refinement matters much more than x/y refinement
- you want a practical thin-substrate workflow without global fine meshing

## Relationship to SBP-SAT subgridding

`rfx` also has SBP-SAT subgridding work, but for day-to-day PCB / patch /
thin-substrate workflows, graded-z meshing is often the simpler and more
practical choice.

## Related features

- [Quick Start](quickstart.md)
- [Simulation API](simulation_api.md)
- [Validation](validation.md)
