# 2026-04-01 Session Handoff v2

## Immediate TODO (next session)

### 1. WirePort S-param Extraction (CRITICAL)
Current WirePort can excite fields but has no S11 extraction.

Implementation plan (read rfx/probes/probes.py SParamProbe pattern):
- V_port = sum(Ez[wire_cells]) * dx (total voltage across wire)
- I_port = H circulation around wire (Ampere's law)
- V_inc = source waveform (same as LumpedPort)
- Distribute across N cells, accumulate DFTs like update_sparam_probe
- Integrate in api.py run() for ports with extent

### 2. 3D Visualization (rfx/visualize3d.py)
- plot_geometry_3d: render domain + shapes + ports
- plot_field_3d: isosurface/volume of field components
- save_field_vtk: VTK export for ParaView
- Use pyvista (optional dep), fallback matplotlib 3D

### 3. Re-run Patch Antenna on GPU
With WirePort S-param working:
- Expect S11 dip near 2.4 GHz
- Far-field via NTFF
- 3D visualization of geometry + fields
- Compare with analytical

## Session Stats
- Tests: 134 → 233 (+99)
- Commits: 45+
- xfails: 3 → 0
- Major features: normalization, passivity, oblique TFSF, CFS-CPML,
  subpixel, eigenmode, conformal PEC, SBP-SAT subgridding (1D+2D+3D),
  Kerr nonlinear, thin wire, WirePort, batch simulation, RCS pipeline,
  objective library, GPU benchmark (1310 Mcells/s)

## Git: main @ b326a18 (or later patch antenna commits)
## VESSL: #369367231330 completed (patch antenna WIP on GPU)
