# rfx Attribution & Licenses

## Core FDTD Algorithm
- **Yee algorithm**: K. S. Yee, "Numerical solution of initial boundary value problems involving Maxwell's equations in isotropic media", IEEE TAP, 1966.
- **License**: Public domain (fundamental algorithm, no copyright)

## PML / CPML
- **CFS-CPML**: J. A. Roden and S. D. Gedney, "Convolution PML (CPML): An efficient FDTD implementation of the CFS-PML for arbitrary media", Microwave and Optical Technology Letters, 27(5), 2000.
- **License**: Algorithm is public; implementation is original rfx code (MIT)

## SBP-SAT Subgridding
- **Reference**: B. Cheng, R. E. Diaz, et al., "Toward the Development of a 3-D SBP-SAT FDTD Method: Subgridding Implementation", IEEE Transactions on Antennas and Propagation, 2025 (DOI: 10.1109/TAP.2025.10836194)
- **License**: Algorithm from published paper; implementation is original rfx code (MIT)
- **Note**: SBP (Summation-By-Parts) operators and SAT (Simultaneous Approximation Terms) are general numerical methods from the finite difference community. Cheng et al. applied them to FDTD subgridding.

## Harminv / Matrix Pencil Method
- **FDM reference**: V. A. Mandelshtam and H. S. Taylor, "Harmonic inversion of time signals and its applications", J. Chem. Phys. 107, 6756, 1997.
- **MPM reference**: T. K. Sarkar and O. Pereira, "Using the Matrix Pencil Method to Estimate the Parameters of a Sum of Complex Exponentials", IEEE Antennas and Propagation Magazine, Feb 1995.
- **License**: Both are published algorithms; implementation is original rfx code (MIT)
- **Note**: Meep uses Harminv (Steven G. Johnson, MIT License) which implements FDM. rfx implements MPM independently — different algorithm, same purpose.

## Subpixel Smoothing
- **Reference**: A. Farjadpour et al., "Improving accuracy by subpixel smoothing in the finite-difference time domain", Optics Letters, 31(20), 2006.
- **License**: Algorithm from published paper; implementation is original (MIT)

## Near-to-Far-Field Transform
- **Reference**: A. Taflove and S. C. Hagness, "Computational Electrodynamics: The Finite-Difference Time-Domain Method", 3rd edition, Chapter 8.
- **License**: Standard FDTD technique; implementation is original (MIT)

## Thin Wire Model
- **Reference**: R. Holland, "Finite-Difference Analysis of EMP Coupling to Thin Struts and Wires", IEEE EMC, 1981.
- **License**: Published algorithm; implementation is original (MIT)

## Software Dependencies
- **JAX** (Google): Apache 2.0 License
- **NumPy**: BSD 3-Clause License
- **SciPy**: BSD 3-Clause License
- **matplotlib**: PSF License (BSD-compatible)
- **pyvista** (optional): MIT License

## rfx License
rfx is released under the **MIT License**.

## Development
- Primary development: Claude Opus 4.6 (Anthropic) + Codex (OpenAI)
- Supervision: Prof. Byungkwan Kim, REMI Lab, Chungnam National University
