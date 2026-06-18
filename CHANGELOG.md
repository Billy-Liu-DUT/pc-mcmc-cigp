# Changelog

## 0.1.0

Initial open-source cleanup.

- Added installable `pc_mcmc_cigp` package.
- Added reaction primitives and atom-balanced candidate generation.
- Added mass-action `MechanismEngine` with SciPy/RK4 solver fallback.
- Added spike-and-slab PC-MCMC sampler.
- Added multi-chain MCMC helper with R-hat and ESS diagnostics.
- Added sklearn-style `CIGPRegressor`.
- Added acquisition functions: EI, GWU, DH, and PC-EI.
- Added HBr mechanism discovery and epoxidation optimization benchmarks.
- Added paper-style experiment scripts and plotting utilities.
- Added PPT log extraction for Fig.4 CIGP vs Standard BO comparison.
- Added Chinese usage/API/journal-strategy documentation.
