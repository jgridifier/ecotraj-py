# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project version is the `version` field in `pyproject.toml`.

## [Unreleased]

### Added

- Logo (`docs/assets/logo.png`, made with Grok Imagine; favicons resized from it) on the README and the Pages site.
- `NOTICE` and a "Provenance and independence" section (README, Pages): built by AI agents from the papers, benchmarked against ecotraj run as a separate program, independent implementation.
- Agent-oriented docs: `AGENTS.md`, `llms.txt`, `docs/api.md`, `CONTRIBUTING.md`, `CITATION.cff`.
- `research/` folder for pre-registrations, simulation runners, and published results, so the installable package stays lean.

### Changed

- README rewritten with a 30-second quickstart, R function map, parity summary, and verified citations.
- `pyproject.toml` metadata filled in for a future PyPI release (not published).

## [0.1.0] - 2026-11-14

First public repository version (install from source). Not released to PyPI.

### Added

- Clean-room port of ecotraj 1.2.2 metrics: `define_trajectories`, lengths, speeds, angles, directionality (with optional permutation test), internal variation, metrics, convergence (three types), segment distances, trajectory distances (DSPD, SPD, Hausdorff).
- `clr`, increment `signflip_test`, and `increment_split`.
- Optional Numba kernels in `ecotrajpy.fast` (Extension 1).
- Extension 2 procedures in `ecotrajpy.ext2` and their pre-registered simulations.
- Randomized R-oracle parity suite (1,260 / 1,280 exact at rtol 1e-8; 20 documented deviations).
