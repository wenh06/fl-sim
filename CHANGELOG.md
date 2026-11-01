# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## [v0.2.0](https://github.com/wenh06/fl-sim/compare/v0.1.0...v0.2.0) — 2026-11-01

### Added

- **Parallel execution layer.** The clients' local steps (evaluate → local update → build upload) can run concurrently on a pluggable executor backend while all server-side state changes stay on the main thread in the fixed sampled-client order:
  - `executor="threads"` / `num_workers` on the server config (plus `FL_SIM_EXECUTOR` / `FL_SIM_NUM_WORKERS` environment overrides); the default `sequential` backend reproduces the pre-parallel behavior bitwise (#48–#53).
  - Typed message protocol: client uploads are built as `ClientToServer` objects and downlinks as `ServerToClient`, with named-parameter exchange (`model_state`) decoupled from `nn.Module` state dicts (#31, #38, #50, #51; adopted by IFCA and FedCR #39, #40).
  - Reproducibility levels spelled out and test-enforced: aggregation arithmetic is bitwise identical across backends; `rng_mode="per_client"` on the client config makes the full run (including final weights) bitwise identical across backends, parallelism degrees and repeats (#53).
  - Benchmarks and an honest account of expected speedups: up to ~1.2x on small CPU workloads, 1.4x+ on compute-heavier CPU models, no benefit on a single GPU (#53, #55).
  - The training loop was restructured into four explicit phases as the foundation for the above (#33).
- **Algorithm theory documentation** for the PFL family, the skipping and primal-dual algorithms, and the operator-splitting machinery (including FedDR), from the thesis wording; zh_CN translations included (#42-#44, #46, #47).
- **Auto-generated algorithm configuration reference** — one table per algorithm, kept in sync by `docs/gen_config_reference.py` (#45, #57).
- **Tutorials and how-to guides.** The documentation gained a full learning track: Quickstart, Core concepts, four task-oriented tutorials (first controlled experiment, CLI experiment matrices, visualization, parallel & reproducible runs) and three how-to guides (custom algorithm, custom dataset, custom optimizer); all shown code is embedded via `literalinclude` from runnable scripts exercised in CI, with zh_CN translations (#56, #58, #59, #61, #62).
- Docker: the package version inside the image is derived from the git tag (#29).

### Changed

- **Python support: 3.11 / 3.12 / 3.13 / 3.14.** Python 3.10 (EOL) is dropped; CI tests 3.11, 3.13 and 3.14; `requires-python >= 3.11` (#63).
- The config system migrated to pydantic v2 (#30).
- `fl_sim/nodes` is now a package (`server.py` / `client.py` / `configs.py`) (#36).
- The slim Docker image moves to `python:3.13-slim`; the torch-1.13.1-based image is retired (torch 1.13 supports at most Python 3.10) (#63).
- The tikz Sphinx extension now comes from the maintained fork (`github.com/wenh06/tikz2`), fixing the gettext build under Sphinx 9 (#64).
- Dependencies are declared in `pyproject.toml` only; the standalone `requirements.txt` is retired (#32).

### Fixed

- **APFL**: server-side aggregation happened on broadcast rounds instead of period-end rounds, so aggregated parameters could be computed before the corresponding client uploads arrived (#54).
- CI: the TeX Live setup retries on transient CTAN mirror failures (#35); tutorial JSON-log fixtures are pinned by file name instead of glob order (#55).

### Removed

- Python 3.10 support (CI matrices, classifiers, Docker base image).
- The torch-1.13.1-cu116 Docker image.
