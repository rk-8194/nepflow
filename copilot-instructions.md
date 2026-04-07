# Copilot Instructions

## Overview

Python-based HPC workflow system for atomistic datasets: generate structures → select representatives → run VASP DFT → train NEP models → validate with GPUMD.

Key aspects:
- Runs on HPC clusters via Slurm (not a local script)
- Resumable and self-submitting
- Modular with persistent state tracking

## Workflow Stages

1. **Generate** base crystals and variants (supercells, strain, rattled, vacancies, substitutions, defects)
2. **Densify** into candidate dataset
3. **Select** sparse representative subset (PCA, farthest-point sampling, optional external tools)
4. **Run VASP** adaptively: generate inputs → submit → parse energies/forces
5. **Build NEP dataset** from results
6. **Train NEP** models reproducibly
7. **Validate** with GPUMD
8. **Control** workflow: detect incomplete work, submit jobs, self-resubmit until complete

## Tech Stack

**Language**: Python ≥ 3.11

**Core libraries**: `ase`, `numpy`, `pandas`, `scikit-learn`, `sqlite3`, `pydantic`, `jinja2`, `pathlib`, `subprocess`

**Optional**: `typer`/`click` for CLI

**External dependencies** (user-installed, configurable paths): VASP, GPUMD, NEP executable, optional NEPTrainKit, Slurm utilities

*Note: In debug mode, external dependencies are not required; .fake files are generated instead.*

## Design Principles

1. **Resumable by design**: Write persistent state after each stage; restart safely after walltime expiry, failure, or interruption
2. **Idempotent execution**: Detect completion via hashes/metadata; skip rather than redo
3. **HPC-first**: Designed for Slurm; handle time limits, partial failures, and heterogeneous environments
4. **Separation of concerns**: Keep science logic separate from scheduling; use clean modules
5. **Explicit state tracking**: SQLite or structured JSON logs with structure IDs, derivations, DFT/training/validation status, Slurm job IDs, timestamps, file locations

## Workflow Execution

**Controller stage**: Inspect project state → detect incomplete tasks → submit required jobs via `sbatch` → use Slurm dependencies where applicable → optionally resubmit controller for continued orchestration → exit cleanly.

Do not design as a daemon; this is batch orchestration, not a persistent service.

**Slurm job generation**: Use Jinja2 templates, parameterize by config (account, partition, nodes, tasks, GPUs, memory, module commands, executable paths, environment setup).

## Configuration

**Format**: INI-format `.config` files (human-editable) with sections for each stage.

**Content**:
- Project metadata, compositions, crystal prototypes
- Perturbation & defect settings
- Selection algorithm parameters
- VASP, NEP training, GPUMD settings
- Slurm settings (separate per stage)
- Executable paths, module load commands
- External API settings (e.g., Materials Project API key)

**Key principles**:
- No magic constants in code; all tunable parameters configurable
- Sensitive data (API keys) stored in config with fallback to environment variables
- User-editable; comments explain all options
- Clear section organization (e.g., `[generation]`, `[materialsproject]`, `[vasp]`)

---

## Logging & Output

**Hierarchy**: Use hierarchical logger names following the module structure:
- `nepflow` — root logger (configured in main entry point)
- `nepflow.workflow` — workflow controller
- `nepflow.generate` — structure generation stage
- `nepflow.select` — selection stage
- etc.

**Format**: Include timestamp, logger name, level, message:
```
2026-04-07 14:06:21 - nepflow.generate - INFO - Structure generation complete
```

**Output visibility**:
- Both console (stderr, realtime) and file handler (rotating logs)
- Report key results with clear structure: sections with dashes, bullet points, formatted values
- Use visual separators (lines of `=` or `-`) for readability in both console and log files
- Example: When fetching lattice parameters, display formatted table with material ID, formula, lattice parameters

**Error handling**: Provide actionable error messages (e.g., "Install with: pip install mp-api") to guide user resolution

---

## Development Practices

**Integration over isolation**: Features should be integrated into existing workflow stages rather than as standalone scripts. External integrations (APIs, tools) are called from within stages to produce results automatically as part of the pipeline.

**Documentation first**: When integrating external libraries or APIs:
1. Read and understand the official documentation before writing code
2. Inspect actual objects (docstrings, dir(), attributes) to validate assumptions
3. Test API calls interactively before integration
4. Include helpful error messages and installation instructions

**User environment**: 
- Assume user will run the program in their own environment
- Support multiple Python environments (conda, venv, etc.)
- Provide clear feedback on missing dependencies with installation commands
- Allow storing configuration and credentials in config files or environment variables

**Caching and efficiency**:
- For external API calls, implement local caching with deterministic hashing
- Avoid repeated calls for identical queries
- Cache results in standard locations (e.g., `~/.cache/`) to benefit all projects

---

## Project System

**Isolation guarantees**:
- Separate config files (one per project)
- Separate state database (SQLite or JSON per project)
- Separate directory tree for structures, VASP runs, NEP models, validation
- Separate Slurm job names/tags for concurrent execution (e.g., `nepflow-project1-generate`, `nepflow-project2-select`)
- No shared mutable state; multiple projects can run concurrently on same cluster without interference

**Project identification**:
- User specifies project name via CLI flag (`--project myproject`) or environment variable
- Project name determines config path, state database path, output directories
- All state tracking, file I/O, Slurm submissions scoped to that project

**Multi-project workflow**:
- User can initialize, run, pause, resume multiple projects independently
- Each project resumes from its own persistent state
- Controller stage only tracks/submits jobs for its own project
- Log files isolated per project for debugging

**Implementation**:
- Config lookup: `config/{project_name}.yaml` or `config/{project_name}.toml`
- State database: `state/{project_name}.db` or `state/{project_name}.json`
- Outputs: `project_{project_name}/` directory tree or tagged subdirs within shared structures/vasp/nep/gpumd
- Slurm job naming: Include project name in job name for easy tracking

**Project Config Format** (INI `.config` file):
```ini
[project]
name=project_name
description=Description of the project
status=initialized

[paths]
structures_path=structures
vasp_path=vasp
nep_path=nep
gpumd_path=gpumd
reports_path=reports

[materialsproject]
api_key=YOUR_API_KEY_HERE

[generation]
elements=W,Cr
crystal_structures=bcc,fcc,hcp

[vasp]
enabled=true

[nep]
enabled=true

[gpumd]
enabled=true

[slurm]
enabled=false
```

---

## Debug Mode

**Purpose**: Test and verify workflow locally without VASP, GPUMD, NEP, or Slurm installed.

**Behavior**: When enabled, compute-intensive stages generate `[output_name].fake` files instead of invoking external programs:
- VASP run → generate fake `OUTCAR`, `vasprun.xml` with synthetic energies/forces
- NEP training → generate fake `.nep` model file with metadata
- GPUMD validation → generate fake validation metrics
- Slurm submission → log job calls, skip actual sbatch invocation

**Use cases**:
- Local development and testing
- Debugging workflow logic without HPC access
- Verifying state tracking and resumption
- Validating config parsing and schema

**Implementation**: Add `debug: true` to config or command-line flag. Each external call wraps real execution with mock path that checks debug mode before subprocess invocation.

---

## Stage Requirements

**Structure Generation** (ASE-based):
- Query Materials Project for lattice parameters of selected element/structure combinations
  - Uses `mp-api` package (installable via pip)
  - Supports caching to local `~/.cache/nepflow/mp` to avoid repeated API calls
  - Filters by crystal system and space group to match user-specified structures (bcc, fcc, hcp, etc.)
  - API key stored in config `[materialsproject]` section or `MP_API_KEY` environment variable
  - Logs detailed lattice parameters (a, b, c, α, β, γ, volume) with formatting for readability
- Crystal prototypes, supercells, strain (isotropic/anisotropic), shear
- Rattling, vacancies, substitutions, interstitials, composition sweeps
- All structures must carry provenance metadata (how generated)

**Selection**:
- Descriptor extraction, PCA, farthest-point sampling
- Optional external CLI tool wrapping (e.g., NEPTrainKit)
- Modular & gracefull fallback if tools missing; no hard GUI dependencies

**VASP**:
- Generate inputs, submit, detect completion/convergence/failure
- Parse energies, forces, optionally stress/virials
- Retry on failure with adjusted settings (configurable)
- Preserve failed outputs for debugging; never silently overwrite
- In debug mode: generate fake OUTCAR/vasprun.xml with synthetic data

**NEP Training**:
- Compile data into training format reproducibly
- Link each model to exact dataset, config, code version
- Archive outputs, store run metadata
- In debug mode: generate fake .nep model file with metadata

**GPUMD Validation**:
- Formal workflow stage, not ad hoc
- Generate inputs, submit, parse outputs, collect metrics
- Compare across candidate models
- Store results structurally
- In debug mode: generate fake validation metrics and results

---

## Project Layout

```
project/
  config/              # YAML/TOML configuration files
  structures/
    seeds/             # Base crystal structures
    generated/         # All derived variants
    selected/          # After selection
  vasp/
    jobs/              # Input files
    results/           # Parsed outputs
  nep/
    datasets/          # Training data
    runs/              # Trained models & metadata
  gpumd/
    validation/        # Validation runs & results
  state/               # Persistent workflow state (SQLite or JSON)
  logs/                # Execution logs
  templates/           # Jinja2 templates for Slurm scripts, config files
  reports/             # Generated reports
```