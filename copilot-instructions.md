# Copilot Instructions

## Project Overview

This project is a Python-based HPC workflow system for generating atomistic datasets, selecting representative structures, running DFT calculations with VASP, training NEP models, and validating them with GPUMD.

The system is intended to run on remote HPC clusters using Slurm. It must be resumable, restartable, and able to self-submit follow-up jobs when a workflow stage is incomplete or when further work is required.

This is **not** a single linear script. It is a modular workflow/orchestration program with persistent state tracking.

---

## Core Goals

The program should support the following end-to-end workflow:

1. Generate base crystal structures.
2. Create many derived variants:
   - different cell sizes / supercells
   - strained / deformed cells
   - rattled structures
   - vacancies
   - substitutions / alloy variations
   - interstitials and other defects where applicable
3. Build a dense candidate dataset from those variants.
4. Reduce the dense dataset to a sparse but representative subset using:
   - PCA
   - farthest point sampling / feedforward parameter selection
   - external tools such as NEPTrainKit if a usable CLI exists
5. Run VASP calculations adaptively on selected structures to obtain:
   - total energies
   - forces
   - optionally virials / stress if available and desired
6. Compile the resulting structures and labels into NEP training datasets.
7. Launch NEP training jobs.
8. Run GPUMD validation jobs on the trained potential.
9. Track progress persistently and self-submit additional Slurm jobs until the workflow is complete.

---

## Language and Technology Requirements

Use **Python** as the primary implementation language.

Recommended ecosystem:

- `python >= 3.11`
- `ase` for structure generation and manipulation
- `numpy`
- `pandas`
- `scikit-learn` for PCA and related selection logic
- `sqlite3` or a lightweight persistent database for state tracking
- `pydantic` or dataclasses for configuration models
- `jinja2` for generating Slurm scripts and input templates
- `subprocess` for calling external programs
- `pathlib` for filesystem work
- optional `typer` or `click` for CLI entry points

External tools are expected to be installed by the user:

- VASP
- GPUMD
- NEP executable
- optional NEPTrainKit / related selection tools
- Slurm utilities such as `sbatch`, `squeue`, `sacct`

Do **not** assume these tools are installed in a fixed location. Their paths must be configurable.

---

## Design Principles

### 1. Resumable by Design

The workflow must always be restartable.

Every stage should write outputs and status metadata to disk so that the program can resume after:

- walltime expiry
- node failure
- manual interruption
- partial job completion
- Slurm requeue / resubmission scenarios

Never rely on a long-lived in-memory process to preserve workflow state.

---

### 2. Idempotent Stage Execution

Each stage must be safe to rerun.

If a structure, calculation, training run, or validation task is already complete, the program should detect that and skip or verify it rather than redoing it blindly.

Use stable identifiers, hashes, or metadata records to detect duplicates and completion.

---

### 3. HPC-First Architecture

This code is intended for remote HPC execution under Slurm.

Design around the reality that:

- jobs have time limits
- jobs can fail partially
- cluster environments differ
- modules and executable paths vary by system
- resource requests must be configurable per stage

The program should generate and submit Slurm jobs rather than assuming local execution.

---

### 4. Separation of Concerns

Keep scientific logic separate from scheduler logic.

Examples:

- structure generation code should not contain Slurm submission logic
- VASP input generation should not directly decide queue policy
- selection algorithms should not depend on filesystem layout more than necessary

Use clean modules with explicit responsibilities.

---

### 5. Explicit State Tracking

Track workflow state persistently.

Preferred options:

- SQLite database
- or structured JSON/JSONL records if kept simple

The state model should record, at minimum:

- structure IDs
- parent-child relationships between derived structures
- generation parameters
- selection status
- DFT status
- training dataset membership
- training run status
- validation run status
- Slurm job IDs where relevant
- timestamps
- file locations
- error messages / failure reasons

---

## Expected Workflow Structure

The codebase should conceptually support stages like:

1. `generate`
2. `densify`
3. `select`
4. `prepare-vasp`
5. `run-vasp`
6. `parse-vasp`
7. `build-nep-dataset`
8. `train-nep`
9. `validate-gpumd`
10. `report`
11. `controller`

The `controller` stage is especially important. It should inspect the project state, determine unfinished work, submit required Slurm jobs, and optionally submit a follow-up controller job if the workflow is not yet complete.

---

## Self-Submission / Controller Behavior

The program must support self-submission under Slurm.

Preferred behavior:

- inspect current project state
- detect incomplete tasks
- submit required child jobs with `sbatch`
- use Slurm dependencies where appropriate
- optionally submit a new controller job to continue orchestration later
- exit cleanly after submission

Do **not** design this as a daemon or an always-running service.

This is a batch workflow orchestrator, not a persistent server.

---

## Slurm Requirements

Slurm job scripts must be generated from templates and parameterized by user config.

Users should be able to provide:

- account
- partition
- qos
- time limit
- node count
- task count
- GPU count
- memory
- module load commands
- executable paths
- environment activation commands

Keep these details in config files, not hardcoded in Python source.

Support stage-specific resource settings, since:
- structure generation may be lightweight
- selection may be CPU-only
- VASP may require MPI and possibly GPUs depending on build
- NEP training may have different requirements
- GPUMD validation may require GPUs

---

## Configuration Requirements

Use a human-editable config format such as YAML or TOML.

The config should include:

- project metadata
- compositions / species
- crystal prototypes to generate
- perturbation and defect settings
- selection settings
- VASP settings
- NEP training settings
- GPUMD validation settings
- Slurm settings
- executable paths
- module/environment setup commands

All tunable scientific and HPC parameters must be configurable.

Avoid magic constants in source code.

---

## Structure Generation Expectations

Use ASE where practical.

The structure generation layer should support:

- crystal prototype construction
- supercell expansion
- random displacements / rattling
- isotropic strain
- anisotropic strain
- shear deformation
- vacancies
- substitutions
- interstitial insertion where meaningful
- composition sweeps where needed

All generated structures should carry metadata describing exactly how they were produced.

Never generate unlabeled structures without provenance.

---

## Dataset Selection Expectations

Selection must reduce a dense candidate pool to a representative subset.

Support at least:

- descriptor extraction
- PCA
- FPS / greedy farthest-point-like selection
- optional external CLI tool invocation if the user has installed it

The implementation should be modular so that selection backends can be swapped.

If NEPTrainKit or related tooling is used, wrap it cleanly and fail gracefully if it is not installed.

Do not hardcode dependency on GUI-only workflows.

---

## VASP Launcher Expectations

The VASP stage must be adaptive and resumable.

The code should be able to:

- generate VASP input files
- submit jobs
- detect completion / non-convergence / failure
- parse energies and forces
- optionally parse stress / virials if required
- retry or escalate settings in a controlled way if configured

Examples of adaptive behavior may include:

- retry with adjusted electronic settings
- retry failed jobs up to a configured limit
- distinguish convergence failure from scheduler failure
- preserve previous outputs for debugging

Do not silently overwrite failed runs.

---

## NEP Training Expectations

The NEP training stage should:

- compile structure/energy/force data into the required training format
- write training inputs reproducibly
- launch the NEP executable
- store run metadata
- archive outputs
- track model provenance

Every trained potential should be linked back to:
- the exact dataset used
- the exact config used
- the exact code version if possible

Reproducibility matters.

---

## GPUMD Validation Expectations

Validation should be treated as a formal workflow stage.

Support:
- validation input generation
- job submission
- output parsing
- metric collection
- comparison across candidate NEP models

Validation results should be stored structurally, not only as ad hoc text logs.

---

## Filesystem and Project Layout

Prefer a clean project layout such as:

```text
project/
  config/
  structures/
    seeds/
    generated/
    selected/
  vasp/
    jobs/
    results/
  nep/
    datasets/
    runs/
  gpumd/
    validation/
  state/
  logs/
  templates/
  reports/