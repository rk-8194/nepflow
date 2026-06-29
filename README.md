# nepflow

NEPFlow is a workflow tool for building atomistic datasets and carrying them through structure generation, selection, VASP calculations, NEP training, and GPUMD validation.

## What It Does

NEPFlow manages a project through a fixed workflow:

1. Initialize a project
2. Generate candidate structures
3. Select representative structures
4. Run VASP calculations
5. Train NEP models
6. Validate with GPUMD

The workflow is resumable. Each run picks up from the saved project stage.

## Workflow Steps

NEPFlow uses these stages in order:

1. `init` - create the project structure and configuration files
2. `generate` - build candidate structures
3. `select` - choose representative structures for calculation
4. `run_vasp` - run the VASP calculations
5. `train_nep` - train the NEP model
6. `validate` - run GPUMD validation

## Requirements

- Python 3.11 or newer
- The Python dependencies listed in `pyproject.toml`
- External tools configured for your environment, such as VASP, GPUMD, and the NEP executable
- SLURM if you plan to run on an HPC cluster

## Usage

### 1. Initialize a project

Choose a project name and create its project directory:

```bash
python nepflow.py --project myproject --init
```

This creates the project structure under `projects/project_myproject/`.
On first run, NEPFlow will prompt for any config values it needs, starting with the Materials Project API key.

### 2. Run local mode first

Before sending a project to HPC, run local mode on your local machine:

```bash
python nepflow.py --project myproject --local
```

Local mode fetches base structures and stops after seed generation. This step must be done locally before transferring the project to an HPC system.

### 3. Transfer to HPC and continue

Copy the project directory to the HPC system, then resume with the normal command:

```bash
python nepflow.py --project myproject
```

NEPFlow will continue from the stage stored in the project directory.

## Common Options

- `--local` runs the local pre-HPC setup step and stops after seed generation.
- If `config/project.config` contains an `hpc.scp_address`, local mode will offer to upload the project to `<scp_address>/projects/<project>` after seed generation. The address should point to the remote directory where `nepflow.py` lives.
- `--memory` runs VASP memory benchmarks without advancing the workflow.
- `--config` opens the project config file in Vim and exits.
- `--debug` enables verbose debug logging.
- `--stage <name>` forces the workflow to start from a specific stage.
- `--output-dir <path>` changes the base directory used for projects.

Run `python nepflow.py --help` to see the full CLI help.

## Project Layout

Each project lives under `projects/project_<name>/` and typically contains:

- `config/` for project settings
- `logs/` for runtime logs
- `state.db` for workflow state
- `.project` for the current workflow stage

The project configuration is stored in the project `config/` directory. Edit it to control:

- the elements and compositions to generate
- structure generation settings
- VASP, NEP, GPUMD, and SLURM options

## Notes

- If a run is interrupted, rerun the same command to continue from the saved state.
- For HPC runs, make sure the cluster modules, executables, and submission settings are configured correctly.
