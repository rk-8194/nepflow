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

## Installation

1. Clone the repository and enter the project directory:

```bash
git clone git@github.com:rk-8194/nepflow.git
cd nepflow
```

2. Create and activate a Python 3.11+ virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

3. Install NEPFlow and its Python dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

4. Verify the installation:

```bash
nepflow --help
```

If you plan to use the Materials Project integration, make sure the `mp-api` dependency is installed and your API key is available in the project config or the `MP_API_KEY` environment variable.

## Usage

### 1. Initialize a project

Choose a project name and create its project directory:

```bash
nepflow --project myproject --init
```

This creates the project structure under `projects/project_myproject/`.
On first run, NEPFlow will prompt for the project configuration values it needs. Materials Project access uses the runtime `MP_API_KEY` environment variable and is never stored in the project configuration.

### 2. Run local mode first

Before sending a project to HPC, run local mode on your local machine:

```bash
nepflow --project myproject --local
```

Local mode fetches base structures and stops after seed generation. This step must be done locally before transferring the project to an HPC system.

### 3. Transfer to HPC and continue

Copy the project directory to the HPC system, then resume with the normal command:

```bash
nepflow --project myproject
```

NEPFlow will continue from the stage stored in the project directory.

## Common Options

- `--local` runs the local pre-HPC setup step and stops after seed generation.
- If `config/project.config` contains an `hpc.scp_address`, local mode will offer to upload the project to `<scp_address>/projects/<project>` after seed generation. The address should point to the remote directory where the installed `nepflow` command is available.
- `--memory` runs VASP memory benchmarks without advancing the workflow.
- `--config` opens the project config file in Vim and exits.
- `--debug` enables verbose debug logging.
- `--stage <name>` forces the workflow to start from a specific stage.
- `--output-dir <path>` changes the base directory used for projects.

Run `nepflow --help` to see the full CLI help.

## Generation workers and progress

Perturbation generation logs its task plan before work begins, then records
each ordered task publication. A `Waiting for ordered task` heartbeat means a
task has not returned; it may be CPU-bound slow work or a stalled external
dependency. It identifies the base and family plan preventing later completed
tasks from being published out of order. It is not a success message and does
not mean a partial artifact has been promoted.

`generation.n_workers=1` is serial and is the best choice for debugging or a
small WSL machine. `0` selects automatic parallelism using every CPU granted
to the current process, bounded by affinity, cgroup quota/cpuset, the SLURM
`--cpus-per-task` allocation, and runnable task count. Information-entropy
selection follows the same policy for `selection.local_descriptor_workers=0`;
descriptor generation is CPU-only, even when later training or labelling uses
GPUs. An optional `[resources] worker_cap` remains an explicit upper bound.

On a full Pitagora Booster node (64 physical CPU cores and four GPUs), request
the resources you intend to use, for example
`--cpus-per-task=64 --gres=gpu:4`. `--gres=gpu:4` reserves four GPUs but does
not grant 64 CPUs. Leave both worker settings at `0` after requesting the CPUs,
or set an explicit lower count when memory is the limiting resource.
Do not run heavy parallel generation or descriptor calculation on login nodes;
submit it through SLURM with the required CPU allocation.

For information-entropy GPU execution, select the backend and device list
explicitly in the selection config. For example:

```ini
entropy_bandwidth_backend=exact_tiled_gpu_knn
entropy_kernel_support_execution=exact_tiled_gpu
entropy_kernel_bandwidth_rule=symmetric_max
entropy_gpu_devices=0,1,2,3
```

The ordinals are process-visible CUDA ordinals and are checked against
`CUDA_VISIBLE_DEVICES` and the SLURM allocation. A missing or unallocated
requested device fails the run; NEPFlow does not silently fall back to CPU or
reduce the requested GPU count. kNN and strict-support work are partitioned in
deterministic source order, and their per-device throughput and transfer/
certification metrics are included in the execution diagnostics.

The scientific kernel rule is separate from support execution. The production
default is `symmetric_max`, where for source row `a` and target row `b`,
`H(a,b)=max(h_a,h_b)` and `W_ab=wendland_c2(d_ab/H(a,b))`. Set
`entropy_kernel_bandwidth_rule=asymmetric_source` to reproduce the historical
source-oriented operator, where `H(a,b)=h_a`. The bounded exact
support providers accept both rules. For `symmetric_max`, c=1 can reuse a
persisted canonical kNN prefix through
the deterministic `symmetric_knn_union` reverse index, while general radii use
`symmetric_tiled_cpu` or `symmetric_tiled_gpu` with CPU-canonical strict
verification.
Its raw weights and strict finite-support masks (`d_ab < H(a,b)`) are
symmetric, but source-column normalization still produces conditional
probabilities `P(b|a)` whose columns sum to one and need not be symmetric.
Duplicate descriptor rows retain multiplicity, equality at the support
boundary contributes zero, and finite-support or FP64-underflow cases do not
receive artificial floors. The same rule and exact support strategy are used
for calibration, LOO, candidate PMFs, checkpoints, and final selection.

## Project Layout

Each project lives under `projects/project_<name>/` and typically contains:

- `config/` for project settings
- `logs/` for runtime logs
- `state.db` for authoritative workflow state and stage history
- `.project` as a legacy/interoperability marker repaired from `state.db`

The project configuration is stored in the project `config/` directory. Edit it to control:

- the elements and compositions to generate
- structure generation settings
- VASP, NEP, GPUMD, and SLURM options

## Notes

- If a run is interrupted, rerun the same command to continue from the saved state.
- For HPC runs, make sure the cluster modules, executables, and submission settings are configured correctly.
