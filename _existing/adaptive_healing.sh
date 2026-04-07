#!/bin/bash
# Self-healing VASP launcher with automatic requeue on OOM
# Detects memory errors and resubmits with more conservative parameters

#SBATCH --partition=boost_fua_prod
#SBATCH --account=fupb1_maleposa
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x_%j.log
#SBATCH --error=slurm_logs/%x_%j.err

# ============================================================================
# SELF-HEALING ADAPTIVE VASP LAUNCHER
# ============================================================================
# If OOM detected, automatically increases NCORE/decreases KPAR and requeues

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1

# Create slurm_logs directory if it doesn't exist
mkdir -p "${SLURM_SUBMIT_DIR}/slurm_logs"

SCRIPT_DIR="$(pwd)"
BASE_DIR="${SCRIPT_DIR}"

# Get structure name from argument
if [ -z "$1" ]; then
    echo "Usage: sbatch adaptive_healing.sh <structure_dir>"
    exit 1
fi

if [[ "$1" = /* ]]; then
    STRUCT_DIR="$1"
else
    STRUCT_DIR="${BASE_DIR}/$1"
fi

STRUCT_NAME=$(basename "$STRUCT_DIR")

NGPU=${SLURM_NTASKS_PER_NODE:-4}
NNODES=${SLURM_NNODES:-1}

# For multi-node: total ranks = NGPU per node * num nodes
if [ $NNODES -gt 1 ]; then
    TOTAL_RANKS=$((NGPU * NNODES))
else
    TOTAL_RANKS=$NGPU
fi

if [ ! -d "$STRUCT_DIR" ]; then
    echo "ERROR: Structure directory not found: $STRUCT_DIR"
    exit 1
fi

STRUCT_NAME=$(basename "$STRUCT_DIR")
RETRY_LEVEL_FILE="${STRUCT_DIR}/.vasp_retry_level"

# Determine current retry level from file (only incremented on requeue)
if [ -f "$RETRY_LEVEL_FILE" ]; then
    RETRY_LEVEL=$(cat "$RETRY_LEVEL_FILE")
else
    RETRY_LEVEL=0
fi

echo "=========================================================================="
echo "Self-Healing VASP Launcher (Retry Level: $RETRY_LEVEL)"
echo "=========================================================================="
echo "Structure: $STRUCT_DIR"
echo "Timestamp: $(date)"
echo "=========================================================================="

# Load modules
module purge
module load profile/chem-phys
module load nvhpc/
module load cuda/
module load openmpi/
module load vasp/6.4.3--hpcx-mpi--2.20--nvhpc--24.9

cd "$STRUCT_DIR"

# ============================================================================
# APPLY RETRY-LEVEL PARAMETERS (INTELLIGENT ESCALATION)
# ============================================================================
# Strategy: Try parameter combinations in order of increasing resource requests
# Levels 0-3: 4 GPUs with different NCORE/KPAR combos
# Level 4+: Request 2 nodes (8 GPUs)

case $RETRY_LEVEL in
    0)
        # Level 0: Default baseline (NCORE=16, KPAR=2)
        echo "Level 0: Baseline - NCORE=16, KPAR=2 (4 GPUs, 2 k-point groups)"
        NCORE_VAL=16
        KPAR_VAL=2
        NNODES=1
        ;;
    1)
        # Level 1: Maximize per-GPU parallelism (NCORE=32, KPAR=2)
        echo "Level 1: Increase NCORE to 32 - more CPU threading per GPU"
        NCORE_VAL=32
        KPAR_VAL=2
        NNODES=1
        ;;
    2)
        # Level 2: Even more CPU threading (NCORE=64, KPAR=2)
        echo "Level 2: Maximize NCORE to 64 - max CPU threading per GPU"
        NCORE_VAL=64
        KPAR_VAL=2
        NNODES=1
        ;;
    3)
        # Level 3: Reduce k-point parallelism (NCORE=64, KPAR=1)
        echo "Level 3: Reduce KPAR to 1 - all k-points on one GPU group"
        NCORE_VAL=64
        KPAR_VAL=1
        NNODES=1
        ;;
    4)
        # Level 4: Request 4 nodes (16 GPUs)
        echo "Level 4: Escalating to 4 nodes (16 GPUs) - 384 GB total GPU memory"
        NNODES=4
        NCORE_VAL=16
        KPAR_VAL=8
        ;;
    5)
        # Level 5: Request 8 nodes (32 GPUs)
        echo "Level 5: Escalating to 8 nodes (32 GPUs) - 768 GB total GPU memory"
        NNODES=8
        NCORE_VAL=16
        KPAR_VAL=16
        ;;
    6)
        # Level 6: Request 16 nodes (64 GPUs) - maximum allowed
        echo "Level 6: Escalating to 16 nodes (64 GPUs) - 1536 GB total GPU memory"
        NNODES=16
        NCORE_VAL=16
        KPAR_VAL=32
        ;;
    *)
        # Level 7+: Give up
        echo "ERROR: Max retry level exceeded (16 nodes exhausted). Job cannot run."
        exit 1
        ;;
esac

# Update INCAR with retry-level parameters
python3 << PYTHON_EOF
struct_dir = "$STRUCT_DIR"
ncore = "$NCORE_VAL"
kpar = "$KPAR_VAL"

with open(f"{struct_dir}/INCAR", 'r') as f:
    lines = f.readlines()

output = []
for line in lines:
    if line.strip().startswith('NCORE'):
        output.append(f"NCORE = {ncore}\n")
    elif line.strip().startswith('KPAR'):
        output.append(f"KPAR = {kpar}\n")
    else:
        output.append(line)

with open(f"{struct_dir}/INCAR", 'w') as f:
    f.writelines(output)

print(f"Updated INCAR: NCORE={ncore}, KPAR={kpar}")
PYTHON_EOF

echo ""
echo "Step 3: Running VASP..."
echo "Nodes: $NNODES, GPUs per node: $NGPU, Total MPI ranks: $TOTAL_RANKS"
echo "Command: mpirun -np $TOTAL_RANKS vasp_std"
echo "=========================================================================="

# Run VASP and capture output to structure directory
VASP_LOG="${STRUCT_DIR}/vasp_output.log"
mpirun -np $TOTAL_RANKS vasp_std > "$VASP_LOG" 2>&1
VASP_EXIT=$?

echo ""
echo "VASP exit code: $VASP_EXIT"
echo "Output saved to: $VASP_LOG"

# Check if OUTCAR was created
if [ -f "$STRUCT_DIR/OUTCAR" ]; then
    OUTCAR_LINES=$(wc -l < "$STRUCT_DIR/OUTCAR")
    echo "OUTCAR created with $OUTCAR_LINES lines"
else
    echo "WARNING: OUTCAR not found in $STRUCT_DIR"
fi

# ============================================================================
# CHECK FOR OOM AND REQUEUE IF NEEDED
# ============================================================================

if grep -q "oom_kill" "$VASP_LOG" 2>/dev/null || [ $VASP_EXIT -eq 137 ]; then
    echo ""
    echo "⚠ OOM detected! Requeuing with next parameter combination..."
    
    # Calculate next level
    NEXT_LEVEL=$((RETRY_LEVEL + 1))
    
    # Get the full script path
    SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
    
    # Determine nodes for next level
    if [ $NEXT_LEVEL -lt 4 ]; then
        # Levels 0-3: Stay on 1 node
        REQUEUE_NODES=1
        REQUEUE_GPUS=4
    elif [ $NEXT_LEVEL -eq 4 ]; then
        # Level 4: 4 nodes
        REQUEUE_NODES=4
        REQUEUE_GPUS=16
    elif [ $NEXT_LEVEL -eq 5 ]; then
        # Level 5: 8 nodes
        REQUEUE_NODES=8
        REQUEUE_GPUS=32
    else
        # Levels 6+: 16 nodes (max)
        REQUEUE_NODES=16
        REQUEUE_GPUS=64
    fi
    
    # Write next level to file for next job to read
    echo "$NEXT_LEVEL" > "$RETRY_LEVEL_FILE"
    
    # Requeue with appropriate resources
    sbatch --nodes=$REQUEUE_NODES --gres=gpu:$((REQUEUE_GPUS/REQUEUE_NODES)) "$SCRIPT_PATH" "$STRUCT_DIR"
    
    echo "Job requeued with retry level $NEXT_LEVEL (nodes=$REQUEUE_NODES, gpus=$REQUEUE_GPUS)"
    exit 0
fi

# ============================================================================
# SUCCESS OR UNRECOVERABLE ERROR
# ============================================================================

if [ $VASP_EXIT -eq 0 ]; then
    echo "✓ Job completed successfully"
    # Clean up retry level file on success
    rm -f "$RETRY_LEVEL_FILE"
    # Keep OUTCAR for monitoring, only clean up temporary files
    cd "$STRUCT_DIR"
    rm -f CHG CHGCAR WAVECAR CONTCAR DOSCAR EIGENVAL PCDAT
    exit 0
else
    echo "✗ VASP failed with exit code $VASP_EXIT (not OOM)"
    echo "Check $STRUCT_DIR/vasp_output.log for details"
    # Clean up retry level file on non-OOM failure too
    rm -f "$RETRY_LEVEL_FILE"
    exit 1
fi
    exit 1
fi
