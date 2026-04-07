#!/bin/bash
# Batch run adaptive launcher on all structures (local testing, not for HPC)
# This runs the adaptive launcher on all struct_* directories to pre-compute
# optimizations before submitting jobs to HPC

# Usage:
#   ./batch_adaptive_launcher.sh          # Run on all structures with 4 GPUs
#   ./batch_adaptive_launcher.sh 2        # Run with 2 GPUs
#   ./batch_adaptive_launcher.sh 1 5      # Run on struct_0000 to struct_0004 (5 structures)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${SCRIPT_DIR}"

NGPU=${1:-4}

# Auto-detect number of structures if not specified
if [ ! -z "$2" ]; then
    NUM_STRUCTS=$2
else
    MAX_STRUCT=$(find "$BASE_DIR" -maxdepth 1 -type d -name "struct_*" | sed 's/.*struct_//' | sort -n | tail -1)
    if [ -z "$MAX_STRUCT" ]; then
        echo "ERROR: No struct_* directories found"
        exit 1
    fi
    # Remove leading zeros to avoid octal interpretation (e.g., 0299 -> 299)
    MAX_STRUCT=$((10#$MAX_STRUCT))
    NUM_STRUCTS=$((MAX_STRUCT + 1))
fi

PYTHON_LAUNCHER="${BASE_DIR}/adaptive_vasp_launcher.py"

echo "=========================================================================="
echo "Batch Adaptive VASP Launcher"
echo "=========================================================================="
echo "Base directory: $BASE_DIR"
echo "GPU configuration: $NGPU"
echo "Structures to process: $NUM_STRUCTS"
echo "Timestamp: $(date)"
echo "=========================================================================="
echo ""

if [ ! -f "$PYTHON_LAUNCHER" ]; then
    echo "ERROR: $PYTHON_LAUNCHER not found"
    exit 1
fi

PROCESSED=0
SUCCEEDED=0
FAILED=0

for i in $(seq 0 $((NUM_STRUCTS-1))); do
    STRUCT_NUM=$(printf "%04d" $i)
    STRUCT_DIR="${BASE_DIR}/struct_${STRUCT_NUM}"
    
    if [ ! -d "$STRUCT_DIR" ]; then
        continue
    fi
    
    echo "[$((PROCESSED+1))/$NUM_STRUCTS] Processing $STRUCT_NUM..."
    
    if python3 "$PYTHON_LAUNCHER" "$STRUCT_DIR" --ngpu $NGPU --max-nodes 4 > /dev/null 2>&1; then
        ((SUCCEEDED++))
    else
        echo "  ✗ Failed"
        ((FAILED++))
    fi
    
    ((PROCESSED++))
done

echo ""
echo "=========================================================================="
echo "BATCH PROCESSING COMPLETE"
echo "=========================================================================="
echo "Processed: $PROCESSED"
echo "Succeeded: $SUCCEEDED"
echo "Failed: $FAILED"
echo "Timestamp: $(date)"
echo "=========================================================================="
echo ""
echo "Review optimizations with: ./review_optimizations.sh"
