#!/bin/bash
# Robust launcher to submit all structures as adaptive VASP jobs
# Handles retry logic, error checking, and progress tracking

BASEDIR="$(pwd)"
LAUNCHER_SCRIPT="adaptive_healing.sh"
MAX_CONCURRENT=20  # Limit concurrent jobs to avoid queue saturation
WAIT_TIME=10

echo "=========================================================================="
echo "Adaptive VASP Array Launcher"
echo "=========================================================================="
echo "Base directory: $BASEDIR"
echo "Max concurrent jobs: $MAX_CONCURRENT"
echo "Timestamp: $(date)"
echo "=========================================================================="

cd "$BASEDIR" || exit 1

# Auto-detect number of structures
MAX_STRUCT=$(find "$BASEDIR" -maxdepth 1 -type d -name "struct_*" | sed 's/.*struct_//' | sort -n | tail -1)
if [ -z "$MAX_STRUCT" ]; then
    echo "ERROR: No struct_* directories found"
    exit 1
fi
# Remove leading zeros to avoid octal interpretation (e.g., 0299 -> 299)
MAX_STRUCT=$((10#$MAX_STRUCT))
TOTAL_STRUCTS=$((MAX_STRUCT + 1))

echo "Found $TOTAL_STRUCTS structures (struct_0000 to struct_$(printf '%04d' $MAX_STRUCT))"
echo ""

# Clean up all logs and VASP output files before starting
echo "Cleaning up previous run files..."
mkdir -p slurm_logs
rm -f slurm_logs/*.log slurm_logs/*.err
for i in $(seq 0 $MAX_STRUCT); do
    STRUCT_NUM=$(printf '%04d' $i)
    STRUCT_DIR="${BASEDIR}/struct_${STRUCT_NUM}"
    if [ -d "$STRUCT_DIR" ]; then
        # Clean up ALL intermediate files, but preserve completed OUTCARs (>100 lines = likely complete)
        # Always clean these temporary/log files:
        rm -f "$STRUCT_DIR/vasp_output.log" "$STRUCT_DIR/.vasp_retry_level"
        rm -f "$STRUCT_DIR/CHG" "$STRUCT_DIR/CHGCAR" "$STRUCT_DIR/WAVECAR" 
        rm -f "$STRUCT_DIR/vasprun.xml" "$STRUCT_DIR/OSZICAR"
        
        # Only clean OUTCAR if it's incomplete (<=100 lines or missing)
        if [ ! -f "$STRUCT_DIR/OUTCAR" ] || [ $(wc -l < "$STRUCT_DIR/OUTCAR" 2>/dev/null) -le 100 ]; then
            rm -f "$STRUCT_DIR/OUTCAR"
        fi
    fi
done
echo "Cleanup complete."

# Create tracking file
SUBMISSION_LOG="${BASEDIR}/submission_$(date +%Y%m%d_%H%M%S).log"
SUBMITTED_FILE="${BASEDIR}/submitted.txt"
touch "$SUBMISSION_LOG"
touch "$SUBMITTED_FILE"

echo "Starting staggered job submission..."

while true; do
    submitted_any=false
    
    for i in $(seq 0 $MAX_STRUCT); do
        STRUCT_NUM=$(printf '%04d' $i)
        STRUCT_DIR="${BASEDIR}/struct_${STRUCT_NUM}"
        
        # Skip if already submitted
        grep -Fxq "$STRUCT_DIR" "$SUBMITTED_FILE" && continue
        
        # Check if structure directory exists
        if [ ! -d "$STRUCT_DIR" ]; then
            echo "✗ struct_$STRUCT_NUM: Directory not found" | tee -a "$SUBMISSION_LOG"
            continue
        fi
        
        # Check required files
        if [ ! -f "$STRUCT_DIR/POSCAR" ] || [ ! -f "$STRUCT_DIR/INCAR" ] || [ ! -f "$STRUCT_DIR/POTCAR" ]; then
            echo "✗ struct_$STRUCT_NUM: Missing POSCAR/INCAR/POTCAR" | tee -a "$SUBMISSION_LOG"
            continue
        fi
        
        # Check job count before submitting - only count running/pending jobs
        job_count=$(squeue -u $(whoami) --noheader --states=RUNNING,PENDING | wc -l)
        if [ $job_count -ge $MAX_CONCURRENT ]; then
            echo "[INFO] Max job limit ($MAX_CONCURRENT) reached. Waiting $WAIT_TIME seconds..."
            sleep $WAIT_TIME
            break
        fi
        
        # Submit the job with unique name
        echo "[+] Submitting struct_$STRUCT_NUM"
        if sbatch --job-name="vasp-struct_${STRUCT_NUM}" --nodes=1 --gres=gpu:4 "$LAUNCHER_SCRIPT" "$STRUCT_DIR" 2>&1 | tee -a "$SUBMISSION_LOG"; then
            echo "$STRUCT_DIR" >> "$SUBMITTED_FILE"
            echo "✓ struct_$STRUCT_NUM: Submitted" | tee -a "$SUBMISSION_LOG"
            submitted_any=true
        else
            echo "✗ struct_$STRUCT_NUM: Submission failed" | tee -a "$SUBMISSION_LOG"
        fi
        
        # Small delay to avoid queue congestion
        sleep 1
    done
    
    # Check if all structures have been submitted
    if [ $(wc -l < "$SUBMITTED_FILE") -eq $TOTAL_STRUCTS ]; then
        echo ""
        echo "=========================================================================="
        echo "ALL JOBS SUBMITTED"
        echo "=========================================================================="
        echo "Total structures: $TOTAL_STRUCTS"
        echo "Timestamp: $(date)"
        echo "Log: $SUBMISSION_LOG"
        echo "=========================================================================="
        break
    fi
    
    if [ "$submitted_any" = false ]; then
        echo "[INFO] No new jobs submitted. Waiting $WAIT_TIME seconds before next round..."
        sleep $WAIT_TIME
    fi
done

echo ""
echo "Monitor progress with:"
echo "  bash monitor_progress.sh"
echo "  squeue -u $(whoami)"
