#!/bin/bash
# Monitor progress of VASP structures
# Proper VASP completion detection

BASEDIR="$(pwd)"

echo "=========================================================================="
echo "VASP Structure Status"
echo "=========================================================================="
echo "Timestamp: $(date)"
echo ""

# Auto-detect number of structures
MAX_STRUCT=$(find "$BASEDIR" -maxdepth 1 -type d -name "struct_*" | sed 's/.*struct_//' | sort -n | tail -1)
if [ -z "$MAX_STRUCT" ]; then
    echo "ERROR: No struct_* directories found"
    exit 1
fi
# Remove leading zeros to avoid octal interpretation (e.g., 0299 -> 299)
MAX_STRUCT=$((10#$MAX_STRUCT))
TOTAL_STRUCTS=$((MAX_STRUCT + 1))

echo "Monitoring $TOTAL_STRUCTS structures..."
echo ""

# Count structures by actual VASP completion status
SUCCESSFUL=0
RUNNING=0
FAILED=0
NOT_STARTED=0

for i in $(seq 0 $MAX_STRUCT); do
    STRUCT_NUM=$(printf '%04d' $i)
    STRUCT_DIR="${BASEDIR}/struct_${STRUCT_NUM}"
    
    # Check for VASP completion by examining OUTCAR ending
    if [ -f "$STRUCT_DIR/OUTCAR" ] && [ -s "$STRUCT_DIR/OUTCAR" ]; then
        # Check if VASP failed with errors (check for error messages)
        if grep -q "ERROR\|REFUSE TO CONTINUE\|I GIVE UP\|VERY BAD NEWS\|EXITING" "$STRUCT_DIR/OUTCAR" 2>/dev/null; then
            ((FAILED++))
            STATUS="✗ FAILED"
        # Check if VASP completed normally (look for completion markers)
        elif tail -20 "$STRUCT_DIR/OUTCAR" | grep -q "General timing and accounting informations for this job:" || \
             tail -20 "$STRUCT_DIR/OUTCAR" | grep -q "Voluntary context switches:"; then
            ((SUCCESSFUL++))
            STATUS="✓ SUCCESSFUL"
        else
            # OUTCAR exists but incomplete - VASP is running or crashed
            if [ -f "$STRUCT_DIR/vasp_output.log" ] && [ -n "$(find "$STRUCT_DIR/vasp_output.log" -mmin -5 2>/dev/null)" ]; then
                ((RUNNING++))
                STATUS="⟳ RUNNING"
            else
                ((RUNNING++))
                STATUS="⟳ INCOMPLETE"
            fi
        fi
    # Check for failed jobs - vasp_output.log exists but no OUTCAR or empty OUTCAR
    elif [ -f "$STRUCT_DIR/vasp_output.log" ]; then
        ((FAILED++))
        STATUS="✗ FAILED"
    # No files = not started
    else
        ((NOT_STARTED++))
        STATUS="⊘ Not started"
    fi
    
    # Print every 10th
    if [ $((i % 10)) -eq 0 ]; then
        printf "struct_%04d: %s\n" $i "$STATUS"
    fi
done

echo ""
echo "=========================================================================="
echo "VASP EXECUTION SUMMARY"
echo "=========================================================================="
if [ $TOTAL_STRUCTS -gt 0 ]; then
    printf "Successful (completed):  %3d/%3d (%.0f%%)\n" $SUCCESSFUL $TOTAL_STRUCTS $((SUCCESSFUL * 100 / TOTAL_STRUCTS))
else
    printf "Successful (completed):  %3d/%3d (0%%)\n" $SUCCESSFUL $TOTAL_STRUCTS
fi
printf "Running (incomplete):    %3d\n" $RUNNING  
printf "Failed (crashed):        %3d\n" $FAILED
printf "Not Started:             %3d\n" $NOT_STARTED
echo "=========================================================================="

echo ""
echo "Last updated: $(date)"