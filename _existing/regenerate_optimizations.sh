#!/bin/bash
# Clean and regenerate optimization configurations
# Use this to force re-computation of all optimizations

BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_LAUNCHER="${BASEDIR}/adaptive_vasp_launcher.py"

echo "=========================================================================="
echo "Clean Regenerate Optimizations"
echo "=========================================================================="

# Option 1: Clean all optimization files
if [ "$1" == "--clean" ] || [ "$1" == "-c" ]; then
    echo "Removing all optimization files..."
    find "$BASEDIR" -name "optimization_config.json" -delete
    find "$BASEDIR" -name "ADAPTIVE_OPTIMIZATION.log" -delete
    echo "✓ Cleaned"
    exit 0
fi

# Option 2: Regenerate specific structures
if [ ! -z "$1" ] && [ "$1" != "--all" ]; then
    # Regenerate specific range (e.g., "0 10" for struct_0000 to struct_0009)
    START=${1:-0}
    END=${2:-10}
    
    echo "Regenerating struct_$(printf '%04d' $START) to struct_$(printf '%04d' $((END-1)))..."
    for i in $(seq $START $((END-1))); do
        STRUCT_NUM=$(printf "%04d" $i)
        STRUCT_DIR="${BASEDIR}/struct_${STRUCT_NUM}"
        
        if [ ! -d "$STRUCT_DIR" ]; then
            continue
        fi
        
        echo "  Removing old data for struct_${STRUCT_NUM}..."
        rm -f "${STRUCT_DIR}/optimization_config.json"
        rm -f "${STRUCT_DIR}/ADAPTIVE_OPTIMIZATION.log"
    done
    
    # Now regenerate
    echo ""
    echo "Regenerating optimizations..."
    for i in $(seq $START $((END-1))); do
        STRUCT_NUM=$(printf "%04d" $i)
        STRUCT_DIR="${BASEDIR}/struct_${STRUCT_NUM}"
        
        if [ ! -d "$STRUCT_DIR" ]; then
            continue
        fi
        
        echo "  [$((i-START+1))/$((END-START))] struct_${STRUCT_NUM}..."
        python3 "$PYTHON_LAUNCHER" "$STRUCT_DIR" --ngpu 4 --max-nodes 4 > /dev/null 2>&1
    done
    
    exit 0
fi

# Option 3: Regenerate all (default)
echo "Full regeneration of all optimizations..."
echo ""

# Auto-detect number of structures
MAX_STRUCT=$(find "$BASEDIR" -maxdepth 1 -type d -name "struct_*" | sed 's/.*struct_//' | sort -n | tail -1)
if [ -z "$MAX_STRUCT" ]; then
    echo "ERROR: No struct_* directories found"
    exit 1
fi
# Remove leading zeros to avoid octal interpretation (e.g., 0299 -> 299)
MAX_STRUCT=$((10#$MAX_STRUCT))

# First, clean
find "$BASEDIR" -name "optimization_config.json" -delete
find "$BASEDIR" -name "ADAPTIVE_OPTIMIZATION.log" -delete

# Then regenerate
echo "Testing different NCORE/KPAR combinations to find optimal parameters..."
echo "This will test combinations in order: conservative (low speed) → fast (high memory usage)"
echo ""

TESTED_COUNT=0
for i in $(seq 0 $MAX_STRUCT); do
    STRUCT_NUM=$(printf "%04d" $i)
    STRUCT_DIR="${BASEDIR}/struct_${STRUCT_NUM}"
    
    if [ ! -d "$STRUCT_DIR" ]; then
        continue
    fi
    
    if [ $((i % 10)) -eq 0 ]; then
        echo "[$((TESTED_COUNT+1))] struct_${STRUCT_NUM}... testing parameter combinations"
    fi
    
    # Backup original INCAR
    cp "${STRUCT_DIR}/INCAR" "${STRUCT_DIR}/INCAR.backup"
    
    # Test parameter combinations in order of increasing speed (decreasing memory conservatism)
    # Format: "NCORE KPAR"
    declare -a PARAM_COMBOS=(
        "64 1"
        "48 1"
        "32 1"
        "32 2"
        "24 2"
        "16 2"
        "16 4"
        "12 4"
        "8 4"
    )
    
    BEST_NCORE=""
    BEST_KPAR=""
    BEST_MEMORY=999
    FOUND_WORKING=false
    
    # Test each combination
    for combo in "${PARAM_COMBOS[@]}"; do
        NCORE=$(echo $combo | awk '{print $1}')
        KPAR=$(echo $combo | awk '{print $2}')
        
        # Apply parameters to INCAR
        python3 << PYTHON_EOF
import re
incar_file = "${STRUCT_DIR}/INCAR"
try:
    with open(incar_file, 'r') as f:
        content = f.read()
    
    # Replace or add NCORE and KPAR
    if re.search(r'^NCORE\s*=', content, re.MULTILINE):
        content = re.sub(r'^NCORE\s*=.*', f'NCORE = {NCORE}', content, flags=re.MULTILINE)
    else:
        content += f'\nNCORE = {NCORE}\n'
    
    if re.search(r'^KPAR\s*=', content, re.MULTILINE):
        content = re.sub(r'^KPAR\s*=.*', f'KPAR = {KPAR}', content, flags=re.MULTILINE)
    else:
        content += f'KPAR = {KPAR}\n'
    
    with open(incar_file, 'w') as f:
        f.write(content)
except Exception as e:
    pass
PYTHON_EOF
        
        # Run launcher to estimate memory and feasibility
        if python3 "$PYTHON_LAUNCHER" "$STRUCT_DIR" --ngpu 4 --max-nodes 4 > /dev/null 2>&1; then
            # Parse memory estimate from optimization_config.json
            if [ -f "${STRUCT_DIR}/optimization_config.json" ]; then
                MEMORY=$(grep -o '"estimated_memory_per_gpu_gb": [0-9.]*' "${STRUCT_DIR}/optimization_config.json" 2>/dev/null | grep -o '[0-9.]*$' 2>/dev/null)
                
                if [ ! -z "$MEMORY" ]; then
                    # Check if memory is feasible (safety margin: < 20 GB for 24GB V100)
                    MEMORY_INT=$(echo "$MEMORY" | cut -d'.' -f1)
                    if [ "$MEMORY_INT" -lt 20 ]; then
                        # Found a working combination - prefer fastest (highest NCORE+KPAR = faster compute)
                        if [ "$FOUND_WORKING" = false ] || (( $(echo "$NCORE + $KPAR > $BEST_NCORE + $BEST_KPAR" | bc -l 2>/dev/null) )); then
                            BEST_NCORE=$NCORE
                            BEST_KPAR=$KPAR
                            BEST_MEMORY=$MEMORY
                            FOUND_WORKING=true
                            # Continue testing faster options
                        fi
                    fi
                fi
            fi
        fi
    done
    
    # Apply best found parameters
    if [ "$FOUND_WORKING" = true ]; then
        python3 << PYTHON_EOF
import re
incar_file = "${STRUCT_DIR}/INCAR"
try:
    with open(incar_file, 'r') as f:
        content = f.read()
    
    if re.search(r'^NCORE\s*=', content, re.MULTILINE):
        content = re.sub(r'^NCORE\s*=.*', f'NCORE = {BEST_NCORE}', content, flags=re.MULTILINE)
    else:
        content += f'\nNCORE = {BEST_NCORE}\n'
    
    if re.search(r'^KPAR\s*=', content, re.MULTILINE):
        content = re.sub(r'^KPAR\s*=.*', f'KPAR = {BEST_KPAR}', content, flags=re.MULTILINE)
    else:
        content += f'KPAR = {BEST_KPAR}\n'
    
    with open(incar_file, 'w') as f:
        f.write(content)
except Exception as e:
    pass
PYTHON_EOF
    else
        # No working combination found - use most conservative
        python3 << PYTHON_EOF
import re
incar_file = "${STRUCT_DIR}/INCAR"
try:
    with open(incar_file, 'r') as f:
        content = f.read()
    
    if re.search(r'^NCORE\s*=', content, re.MULTILINE):
        content = re.sub(r'^NCORE\s*=.*', 'NCORE = 64', content, flags=re.MULTILINE)
    else:
        content += '\nNCORE = 64\n'
    
    if re.search(r'^KPAR\s*=', content, re.MULTILINE):
        content = re.sub(r'^KPAR\s*=.*', 'KPAR = 1', content, flags=re.MULTILINE)
    else:
        content += 'KPAR = 1\n'
    
    with open(incar_file, 'w') as f:
        f.write(content)
except Exception as e:
    pass
PYTHON_EOF
    fi
    
    # Generate final optimization config with best parameters
    python3 "$PYTHON_LAUNCHER" "$STRUCT_DIR" --ngpu 4 --max-nodes 4 > /dev/null 2>&1
    
    # Clean up
    rm -f "${STRUCT_DIR}/INCAR.backup"
    ((TESTED_COUNT++))
done

echo ""
echo "=========================================================================="
echo "Parameter optimization complete!"
echo "Each structure now has NCORE/KPAR optimized for its specific geometry"
echo "Review parameters with: ./review_optimizations.sh"
echo "=========================================================================="
