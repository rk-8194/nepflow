#!/bin/bash
# Utility script to monitor adaptive VASP jobs and review optimization logs

# Usage:
#   ./review_optimizations.sh              # Show all optimizations
#   ./review_optimizations.sh struct_0011  # Show specific structure
#   ./review_optimizations.sh --memory     # Sort by estimated memory
#   ./review_optimizations.sh --high-mem   # Show high memory structures

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${SCRIPT_DIR}"

echo "=========================================================================="
echo "Adaptive VASP Optimization Review"
echo "=========================================================================="
echo "Base directory: $BASE_DIR"
echo ""

# Count optimized structures
TOTAL=$(find "$BASE_DIR" -name "optimization_config.json" | wc -l)

# Auto-detect max structures
MAX_STRUCT=$(find "$BASE_DIR" -maxdepth 1 -type d -name "struct_*" | sed 's/.*struct_//' | sort -n | tail -1)
# Remove leading zeros to avoid octal interpretation (e.g., 0299 -> 299)
MAX_STRUCT=$((10#$MAX_STRUCT))
TOTAL_AVAILABLE=$((MAX_STRUCT + 1))

echo "Structures with optimization data: $TOTAL/$TOTAL_AVAILABLE"
echo ""

if [ -z "$1" ]; then
    # Show summary of all optimizations
    echo "OPTIMIZATION SUMMARY:"
    echo "------------------------------------------------------------------------"
    echo "Structure | Atoms | K-pts | Est.Mem(GB) | NCORE | KPAR | Strategy"
    echo "------------------------------------------------------------------------"
    
    find "$BASE_DIR" -name "optimization_config.json" | sort | while read config; do
        struct=$(jq -r '.struct_name' "$config")
        atoms=$(jq -r '.n_atoms' "$config")
        kpts=$(jq -r '.n_kpoints_irreducible' "$config")
        memory=$(jq -r '.estimated_memory_per_gpu_gb' "$config")
        ncore=$(jq -r '.final_ncore' "$config")
        kpar=$(jq -r '.final_kpar' "$config")
        
        printf "%-12s %5d %6d %11.2f %6d %5d\n" "$struct" "$atoms" "$kpts" "$memory" "$ncore" "$kpar"
    done
    
elif [ "$1" == "--memory" ]; then
    # Sort by estimated memory
    echo "STRUCTURES SORTED BY MEMORY REQUIREMENT:"
    echo "------------------------------------------------------------------------"
    echo "Structure | Est.Mem(GB) | NCORE | KPAR"
    echo "------------------------------------------------------------------------"
    
    find "$BASE_DIR" -name "optimization_config.json" | sort | while read config; do
        struct=$(jq -r '.struct_name' "$config")
        memory=$(jq -r '.estimated_memory_per_gpu_gb' "$config")
        ncore=$(jq -r '.final_ncore' "$config")
        kpar=$(jq -r '.final_kpar' "$config")
        echo "$memory $struct $ncore $kpar"
    done | sort -rn | while read memory struct ncore kpar; do
        printf "%-12s %11.2f %6d %5d\n" "$struct" "$memory" "$ncore" "$kpar"
    done
    
elif [ "$1" == "--high-mem" ]; then
    # Show high-memory structures (>5GB)
    echo "HIGH-MEMORY STRUCTURES (>5GB):"
    echo "------------------------------------------------------------------------"
    echo "Structure | Est.Mem(GB) | NCORE | KPAR | Reasoning"
    echo "------------------------------------------------------------------------"
    
    find "$BASE_DIR" -name "optimization_config.json" | sort | while read config; do
        memory=$(jq -r '.estimated_memory_per_gpu_gb' "$config")
        if (( $(echo "$memory > 5" | bc -l) )); then
            struct=$(jq -r '.struct_name' "$config")
            ncore=$(jq -r '.final_ncore' "$config")
            kpar=$(jq -r '.final_kpar' "$config")
            reason=$(jq -r '.reasoning[0]' "$config")
            printf "%-12s %11.2f %6d %5d | %s\n" "$struct" "$memory" "$ncore" "$kpar" "$reason"
        fi
    done
    
else
    # Show specific structure
    STRUCT=$1
    CONFIG="${BASE_DIR}/${STRUCT}/optimization_config.json"
    
    if [ ! -f "$CONFIG" ]; then
        echo "ERROR: Configuration not found for $STRUCT"
        exit 1
    fi
    
    echo "DETAILED OPTIMIZATION: $STRUCT"
    echo "------------------------------------------------------------------------"
    jq '.' "$CONFIG"
    
    # Show log if available
    LOG="${BASE_DIR}/${STRUCT}/ADAPTIVE_OPTIMIZATION.log"
    if [ -f "$LOG" ]; then
        echo ""
        echo "------------------------------------------------------------------------"
        echo "OPTIMIZATION LOG:"
        echo "------------------------------------------------------------------------"
        cat "$LOG"
    fi
fi

echo ""
echo "=========================================================================="
