#!/bin/bash
# SLURM job to run the adaptive VASP launcher
# This script submits the launcher as a background job on the login node

#SBATCH --job-name=vasp-launcher
#SBATCH --partition=boost_fua_prod
#SBATCH --account=fupb1_maleposa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=23:59:00
#SBATCH --output=slurm_logs/launcher_%j.log
#SBATCH --error=slurm_logs/launcher_%j.err

# This SLURM job runs the launcher script which submits all structure jobs
# The launcher itself runs on the login node and monitors job submission

# Use the current working directory from where sbatch was launched
cd "$SLURM_SUBMIT_DIR" || exit 1

echo "=========================================================================="
echo "VASP Launcher Job Started"
echo "=========================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Timestamp: $(date)"
echo "=========================================================================="

# Run the launcher script
bash launch_all.sh

LAUNCHER_EXIT=$?

echo ""
echo "=========================================================================="
if [ $LAUNCHER_EXIT -eq 0 ]; then
    echo "✓ Launcher completed successfully"
else
    echo "✗ Launcher failed with exit code $LAUNCHER_EXIT"
fi
echo "Timestamp: $(date)"
echo "=========================================================================="

exit $LAUNCHER_EXIT
