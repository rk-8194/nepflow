#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --error=myJob.err
#SBATCH --output=myJob.out
#SBATCH --account=FUPB2_MALPSAMS
#SBATCH --partition=boost_fua_prod

module purge
module load profile/chem-phys
module load nvhpc/
module load cuda/
module load openmpi/
module load vasp/6.4.3--hpcx-mpi--2.20--nvhpc--24.9

source $HOME/ase/bin/activate