#!/bin/bash
#SBATCH -s
#SBATCH -n 1
#SBATCH -o ./logs/log_%j.out
#SBATCH -J flux_g
#SBATCH -p normal
#SBATCH -c 4

source /NFSHOME/gdaloisio/miniconda3/etc/profile.d/conda.sh
conda activate imgbias

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun python flux.py --style g
