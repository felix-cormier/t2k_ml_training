#!/bin/bash
#SBATCH --account=def-blairt2k
#SBATCH --output=/scratch/fcormier/t2k/ml/logfiles/%x.%A.out
#SBATCH --error=/scratch/fcormier/t2k/ml/logfiles/%x.%A.err
#SBATCH --gpus-per-node=v100l:4         # Number of GPU(s) per node
#SBATCH --cpus-per-gpu=6         # CPU cores/threads
#SBATCH --mem=160000M               # memory per node
#SBATCH --time=3-00:00
export OMP_NUM_THREADS=$SLURM_CPUS_PER_GPU
export HYDRA_FULL_ERROR=1


#Copy file you want to use to train to the GPU machine
#cp /scratch/fcormier/t2k/ml/output_skdetsim/apr3_eMuPiPlus_1500MeV_2M_1/multi_combine.hy $SLURM_TMPDIR/
cp /scratch/fcormier/t2k/ml/output_skdetsim/jul18_muons_2GeV_2M_combine/multi_combine.hy $SLURM_TMPDIR/

module load StdEnv/2023
module load apptainer/1.2.4


export APPTAINER_BINDPATH="/scratch/,/localscratch/"
apptainer exec --nv /project/rpp-blairt2k/machine_learning/containers/container_base_ml_v3.0.0.sif bash "/home/fcormier/t2k/ml/training/t2k_ml_training/run_training.sh"