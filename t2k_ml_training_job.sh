#!/bin/bash
#SBATCH --account=rpp-blairt2k_gpu
#SBATCH --output=/project/rpp-blairt2k/fcormier/t2k/ml/logfiles/%x.%A.out
#SBATCH --error=/project/rpp-blairt2k/fcormier/t2k/ml/logfiles/%x.%A.err
#SBATCH --gpus-per-node=h100:4         # Number of GPU(s) per node
#SBATCH --ntasks-per-node=32
#SBATCH --exclusive
#SBATCH --mem=192000M  # memory per no
#SBATCH --time=5-00:00
export OMP_NUM_THREADS=$SLURM_NTASKS_PER_NODE
export HYDRA_FULL_ERROR=1


#Copy file you want to use to train to the GPU machine
#cp /scratch/fcormier/t2k/ml/output_skdetsim/apr3_eMuPiPlus_1500MeV_2M_1/multi_combine.hy $SLURM_TMPDIR/
#cp /scratch/fcormier/t2k/ml/output_skdetsim/jul18_muons_2GeV_2M_combine/multi_combine.hy $SLURM_TMPDIR/
#cp /scratch/fcormier/t2k/ml/output_skdetsim/mar22_muons_forRegression_5M_1/digi_combine.hy $SLURM_TMPDIR/
#cp /scratch/fcormier/t2k/ml/output_skdetsim/aug19_electrons_2GeV_2M_combine/multi_combine.hy $SLURM_TMPDIR
#cp /scratch/fcormier/t2k/ml/output_skdetsim/jan28_electronsOnly_5M/digi_combine.hy $SLURM_TMPDIR/
#cp /scratch/fcormier/t2k/ml/output_skdetsim/apr3_eMuPiPlus_1500MeV_2M_1/multi_combine.hy $SLURM_TMPDIR/ 

#cp /scratch/fcormier/t2k/ml/output_skdetsim/sep26_electrons_2GeV_seed_combine_6M//multi_combine.hy $SLURM_TMPDIR
#Old Muons
#cp /scratch/fcormier/t2k/ml/output_skdetsim/oct7_muons_2GeV_seed_1M_combine_1/multi_combine.hy $SLURM_TMPDIR
#cp /scratch/fcormier/t2k/ml/output_skdetsim/jan6_muons_dwallM50_combine_1/multi_combine.hy $SLURM_TMPDIR
#New 3-class
#cp /scratch/fcormier/t2k/ml/output_skdetsim/may4_2025_eMuPiPlus_combine_2/multi_combine.hy $SLURM_TMPDIR
#New muons
#cp /scratch/fcormier/t2k/ml/output_skdetsim/may29_2025_muons_pg_combine_1/multi_combine.hy $SLURM_TMPDIR
#New electrons
cp /scratch/fcormier//t2k/ml/output_skdetsim//jun12_electronsCombine_1/multi_combine.hy $SLURM_TMPDIR
#cp /scratch/fcormier/t2k/ml/output_skdetsim/oct11_eMuPosPions_2GeV_seed_1M_combine_1/multi_combine.hy $SLURM_TMPDIR

module load StdEnv/2023
module load apptainer/1.2.4


export APPTAINER_BINDPATH="/scratch/,/localscratch/"
apptainer exec --nv /project/rpp-blairt2k/machine_learning/containers/container_base_ml_v3.0.0.sif bash "/home/fcormier/t2k/ml/training/t2k_ml_training/run_training.sh"
