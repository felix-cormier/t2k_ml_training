#!/bin/bash

cd /home/fcormier/t2k/ml/training/t2k_ml_training/

source setup.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_GPU
export HYDRA_FULL_ERROR=1


python training_runner.py