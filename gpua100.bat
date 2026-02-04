#!/bin/bash --login

#SBATCH -p biggpu
#SBATCH -N 1 -n 1

#SBATCH -t 10:00:00

#SBATCH --comment="Source=1; Project=TPT; Program=NBODY; Area=MC; Task=1; Param=(0)"

module load gcc/11.2.0
module load cuda/11.7.0
module load nvidia_hpc_sdk/22.7

export T3_DATA=/cluster/users/70-gaa/T3_DATA

./build/Test

