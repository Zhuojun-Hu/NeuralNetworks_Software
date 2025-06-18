#!/bin/bash

# SLURM options:

#SBATCH --account=t2k
#SBATCH --partition=gpu                              # Partition choice
#SBATCH --ntasks=1                                   # Maximum number of parallel processes
#SBATCH --cpus-per-task=5                            # Number of threads per process
#SBATCH --gres=gpu:v100:1


echo "Sweep ID: $SWEEP_ID"

wandb agent \
    -e $ENTITY \
    -p $PROJECT \
    --count $COUNT \
    $SWEEP_ID