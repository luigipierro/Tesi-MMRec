#!/bin/bash
#SBATCH --job-name=green_mmrec
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH -p boost_usr_prod
#SBATCH --account=IscrC_MMRecSys

###############################################
# USER PARAMETERS — EDIT THESE FOR YOUR JOB
###############################################

CONTAINER=""            # e.g. /path/my_container.sif  OR leave empty if using venv
VENV="../../../_env"          # Path to virtualenv  (e.g. ../_env)
LAUNCHER="python"       # python | accelerate
SCRIPT="main.py"   # Python script to run

###############################################

# setup project directory
PROJECT=`pwd`
ROOT=$(echo "$PROJECT" | cut -d/ -f 1-3)

module purge

# networking stuff
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
export MASTER_PORT=8000
export GPU_PER_NODE=4
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Leonardo has no internet
export TRANSFORMERS_OFFLINE="1"
export HF_DATASETS_OFFLINE="1"
export WANDB_MODE="offline"

export TF_ENABLE_ONEDNN_OPTS=0

###############################################
# Argument validation (replaces getopts)
###############################################

if [[ -z "$CONTAINER" && -z "$VENV" ]]; then
    echo "ERROR: You must specify either a Singularity container OR a venv."
    exit 1
fi

if [[ -z "$SCRIPT" ]]; then
    echo "ERROR: SCRIPT variable cannot be empty."
    exit 1
fi

###############################################
# Load modules / prepare environment
###############################################

module load python/3.11.6--gcc--8.5.0
module load cuda/12.1

###############################################
# Container mode
###############################################
if [[ -n "$CONTAINER" ]]; then
    echo "Running in Singularity container: $CONTAINER"
    export SINGULARITY_TMPDIR=$CINECA_SCRATCH/tmp
    module load singularity/3.9-pro

    CMD="srun -l \
        singularity exec --nv \
        --bind $PROJECT:$PROJECT \
        --bind $SCRATCH:$SCRATCH \
        $CONTAINER \
        bash $PROJECT/run_in_container.sh"
fi

###############################################
# Virtualenv mode
###############################################
if [[ -z "$CONTAINER" ]]; then
    echo "Running in Python virtualenv: $VENV"
    source $VENV/bin/activate

    CMD="srun -l bash $PROJECT/run_in_container.sh"
fi

###############################################
# Expose parameters to run_in_container.sh
###############################################
export DISTRIBUTED_CONFIG="none"
export SCRIPT_TO_RUN=$SCRIPT
export LAUNCHER=$LAUNCHER

###############################################
# RUN
###############################################
set -x -e
$CMD

rm -f nodefile
