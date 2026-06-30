#!/bin/bash

# useful for debugging/determinism
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

# SLURM_PROCID is evaluated within this script, this is why this script exists instead of doing 
# everything in launch.slurm 
if [[ "$DISTRIBUTED_CONFIG" != "none" ]]
then
    export LAUNCHER="accelerate launch \
        --config_file configs/${DISTRIBUTED_CONFIG}_config.yaml \
        --main_process_ip $MASTER_ADDR \
        --main_process_port $MASTER_PORT \
        --num_processes $(( $GPU_PER_NODE * $COUNT_NODE)) \
        --num_machines $COUNT_NODE \
        --machine_rank $SLURM_PROCID \
        "
else
    export LAUNCHER="python3"
fi

CMD="$LAUNCHER $SCRIPT_TO_RUN"
$CMD
