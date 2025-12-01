#!/bin/bash

# Usage: ./run_net_arch_change.sh <algo> <batch_size>
# Example: ./run_net_arch_change.sh DQN 5

algo=$1
BATCH_SIZE=${2:-5}

mkdir -p logs/net-arch

if [ -z "$algo" ]; then
    echo "Usage: ./run_net_arch_change.sh <algo> <batch_size>"
    echo "  algo: DQN, DoubleDQN, RPI, or PPO"
    echo "  batch_size: number of parallel runs (default: 5)"
    exit 1
fi

MAIN_SCRIPT="main_cartpole.py"
USE_WANDB=true
WANDB_PROJECT="Cartpole-NetArch-GitHub"


TOTAL_RUNS=10
WIDTHS=(8 16 32 64 128 256)
DEPTHS=(2)

echo "========================================="
echo "Network Architecture Experiment"
echo "Algorithm: $algo"
echo "Batch size: $BATCH_SIZE"
echo "Total runs per configuration: $TOTAL_RUNS"
echo "Widths: ${WIDTHS[@]}"
echo "Depths: ${DEPTHS[@]}"
echo "========================================="

for width in "${WIDTHS[@]}"; do
    for depth in "${DEPTHS[@]}"; do
        echo ""
        echo "==== Testing $algo with width=$width, depth=$depth ===="

        for ((i=0; i<TOTAL_RUNS; i+=BATCH_SIZE)); do
            batch_num=$((i / BATCH_SIZE + 1))
            echo "  Starting batch $batch_num for width=$width, depth=$depth"

            for ((j=0; j<BATCH_SIZE && i+j<TOTAL_RUNS; j++)); do
                seed=$((i + j))
                echo "    Running seed: $seed"

                WANDB_ARGS=""
                if [ "$USE_WANDB" = true ]; then
                    WANDB_ARGS="--use-wandb --wandb-project $WANDB_PROJECT"
                fi

                python -u $MAIN_SCRIPT \
                    --seed $seed \
                    --algo $algo \
                    --exp-type net-arch \
                    --net-width $width \
                    --net-depth $depth \
                    $WANDB_ARGS \
                    > logs/net-arch/${algo}_width_${width}_depth_${depth}_seed_${seed}.log 2>&1 &
            done

            wait
            echo "  Finished batch $batch_num for width=$width, depth=$depth"
        done
    done
done

echo ""
echo "========================================="
echo "All experiments completed!"
echo "========================================="
