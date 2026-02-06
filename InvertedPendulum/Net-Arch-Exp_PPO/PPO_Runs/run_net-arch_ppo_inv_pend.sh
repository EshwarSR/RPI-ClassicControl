#!/bin/bash

# Network Architecture Experiment for PPO on InvertedPendulum-v5
# Tests different network architectures with multiple seeds

# Network architectures to test
ARCHS=("32,32" "64,64" "128,128" "256,256" "400,300" "512,512")

# Number of seeds per architecture
NUM_SEEDS=10

# Number of parallel jobs per batch
BATCH_SIZE=${1:-10}

# Training configuration
N_TIMESTEPS=100_000
EVAL_FREQ=100
N_EVAL_EPISODES=100
DEVICE="cpu"

# WandB configuration
WANDB_PROJECT="Inverted-Pendulum-NetArch-GitHub"

# Create logs directory if it doesn't exist
mkdir -p logs

# --- Main loop to iterate over each architecture ---
for arch in "${ARCHS[@]}"; do
    # Replace commas with underscores for log file names
    arch_clean=$(echo $arch | tr ',' '_')

    echo "===================================================="
    echo "Starting runs for architecture: $arch"
    echo "===================================================="

    # --- Batching logic for parallel execution ---
    for ((i=0; i<NUM_SEEDS; i+=BATCH_SIZE)); do
        echo "Starting batch $((i / BATCH_SIZE + 1)) for arch $arch"

        for ((j=0; j<BATCH_SIZE && i+j<NUM_SEEDS; j++)); do
            current_seed=$((i + j))
            echo "  Running seed $current_seed for arch $arch"

            python -u main_ppo_invpend.py \
                --seed $current_seed \
                --device $DEVICE \
                --exp-type net-archh \
                --net-arch "$arch" \
                --n-timesteps $N_TIMESTEPS \
                --eval-freq $EVAL_FREQ \
                --n-eval-episodes $N_EVAL_EPISODES \
                --use-wandb \
                --wandb-project $WANDB_PROJECT \
                > logs/InvertedPendulum-v5_PPO_arch_${arch_clean}_seed_${current_seed}.log 2>&1 &
        done

        # Wait for all jobs in this batch to complete
        wait
        echo "Finished batch $((i / BATCH_SIZE + 1)) for arch $arch"
    done

    echo "Completed all seeds for architecture: $arch"
    echo ""
done

echo "===================================================="
echo "All network architecture experiments completed!"
echo "===================================================="