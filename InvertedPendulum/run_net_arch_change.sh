#!/bin/bash
 
algo=$1
BATCH_SIZE=${2:-5}

wandb_project="Inverted-Pendulum-NetArch-GitHub"

mkdir -p logs/net-arch/
# Total number of runs
ARCHS=("[32, 32]" "[64, 64]" "[128, 128]" "[256, 256]" "[400, 300]" "[512, 512]" )
 
TOTAL_RUNS=10
 
# Number of parallel jobs per batch

 
# --- Main loop to iterate over each architecture ---
for arch in "${ARCHS[@]}"; do
    # Create a clean version of the arch string for the log file name
    # e.g., "[256, 256]" becomes "256_256"
    echo "----------------------------------------------------"
    echo "Starting runs for architecture: $arch"
    echo "----------------------------------------------------"

    # Create a clean arch string for logging (e.g., "[128, 128]" -> "128-128")
    arch_clean=$(echo "$arch" | tr -d '[],' | tr ' ' '-')
 
    # --- Your original batching logic, now nested inside the arch loop ---
    for ((i=0; i<TOTAL_RUNS; i+=BATCH_SIZE)); do
        echo "Starting batch $((i / BATCH_SIZE + 1)) for arch $arch $arch_clean"
 
        for ((j=0; j<BATCH_SIZE && i+j<TOTAL_RUNS; j++)); do
            current_seed=$((i + j))
            echo "Running seed $current_seed for arch $arch"
 
            python -u main.py \
            --env 'InvertedPendulum-v5' \
            --seed $current_seed \
            --policy $algo \
            --net_arch "$arch" \
            --wandb_project $wandb_project \
            --experiment_name net-arch \
            --use_wandb \
            > logs/net-arch/InvertedPendulum-v5_${algo}_arch_${arch_clean}_seed_${current_seed}.log 2>&1 &
        done
 
        wait
        echo "Finished batch $((i / BATCH_SIZE + 1)) for arch $arch"
    done
done
 
echo "All runs for all architectures completed."
