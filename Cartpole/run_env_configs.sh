#!/bin/bash

# Usage: ./run_env_configs.sh <algorithm> <batch_size>
# Example: ./run_env_configs.sh RPI 5
#          ./run_env_configs.sh DQN 3

ALGO=${1:-RPI}
BATCH_SIZE=${2:-5}
wandb_project=Cartpole-Env-Config-GitHub

# Number of parallel jobs per batch
TOTAL_RUNS=10

GRAVITIES=(4.9 19.6)
MASS_CARTS=(0.5 2.0)
MASS_POLES=(0.05 0.2)

echo "========================================="
echo "CartPole Environment Configuration Experiments"
echo "Algorithm: $ALGO"
echo "Batch size: $BATCH_SIZE"
echo "Total runs per configuration: $TOTAL_RUNS"
echo "========================================="

# Create logs directory
mkdir -p logs/env-config

echo "======================================"
echo "Environment Configuration Experiment"
echo "======================================"

for ((i=0; i<TOTAL_RUNS; i+=BATCH_SIZE)); do
    echo "Starting batch $((i / BATCH_SIZE + 1))"

    for ((j=0; j<BATCH_SIZE && i+j<TOTAL_RUNS; j++)); do
        echo "running for seed $((i + j))"

        python -u main_cartpole.py \
        --seed $((i + j)) \
        --algo $ALGO \
        --exp-type env-config \
        --use-wandb \
        --wandb-project $wandb_project \
        > logs/env-config/default_seed_$((i + j)).log 2>&1 & 
        
    done

    wait  
    echo "Finished batch $((i / BATCH_SIZE + 1))"
done

echo "======================================"
echo " Default experiments completed!"
echo "======================================"



for GRAV in "${GRAVITIES[@]}" 
do
    for ((i=0; i<TOTAL_RUNS; i+=BATCH_SIZE)); do
        echo "Starting batch $((i / BATCH_SIZE + 1))"

        for ((j=0; j<BATCH_SIZE && i+j<TOTAL_RUNS; j++)); do
            echo "running for seed $((i + j))"

            python -u main_cartpole.py \
            --seed $((i + j)) \
            --algo $ALGO \
            --exp-type env-config \
            --env-gravity ${GRAV} \
            --use-wandb \
            --wandb-project $wandb_project \
            > logs/env-config/gravity_${GRAV}_seed_$((i + j)).log 2>&1 & 
            
        done

        wait  
        echo "Finished batch $((i / BATCH_SIZE + 1))"
    done
done

echo "======================================"
echo "Gravity experiments completed!"
echo "======================================"




for MSSCRT in "${MASS_CARTS[@]}" 
do
    for ((i=0; i<TOTAL_RUNS; i+=BATCH_SIZE)); do
        echo "Starting batch $((i / BATCH_SIZE + 1))"

        for ((j=0; j<BATCH_SIZE && i+j<TOTAL_RUNS; j++)); do
            echo "running for seed $((i + j))"

            python -u main_cartpole.py \
            --seed $((i + j)) \
            --algo $ALGO \
            --exp-type env-config \
            --env-masscart ${MSSCRT} \
            --use-wandb \
            --wandb-project $wandb_project \
            > logs/env-config/masscart_${MSSCRT}_seed_$((i + j)).log 2>&1 & 
            
        done

        wait  
        echo "Finished batch $((i / BATCH_SIZE + 1))"
    done
done

echo "======================================"
echo " MassCart experiments completed!"
echo "======================================"



for MSSPL in "${MASS_POLES[@]}" 
do
    for ((i=0; i<TOTAL_RUNS; i+=BATCH_SIZE)); do
        echo "Starting batch $((i / BATCH_SIZE + 1))"

        for ((j=0; j<BATCH_SIZE && i+j<TOTAL_RUNS; j++)); do
            echo "running for seed $((i + j))"

            python -u main_cartpole.py \
            --seed $((i + j)) \
            --algo $ALGO \
            --exp-type env-config \
            --env-masspole ${MSSPL} \
            --use-wandb \
            --wandb-project $wandb_project \
            > logs/env-config/masspole_${MSSPL}_seed_$((i + j)).log 2>&1 & 
            
        done

        wait  
        echo "Finished batch $((i / BATCH_SIZE + 1))"
    done
done

echo "======================================"
echo " MassPole experiments completed!"
echo "======================================"




echo "======================================"
echo "All experiments completed!"
echo "======================================"
