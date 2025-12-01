#!/bin/bash

# Environment Configuration Experiment
# Tests different cart mass, pole mass, and gravity values

POLICY=$1
BATCH_SIZE=${2:-5}

# Configuration
TOTAL_RUNS=10
ENV="InvertedPendulum-v5"
wandb_project="Inverted-Pendulum-Env-Config-GitHub"

# # Default environment parameters (from InvertedPendulum-v5)
# DEFAULT_CART_MASS=10.47197551
# DEFAULT_POLE_MASS=5.01859164
# DEFAULT_GRAVITY=-9.81


CART_MASSES=(20.94395102 5.235987755)
POLE_MASSES=(10.03718328 2.50929582)
GRAVITIES=(-4.905 -19.62)


# Create logs directory
mkdir -p logs/env-config

echo "======================================"
echo "Environment Configuration Experiment"
echo "======================================"



for ((i=0; i<TOTAL_RUNS; i+=BATCH_SIZE)); do
    echo "Starting batch $((i / BATCH_SIZE + 1))"

    for ((j=0; j<BATCH_SIZE && i+j<TOTAL_RUNS; j++)); do
        echo "running for seed $((i + j))"
        python -u main.py \
        --env $ENV \
        --policy $POLICY \
        --seed $((i + j)) \
        --use_wandb \
        --wandb_project $wandb_project \
        --experiment_name "env-config" \
        > logs/env-config/default_seed_$((i + j)).log 2>&1 &

    done

    wait  
    echo "Finished batch $((i / BATCH_SIZE + 1))"
done

echo "======================================"
echo " Default experiments completed!       "
echo "======================================"


echo "======================================"
echo "Starting Gravity experiments!        "
echo "======================================"

for GRAV in "${GRAVITIES[@]}" 
do
    for ((i=0; i<TOTAL_RUNS; i+=BATCH_SIZE)); do
        echo "Starting batch $((i / BATCH_SIZE + 1))"

        for ((j=0; j<BATCH_SIZE && i+j<TOTAL_RUNS; j++)); do
            echo "running for seed $((i + j))"

            python -u main.py \
                --env $ENV \
                --policy $POLICY \
                --seed $((i + j)) \
                --use_wandb \
                --wandb_project $wandb_project \
                --experiment_name "env-config" \
                --gravity $GRAV \
                > logs/env-config/gravity_${GRAV}_seed_$((i + j)).log 2>&1 &
        done

        wait  
        echo "Finished batch $((i / BATCH_SIZE + 1))"
    done
done

echo "======================================"
echo "Gravity experiments completed!        "
echo "======================================"


echo "======================================"
echo "Starting Mass Cart experiments!        "
echo "======================================"

for MSSCRT in "${CART_MASSES[@]}" 
do
    for ((i=0; i<TOTAL_RUNS; i+=BATCH_SIZE)); do
        echo "Starting batch $((i / BATCH_SIZE + 1))"

        for ((j=0; j<BATCH_SIZE && i+j<TOTAL_RUNS; j++)); do
            echo "running for seed $((i + j))"

            python -u main.py \
                --env $ENV \
                --policy $POLICY \
                --seed $((i + j)) \
                --use_wandb \
                --wandb_project $wandb_project \
                --experiment_name "env-config" \
                --cart_mass $MSSCRT \
                > logs/env-config/mass_cart_${MSSCRT}_seed_$((i + j)).log 2>&1 &
        done

        wait  
        echo "Finished batch $((i / BATCH_SIZE + 1))"
    done
done

echo "======================================"
echo "Mass Cart experiments completed!      "
echo "======================================"



echo "======================================"
echo "Starting Pole Masses experiments!        "
echo "======================================"

for PLMS in "${POLE_MASSES[@]}" 
do
    for ((i=0; i<TOTAL_RUNS; i+=BATCH_SIZE)); do
        echo "Starting batch $((i / BATCH_SIZE + 1))"

        for ((j=0; j<BATCH_SIZE && i+j<TOTAL_RUNS; j++)); do
            echo "running for seed $((i + j))"

            python -u main.py \
                --env $ENV \
                --policy $POLICY \
                --seed $((i + j)) \
                --use_wandb \
                --wandb_project $wandb_project \
                --experiment_name "env-config" \
                --pole_mass $PLMS \
                > logs/env-config/mass_pole_${PLMS}_seed_$((i + j)).log 2>&1 &
        done

        wait  
        echo "Finished batch $((i / BATCH_SIZE + 1))"
    done
done

echo "======================================"
echo "Pole Masses experiments completed!    "
echo "======================================"



echo "======================================"
echo "All experiments completed!            "
echo "======================================"