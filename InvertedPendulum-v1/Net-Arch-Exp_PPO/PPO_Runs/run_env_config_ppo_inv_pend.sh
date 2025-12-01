#!/bin/bash

# Network Architecture Experiment for PPO on InvertedPendulum-v5
# Tests different network architectures with multiple seeds

# Network architectures to test


# Number of seeds per architecture
TOTAL_RUNS=10

# Number of parallel jobs per batch
BATCH_SIZE=${1:-10}

# Training configuration
N_TIMESTEPS=100_000
EVAL_FREQ=1000
N_EVAL_EPISODES=100
DEVICE="cpu"

# Create logs directory if it doesn't exist
mkdir -p logs/env-config

# Environment Configuration Experiment
# Tests different cart mass, pole mass, and gravity values

wandb_project="Inverted-Pendulum-Env-Config-GitHub"

# # Default environment parameters (from InvertedPendulum-v5)
# DEFAULT_CART_MASS=10.47197551
# DEFAULT_POLE_MASS=5.01859164
# DEFAULT_GRAVITY=-9.81


CART_MASSES=(20.94395102 5.235987755)
POLE_MASSES=(10.03718328 2.50929582)
GRAVITIES=(-4.905 -19.62)


echo "======================================"
echo "Gravity experiments Started!        "
echo "======================================"


for GRAV in "${GRAVITIES[@]}" 
do
    for ((i=0; i<TOTAL_RUNS; i+=BATCH_SIZE)); do
        echo "Starting batch $((i / BATCH_SIZE + 1))"

        for ((j=0; j<BATCH_SIZE && i+j<TOTAL_RUNS; j++)); do
            echo "running for seed $((i + j))"

            python -u main_ppo_invpend.py \
                --seed $((i + j)) \
                --device $DEVICE \
                --exp-type env-change \
                --n-timesteps $N_TIMESTEPS \
                --eval-freq $EVAL_FREQ \
                --n-eval-episodes $N_EVAL_EPISODES \
                --use-wandb \
                --wandb-project $wandb_project \
                --env-gravity $GRAV \
                > logs/env-config/InvertedPendulum-v5_PPO_gravity_${GRAV}_seed_$((i + j)).log 2>&1 &

        done

        wait  
        echo "Finished batch $((i / BATCH_SIZE + 1))"
    done
done

echo "======================================"
echo "Gravity experiments completed!        "
echo "======================================"



echo "======================================"
echo "Mass Cart experiments Started!        "
echo "======================================"

for MSSCRT in "${CART_MASSES[@]}" 
do
    for ((i=0; i<TOTAL_RUNS; i+=BATCH_SIZE)); do
        echo "Starting batch $((i / BATCH_SIZE + 1))"

        for ((j=0; j<BATCH_SIZE && i+j<TOTAL_RUNS; j++)); do
            echo "running for seed $((i + j))"


            python -u main_ppo_invpend.py \
                --seed $((i + j)) \
                --device $DEVICE \
                --exp-type env-change \
                --n-timesteps $N_TIMESTEPS \
                --eval-freq $EVAL_FREQ \
                --n-eval-episodes $N_EVAL_EPISODES \
                --use-wandb \
                --wandb-project $wandb_project \
                --env-masscart $MSSCRT \
                > logs/env-config/InvertedPendulum-v5_PPO_cart_mass_${MSSCRT}_seed_$((i + j)).log 2>&1 &

        done

        wait  
        echo "Finished batch $((i / BATCH_SIZE + 1))"
    done
done

echo "======================================"
echo "Mass Cart experiments completed!      "
echo "======================================"



echo "======================================"
echo "Pole Masses experiments started!    "
echo "======================================"


for PLMS in "${POLE_MASSES[@]}" 
do
    for ((i=0; i<TOTAL_RUNS; i+=BATCH_SIZE)); do
        echo "Starting batch $((i / BATCH_SIZE + 1))"

        for ((j=0; j<BATCH_SIZE && i+j<TOTAL_RUNS; j++)); do
            echo "running for seed $((i + j))"

            python -u main_ppo_invpend.py \
                --seed $((i + j)) \
                --device $DEVICE \
                --exp-type env-change \
                --n-timesteps $N_TIMESTEPS \
                --eval-freq $EVAL_FREQ \
                --n-eval-episodes $N_EVAL_EPISODES \
                --use-wandb \
                --wandb-project $wandb_project \
                --env-masspole $PLMS \
                > logs/env-config/InvertedPendulum-v5_PPO_pole_mass_${PLMS}_seed_$((i + j)).log 2>&1 &
        done
        wait  
        echo "Finished batch $((i / BATCH_SIZE + 1))"
    done
done

echo "======================================"
echo "Pole Masses experiments completed!    "
echo "======================================"





echo "======================================"
echo "Environment Configuration Experiment"
echo "======================================"



# for ((i=0; i<TOTAL_RUNS; i+=BATCH_SIZE)); do
#     echo "Starting batch $((i / BATCH_SIZE + 1))"

#     for ((j=0; j<BATCH_SIZE && i+j<TOTAL_RUNS; j++)); do
#         echo "running for seed $((i + j))"
#         python -u main.py \
#         --env $ENV \
#         --policy $POLICY \
#         --seed $((i + j)) \
#         --use_wandb \
#         --wandb_project $wandb_project \
#         --experiment_name "env-config" \
#         > logs/env-config/default_seed_$((i + j)).log 2>&1 &

#     done

#     wait  
#     echo "Finished batch $((i / BATCH_SIZE + 1))"
# done

# echo "======================================"
# echo " Default experiments completed!       "
# echo "======================================"







echo "======================================"
echo "All experiments completed!            "
echo "======================================"






# # #################### Old Code ####################


# # --- Main loop to iterate over each architecture ---

# # Replace commas with underscores for log file names
# echo "===================================================="
# echo "Starting runs "
# echo "===================================================="

# # --- Batching logic for parallel execution ---
# for ((i=0; i<NUM_SEEDS; i+=BATCH_SIZE)); do
#     echo "Starting batch $((i / BATCH_SIZE + 1))"

#     for ((j=0; j<BATCH_SIZE && i+j<NUM_SEEDS; j++)); do
#         python -u main_ppo_invpend.py \
#             --seed $((i + j)) \
#             --device $DEVICE \
#             --n-timesteps $N_TIMESTEPS \
#             --eval-freq $EVAL_FREQ \
#             --n-eval-episodes $N_EVAL_EPISODES \
#             --use-wandb \
#             > logs/InvertedPendulum-v5_PPO_arch_${arch_clean}_seed_$((i + j)).log 2>&1 &
#     done

#     # Wait for all jobs in this batch to complete
#     wait
#     echo "Finished batch $((i / BATCH_SIZE + 1))"
# done

# echo "Completed all seeds"
# echo ""


# echo "===================================================="
# echo "All network architecture experiments completed!"
# echo "===================================================="