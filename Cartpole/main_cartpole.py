"""
CartPole Reinforcement Learning Experiments
Supports three experiment types:
  1. net-arch: Network architecture experiments (varying width/depth)
  2. penalty: Penalty function experiments (different loss functions for RPI)
  3. sensitivity: Sensitivity experiments (varying c ,lambda1 and possibly lambda2 hyperparameters)
"""

import argparse
import os
import gymnasium as gym
from stable_baselines3 import DQN, PPO
# from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from typing import Callable
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import CallbackList

from my_sb3_rpi_dqn import RPI as RPI_DQN
from my_sb3_double_dqn import DoubleDQN
from custom_eval import EvalWithCriticCallback


# Linear schedule helper function
# https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
    current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func




def parse_args():
    parser = argparse.ArgumentParser(description='CartPole RL Experiments')

    # Basic arguments
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    parser.add_argument('--algo', type=str, required=True,
                        choices=['DQN', 'DoubleDQN', 'RPI', 'PPO'],
                        help='Algorithm to use')
    parser.add_argument('--exp-type', type=str, required=True,
                        choices=['net-arch', 'penalty', 'env-config'],
                        help='Experiment type')

    # Network architecture
    parser.add_argument('--net-width', type=int, default=None,
                        help='Network width (neurons per layer)')
    parser.add_argument('--net-depth', type=int, default=None,
                        choices=[1,2],
                        help='Network depth: single or double layer')

    # Training parameters
    parser.add_argument('--n-timesteps', type=int, default=100_000,
                        help='Total training timesteps')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu or cuda)')
    parser.add_argument('--eval-freq', type=int, default=100,
                        help='Evaluation frequency (in timesteps)')
    parser.add_argument('--n-eval-episodes', type=int, default=100,
                        help='Number of episodes per evaluation')

    # RPI-specific parameters
    parser.add_argument('--c', type=float, default=1.0,
                        help='RPI parameter c')
    parser.add_argument('--lambda1', type=float, default=10.0,
                        help='RPI parameter lambda1')
    parser.add_argument('--lambda2', type=float, default=2.0,
                        help='RPI parameter lambda2')
    parser.add_argument('--r-min', type=float, default=1.0,
                        help='RPI parameter r_min')
    parser.add_argument('--penalty-function', type=str, default='relu',
                        choices=['relu', 'quadratic', 'cubic', 'exponential'],
                        help='Penalty function for RPI')

    # Dynamic lambda1 parameters
    parser.add_argument('--dynamic-lambda1', action='store_true',
                        help='Enable dynamic lambda1 switching based on TD error')
    parser.add_argument('--lambda1-b01', type=float, default=500,
                        help='Lambda1 value for small TD errors (|TD| < threshold)')
    parser.add_argument('--lambda1-threshold', type=float, default=1.0,
                        help='Threshold for switching between lambda1 and lambda1-b01')


    # Logging
    parser.add_argument('--use-wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='cartpole-rl',
                        help='Weights & Biases project name')
    
    # Used only in DQN, DDQN and RPI
    parser.add_argument('--buffer_size', type=int, default=100000)
    parser.add_argument('--learning_starts', type=int, default=1000)
    parser.add_argument('--train_freq', type=int, default=256)

    # CartPole environment configuration # TODO: pass correctly from command line
    parser.add_argument('--env-gravity', type=float, default=None,
                        help='Environment gravity (overrides preset)')
    parser.add_argument('--env-masscart', type=float, default=None,
                        help='Environment cart mass (overrides preset)')
    parser.add_argument('--env-masspole', type=float, default=None,
                        help='Environment pole mass (overrides preset)')
    parser.add_argument('--env-length', type=float, default=None,
                        help='Environment pole length (overrides preset)')
    

    return parser.parse_args()


def main():
    args = parse_args()
    print("Args:", args)

    # DQN/DoubleDQN/RPI hyperparameters (from SB3 zoo)
    # https://github.com/DLR-RM/rl-baselines3-zoo/blob/ab4aadb57c6c42abcf1016318c2bebe35c4c1270/hyperparams/dqn.yml#L20
    # DQN/DoubleDQN/RPI hyperparameters dictionary
    hyperparams = {
        'policy': 'MlpPolicy',
        'learning_rate': 2.3e-3,
        'batch_size': 64,
        'buffer_size': args.buffer_size,
        'learning_starts': args.learning_starts,
        'gamma': 0.99,
        'target_update_interval': 10,
        'train_freq': args.train_freq,
        'gradient_steps': 128,
        'exploration_fraction': 0.16,
        'exploration_final_eps': 0.04,
        'policy_kwargs': dict(net_arch=[256, 256]),
        'verbose': 1,
        'seed': args.seed,
        'device': args.device
    }
    if args.net_width is not None or args.net_depth is not None:
        # Need to change both. Else the script will fail
        hyperparams['policy_kwargs'] = dict(net_arch=[args.net_width] * args.net_depth)

    # PPO hyperparameters dictionary
    # Taken from SB3 zoo:
    # https://github.com/DLR-RM/rl-baselines3-zoo/blob/ab4aadb57c6c42abcf1016318c2bebe35c4c1270/hyperparams/ppo.yml#L32

    ppo_hyperparameters = {
        'policy': 'MlpPolicy',
        'learning_rate': linear_schedule(0.001),
        'n_steps': 32,
        'batch_size': 256,
        'n_epochs': 20,
        'gamma': 0.99,
        'gae_lambda': 0.8,
        'clip_range': linear_schedule(0.2),
        'ent_coef': 0.0,
        'verbose': 1,
        'seed': args.seed,
        'device': args.device
    }

    if args.net_width is not None or args.net_depth is not None:
        # Need to change both. Else the script will fail
        ppo_hyperparameters['policy_kwargs'] = dict(net_arch=[args.net_width] * args.net_depth)

    # Get environment configuration (will be used later and logged to WandB)
    # Default values are taken from https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/cartpole.py
    env_config = {'gravity': 9.8, 'masscart': 1.0, 'masspole': 0.1, 'length': 0.5}
    # Override with individual parameters if provided
    if args.env_gravity is not None:
        env_config['gravity'] = args.env_gravity
    if args.env_masscart is not None:
        env_config['masscart'] = args.env_masscart
    if args.env_masspole is not None:
        env_config['masspole'] = args.env_masspole
    if args.env_length is not None:
        env_config['length'] = args.env_length

    # Setup environments with custom configuration
    # https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#example
    def make_env():
        env = gym.make("CartPole-v1")
        # Apply custom configuration
        env.unwrapped.gravity = env_config['gravity']
        env.unwrapped.masscart = env_config['masscart']
        env.unwrapped.masspole = env_config['masspole']
        env.unwrapped.length = env_config['length']
        env.unwrapped.total_mass = env.unwrapped.masspole + env.unwrapped.masscart
        env.unwrapped.polemass_length = env.unwrapped.masspole * env.unwrapped.length
        return Monitor(env)

    if args.algo == 'PPO':
        n_envs = 8
        env = DummyVecEnv([make_env for _ in range(n_envs)])
    else:
        n_envs = 1
        env = DummyVecEnv([make_env])
    
    eval_env = DummyVecEnv([make_env])
    
    env.seed(args.seed)
    eval_env.seed(args.seed)

    # Generate output paths based on experiment type
    if args.exp_type == 'net-arch':
        folder_path = f"./results/net-arch/{args.algo}/width_{args.net_width}_depth_{args.net_depth}/seed_{args.seed}"
        run_name = f"{args.algo}_width_{args.net_width}_depth_{args.net_depth}_seed_{args.seed}"
    
    elif args.exp_type == 'penalty':
        dyn = "_dynLambda1" if args.dynamic_lambda1 else ""
        folder_path = f"./results/penalty/{args.penalty_function}{dyn}/seed_{args.seed}"
        run_name = f"{args.penalty_function}{dyn}_seed_{args.seed}"
    
    elif args.exp_type == 'env-config':
        folder_path = f"./results/env-config/{args.algo}/g_{env_config['gravity']}_mc_{env_config['masscart']}_mp_{env_config['masspole']}_l_{env_config['length']}/{args.algo}/seed_{args.seed}"
        run_name = f"{args.algo}_g_{env_config['gravity']}_mc_{env_config['masscart']}_mp_{env_config['masspole']}_l_{env_config['length']}_seed_{args.seed}"
    
    else:
        raise ValueError(f"Unknown experiment type: {args.exp_type}")

    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(f"./logs/{args.exp_type}", exist_ok=True)

    # Initialize WandB if requested
    if args.use_wandb:
        import wandb
        config_dict = vars(args).copy()
        # Add actual environment config values to WandB config
        config_dict['env_config'] = env_config
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=config_dict,
            sync_tensorboard=True
        )
        # Log environment config as summary metrics (visible in WandB UI)
        wandb.config.update({
            'env/gravity': env_config['gravity'],
            'env/masscart': env_config['masscart'],
            'env/masspole': env_config['masspole'],
            'env/length': env_config['length'],
        }, allow_val_change=True)

    # Print configuration
    print("\n" + "="*80)
    print(f"Running {args.algo} on CartPole-v1")
    print(f"Experiment Type: {args.exp_type}")
    print(f"Seed: {args.seed} | Device: {args.device}")
    print(f"Network: width={args.net_width}, depth={args.net_depth}")


    print(f"Environment Config: ")
    print(f"  gravity={env_config['gravity']}, masscart={env_config['masscart']}, "
            f"masspole={env_config['masspole']} length={env_config['length']}")

    if args.algo == 'RPI':
        print(
            f"RPI params: c={args.c}, lambda1={args.lambda1}, lambda2={args.lambda2}, r_min={args.r_min}")
        print(f"Penalty function: {args.penalty_function}")
        if args.dynamic_lambda1:
            print(
                f"Dynamic lambda1: enabled (lambda1_b01={args.lambda1_b01}, threshold={args.lambda1_threshold})")

    print(
        f"Training: {args.n_timesteps} timesteps | Eval freq: {args.eval_freq} | Eval episodes: {args.n_eval_episodes}")
    if args.use_wandb:
        print(
            f"WandB: Logging to project '{args.wandb_project}' as run '{run_name}'")
    print("="*80 + "\n")

    # Create model based on algorithm (using **hyperparams to unpack the dictionary)
    if args.algo == "DQN":
        model = DQN(env=env, **hyperparams)

    elif args.algo == "DoubleDQN":
        model = DoubleDQN(env=env, **hyperparams)

    elif args.algo == "RPI":
        model = RPI_DQN(
            env=env,

            **hyperparams,

            my_c=args.c,
            my_lambda1=args.lambda1,
            my_lambda2=args.lambda2,
            my_r_min=args.r_min,
            my_lambda1_b01=args.lambda1_b01,
            penalty_function=args.penalty_function,
            dynamic_lambda1=args.dynamic_lambda1,
            lambda1_threshold=args.lambda1_threshold,
            use_wandb=args.use_wandb
        )

    elif args.algo == "PPO":
        model = PPO(env=env, **ppo_hyperparameters)

    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")

    # Setup evaluation callback
    eval_callback = EvalWithCriticCallback(
        eval_env=eval_env,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        save_path=folder_path,
        algo=args.algo,
        verbose=1,
        gamma=hyperparams['gamma']
    )

    # Setup callbacks list
    if args.use_wandb:
        import wandb
        wandb_callback = WandbCallback(
            model_save_path=f"models/{wandb.run.id}",
            verbose=2,
        )
        callback_list = CallbackList([eval_callback, wandb_callback])
    else:
        callback_list = eval_callback

    # Train the model
    print("Starting training...")
    model.learn(total_timesteps=args.n_timesteps, callback=callback_list)
    print(f"\nTraining complete! Results saved to: {folder_path}")


if __name__ == "__main__":
    main()
