"""
PPO Training for InvertedPendulum-v5
Hyperparameters from RL Baselines3 Zoo:
https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml#L536
"""

import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList
import mujoco

from custom_eval import EvalWithCriticCallback
import wandb
from wandb.integration.sb3 import WandbCallback


def parse_args():
    parser = argparse.ArgumentParser(description='PPO Training for InvertedPendulum-v5')

    # Basic arguments
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu or cuda)')
    parser.add_argument('--env', type=str, default='InvertedPendulum-v5', help='Gym environment ID')
    
    parser.add_argument('--exp-type', type=str, default='net-archh',
                        choices=['net-archh', 'env-change'],
                        help='Experiment type: net-archh or env-change')

    # Network architecture
    parser.add_argument('--net-arch', type=str, default=None,
                        help='Network architecture (comma-separated, e.g., "256,256")')

    # Training parameters
    parser.add_argument('--n-timesteps', type=int, default=100000, help='Total training timesteps')
    parser.add_argument('--n-envs', type=int, default=1, help='Number of parallel environments')

    # Evaluation
    parser.add_argument('--eval-freq', type=int, default=100, help='Evaluation frequency (timesteps)')
    parser.add_argument('--n-eval-episodes', type=int, default=100, help='Number of evaluation episodes')

    # PPO Hyperparameters (from RL Baselines3 Zoo)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--n-steps', type=int, default=32)
    # parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--learning-rate', type=float, default=0.000222425)
    parser.add_argument('--ent-coef', type=float, default=1.37976e-07)
    parser.add_argument('--clip-range', type=float, default=0.4)
    parser.add_argument('--n-epochs', type=int, default=5)
    parser.add_argument('--gae-lambda', type=float, default=0.9)
    parser.add_argument('--max-grad-norm', type=float, default=0.3)
    parser.add_argument('--vf-coef', type=float, default=0.19816)

    # Logging
    parser.add_argument('--use-wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='ppo-inverted-pendulum',
                        help='Weights & Biases project name')
    
    # Env config overrides
    parser.add_argument('--env-gravity', type=float, default=None,
                        help='Environment gravity (overrides preset)')
    parser.add_argument('--env-masscart', type=float, default=None,
                        help='Environment cart mass (overrides preset)')
    parser.add_argument('--env-masspole', type=float, default=None,
                        help='Environment pole mass (overrides preset)')

    return parser.parse_args()


args = parse_args()

# Parse network architecture
if args.net_arch is not None:
    try:
        net_arch = [int(s) for s in args.net_arch.split(',')]
        arch_str = '-'.join(map(str, net_arch))
    except ValueError:
        print(f"Error: Invalid --net-arch format: {args.net_arch}. Use comma-separated integers (e.g., '256,256')")
        exit(1)
else:
    arch_str = 'default'
    # net_arch = 'default'


env_config = {'gravity': -9.81, 'cart_mass': 10.47197551, 'pole_mass': 5.01859164}
# Override with individual parameters if provided
if args.env_gravity is not None:
    env_config['gravity'] = args.env_gravity
if args.env_masscart is not None:
    env_config['cart_mass'] = args.env_masscart
if args.env_masspole is not None:
    env_config['pole_mass'] = args.env_masspole


# Setup paths and run name based on experiment type
if args.exp_type == 'net-archh':
    # Network architecture experiment: results/net-arch/{env}/PPO/arch_{arch_str}/seed_{seed}
    save_path = f"./results/net-arch/{args.env}/PPO/arch_{arch_str}/seed_{args.seed}"
    run_name = f"PPO_netarch_{arch_str}_seed_{args.seed}"

elif args.exp_type == 'env-change':
    # Environment change experiment: results/env-change/{env}/PPO/arch_{arch_str}_gravity_{g}_masscart_{mc}_masspole_{mp}/seed_{seed}
    save_path = f"./results/env-change/{args.env}/PPO/arch_{arch_str}_gravity_{env_config['gravity']}_masscart_{env_config['cart_mass']}_masspole_{env_config['pole_mass']}/seed_{args.seed}"
    run_name = f"PPO_envchange_arch_{arch_str}_g_{env_config['gravity']}_mc_{env_config['cart_mass']}_mp_{env_config['pole_mass']}_seed_{args.seed}"
else:
    raise ValueError(f"Unknown experiment type: {args.exp_type}")

# Initialize WandB
if args.use_wandb:
    run = wandb.init(
        name=run_name,
        project=args.wandb_project,
        config=vars(args),
        sync_tensorboard=True,
        save_code=True,
    )
    run.define_metric("eval/*", step_metric="global_step")

# Print configuration
print("=" * 70)
print(f"PPO Training: {args.env}")
print(f"Seed: {args.seed} | Device: {args.device} | Architecture: {args.net_arch}")
print(f"Timesteps: {args.n_timesteps} | Eval freq: {args.eval_freq}")
print("=" * 70)

def make_env():
    
    # Modify environment parameters if provided
    env = gym.make("InvertedPendulum-v5")
    # Find the Body IDs
    model = env.unwrapped.model
    body_ids = {}
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name:
            body_ids[name] = i

    print(f"Found Body IDs: {body_ids}")

    # Get the specific IDs we need
    cart_id = body_ids.get("cart")
    pole_id = body_ids.get("pole")

    

    # Modify body masses if provided
    if args.env_masscart is not None:
        print(f"Original cart mass: Train: {model.body_mass[cart_id]}")
        model.body_mass[cart_id] = args.env_masscart
        print(f"New cart mass: Train: {model.body_mass[cart_id]}")

    if args.env_masspole is not None:
        print(f"Original pole mass: Train: {model.body_mass[pole_id]}")
        model.body_mass[pole_id] = args.env_masspole
        print(f"New pole mass: Train: {model.body_mass[pole_id]}")

    # Modify gravity if provided
    if args.env_gravity is not None:
        print(f"Original gravity: Train: {model.opt.gravity.copy()}")
        model.opt.gravity[2] = args.env_gravity
        print(f"New gravity: Train: {model.opt.gravity}")
    return env


# Setup environments
env = make_vec_env(make_env, n_envs=args.n_envs, seed=args.seed)
eval_env = DummyVecEnv([lambda: Monitor(make_env())])

# env = make_vec_env(args.env, n_envs=args.n_envs, seed=args.seed)
# eval_env = DummyVecEnv([lambda: Monitor(gym.make(args.env))])
eval_env.seed(args.seed)

# Create policy kwargs
if args.net_arch is not None:
    policy_kwargs = dict(net_arch={"pi": net_arch, "vf": net_arch})

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        normalize_advantage=True,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        gamma=0.999,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
        n_epochs=args.n_epochs,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        vf_coef=args.vf_coef,
        verbose=1,
        seed=args.seed,
        device=args.device,
        policy_kwargs=policy_kwargs,
        tensorboard_log=f"runs/{run_name}" if args.use_wandb else None,
    )
else:
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        normalize_advantage=True,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        gamma=0.999,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
        n_epochs=args.n_epochs,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        vf_coef=args.vf_coef,
        verbose=1,
        seed=args.seed,
        device=args.device,
        tensorboard_log=f"runs/{run_name}" if args.use_wandb else None,
    )


# Setup callbacks
eval_callback = EvalWithCriticCallback(
    eval_env=eval_env,
    eval_freq=args.eval_freq,
    n_eval_episodes=args.n_eval_episodes,
    save_path=save_path,
    algo="PPO",
    verbose=1,
    gamma=0.99
)

if args.use_wandb:
    wandb_callback = WandbCallback(model_save_path=f"models/{run.id}", verbose=2)
    callbacks = CallbackList([eval_callback, wandb_callback])
else:
    callbacks = eval_callback

# Train the model
print("\nStarting training...")
model.learn(total_timesteps=args.n_timesteps, callback=callbacks)
print(f"\nTraining complete! Results saved to: {save_path}")

if args.use_wandb:
    run.finish()