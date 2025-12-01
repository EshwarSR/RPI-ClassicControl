import numpy as np
import torch
import gymnasium as gym
import mujoco
import argparse
import os
from itertools import count
import ast # Import ast to safely evaluate the list string

import random
import utils
import ALGO_TD3 as TD3
import ALGO_DDPG as DDPG
# import TD3_DDPG
import ALGO_DDPG_RPI as DDPG_RPI

import wandb


def eval_policy(policy, eval_env, seed, eval_episodes, gamma, policy_name, device, cart_mass=None, pole_mass=None, gravity=None):
	"""Evaluate policy for X episodes and return metrics"""
	all_tot_disc_rews = []
	all_tot_rews = []
	all_q_hats = []
	episode_lengths = []

	for i_episode in range(eval_episodes):
		tot_disc_rew = 0
		tot_rew = 0
		state, _ = eval_env.reset(seed=(108 * seed)+i_episode)
		start_state = state.copy()
		start_state_tensor = torch.FloatTensor(start_state.reshape(1, -1)).to(device)
		
		for t in count():
			with torch.no_grad():
				action = policy.select_action(np.array(state))
			if t == 0:
				start_action_tensor = torch.FloatTensor(action.reshape(1, -1)).to(device)
			state, reward, terminated, truncated, _ = eval_env.step(action)
			tot_disc_rew += (gamma ** t) * reward
			tot_rew += reward
			if terminated or truncated:
				with torch.no_grad():
					if policy_name in ["TD3"]:
						q_hat = policy.critic.Q1(start_state_tensor, start_action_tensor)[0].item()
					elif policy_name in ["DDPG_RPI", "DDPG"]:
						q_hat = policy.critic(start_state_tensor, start_action_tensor)[0].item()
				all_q_hats.append(q_hat)
				all_tot_disc_rews.append(tot_disc_rew)
				episode_lengths.append(t+1)
				all_tot_rews.append(tot_rew)
				break

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: Eps Length {np.mean(episode_lengths):.3f} | Eps Tot Rew {np.mean(all_tot_rews):.3f} | Eps Disc Rew {np.mean(all_tot_disc_rews):.3f} | Eps Q_hat {np.mean(all_q_hats):.3f}")
	print("---------------------------------------")
	return all_tot_disc_rews, all_q_hats, episode_lengths, all_tot_rews


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG, DDPG_RPI)
	parser.add_argument("--env", default="InvertedPendulum-v5")          # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=1000, type=int)# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=1000, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=100_000, type=int)   # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1, type=float)    # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
	parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--save_model", action="store_true", default=True)        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	parser.add_argument("--eval_episodes", default=100)              # No of episodes to consider for evaluation

	# Our algo params
	parser.add_argument("--c", default=2.0, type=float)              			# c of our algo
	parser.add_argument("--lambda1", default=10.0, type=float)              	# lambda1 of our algo
	parser.add_argument("--lambda1_b01", default=100.0, type=float)             # lambda1 when |TD error| < 1
	parser.add_argument("--lambda2", default=2.0, type=float)              		# lambda2 of our algo
	parser.add_argument("--r_min", default=1.0, type=float)              		# r_min of our algo

	# Network architecture parameter
	parser.add_argument("--net_arch", default=None, type=str, help='Custom network architecture, e.g., "[256, 256]"')

	# Penalty function and dynamic lambda parameters
	parser.add_argument("--penalty_function", default="relu", type=str, choices=["relu", "quadratic", "cubic"], help="Penalty function for RPI loss term2")
	parser.add_argument("--use_dynamic_lambda_one", action="store_true", help="Use dynamic lambda1 switching based on TD error magnitude")

	# Gradient clipping parameter
	parser.add_argument("--enable_clipping", action="store_true", help="Enable gradient clipping for critic")
	parser.add_argument("--max_grad_norm", default=10.0, type=float, help="Maximum gradient norm for clipping (critic only)")
	

	# Experiment organization parameter
	parser.add_argument("--experiment_name", default="net-arch", type=str, choices=["net-arch", "penalty-fn", "sensitivity", "env-config"], help="Experiment type for organizing results")

	parser.add_argument("--device", default="cpu")
 
	parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
	parser.add_argument("--wandb_project", default="DDPG_ARCH_TUNING-SENSITIVITY--CONSISTENT-CLIPPING", type=str, help="WandB project name")
	parser.add_argument("--wandb_entity", default=None, type=str, help="WandB entity (user or team name)")
 
	parser.add_argument("--cart_mass", type=float, default=None)
	parser.add_argument("--pole_mass", type=float, default=None)
	parser.add_argument("--gravity", type=float, default=None)
	
	args = parser.parse_args()
	
	print("Args:", args)
 
	# --- Initialize WandB ---
	# Modify environment parameters if provided
	env_config = {'gravity': -9.81, 'cart_mass': 10.47197551, 'pole_mass': 5.01859164}
	# Override with individual parameters if provided
	if args.gravity is not None:
		env_config['gravity'] = args.gravity
	if args.cart_mass is not None:
		env_config['cart_mass'] = args.cart_mass
	if args.pole_mass is not None:
		env_config['pole_mass'] = args.pole_mass
	
	if args.net_arch is not None:
		try:
			net_arch = ast.literal_eval(args.net_arch)
		except (ValueError, SyntaxError):
			print(f"Error: Invalid format for --net_arch. Please use a list format like '[256, 256]'. Got: {args.net_arch}")
			exit()
		
	print("Args:", args)
	device = torch.device(args.device)

	# Construct folder name based on experiment type
	if args.experiment_name == "net-arch":
		arch_str = '-'.join(map(str, net_arch))
		folder_name = f"results/net-arch/{args.env}/{args.policy}/arch_{arch_str}/{args.seed}"
		run_name = f"{args.policy}_arch_{arch_str}_seed_{args.seed}"
	elif args.experiment_name == "penalty-fn":
		folder_name = f"results/penalty-fn/{args.env}/{args.policy}/penalty_{args.penalty_function}_{args.use_dynamic_lambda_one}/{args.seed}"
		run_name = f"penalty_{args.penalty_function}_{args.use_dynamic_lambda_one}_seed_{args.seed}"
	elif args.experiment_name == "env-config":
		folder_name = f"results/env-config/{args.env}/{args.policy}/c_{args.c}_lambda1_{args.lambda1}_lambda2_{args.lambda2}_rmin_{args.r_min}/g_{env_config['gravity']}_cm_{env_config['cart_mass']}_pm_{env_config['pole_mass']}/{args.seed}"
		run_name = f"g_{env_config['gravity']}_cm_{env_config['cart_mass']}_pm_{env_config['pole_mass']}_c_{args.c}_lambda1_{args.lambda1}_lambda2_{args.lambda2}_rmin_{args.r_min}_seed_{args.seed}"
	else:
		raise ValueError(f"Invalid experiment name: {args.experiment_name}")

 
	if args.use_wandb:
		config=vars(args)
		config["gravity"] = env_config['gravity']
		config["cart_mass"] = env_config['cart_mass']
		config["pole_mass"] = env_config['pole_mass']
		wandb.init(
			project=args.wandb_project,
			entity=args.wandb_entity,
			config=config,
			name=run_name,
			sync_tensorboard=True,
			# monitor_gym=True,
			save_code=True,
		)

	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists(folder_name):
		os.makedirs(folder_name)

	# if args.save_model and not os.path.exists("./models"):
	# 	os.makedirs("./models")

	env = gym.make(args.env)
	eval_env = gym.make(args.env)

	model = env.unwrapped.model
	eval_model = eval_env.unwrapped.model

	# Find the Body IDs
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
	if args.cart_mass is not None:
		print(f"Original cart mass: Train: {model.body_mass[cart_id]} Eval: {eval_model.body_mass[cart_id]}")
		model.body_mass[cart_id] = args.cart_mass
		eval_model.body_mass[cart_id] = args.cart_mass
		print(f"New cart mass: Train: {model.body_mass[cart_id]} Eval: {eval_model.body_mass[cart_id]}")

	if args.pole_mass is not None:
		print(f"Original pole mass: Train: {model.body_mass[pole_id]} Eval: {eval_model.body_mass[pole_id]}")
		model.body_mass[pole_id] = args.pole_mass
		eval_model.body_mass[pole_id] = args.pole_mass
		print(f"New pole mass: Train: {model.body_mass[pole_id]} Eval: {eval_model.body_mass[pole_id]}")

	# Modify gravity if provided
	if args.gravity is not None:
		print(f"Original gravity: Train: {model.opt.gravity.copy()} Eval: {eval_model.opt.gravity.copy()}")
		model.opt.gravity[2] = args.gravity
		eval_model.opt.gravity[2] = args.gravity
		print(f"New gravity: Train: {model.opt.gravity} Eval: {eval_model.opt.gravity}")

	# Set seeds
	# env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)
	
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}
	if args.net_arch is not None:
		kwargs["arch"] = net_arch

	# Initialize policy
	if args.policy == "TD3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		
		policy = TD3.TD3(device=device, **kwargs)


	elif args.policy == "DDPG_RPI": # Our critic in DDPG

		kwargs["c"] = args.c
		kwargs["lambda1"] = args.lambda1
		kwargs["lambda2"] = args.lambda2
		kwargs["lambda1_b01"] = args.lambda1_b01
		kwargs["r_min"] = args.r_min
		kwargs["max_grad_norm"] = args.max_grad_norm
		kwargs["enable_clipping"] = args.enable_clipping
		kwargs["penalty_function"] = args.penalty_function
		kwargs["use_dynamic_lambda_one"] = args.use_dynamic_lambda_one
		policy = DDPG_RPI.DDPG_RPI(device=device,**kwargs)

	elif args.policy == "DDPG":
		policy = DDPG.DDPG(device=device, **kwargs)
  
	# elif args.policy == "TD3_DDPG":
	# 	policy = TD3_DDPG.TD3_DDPG(device=device, **kwargs)
 
	else:
		print("Invalid Policy Name")
		exit()

	if args.load_model != "":
		policy_file = args.load_model
		policy.load(f"{policy_file}")

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)


	# Lists to store the metrics
	episode_durations = []
	true_q_vals = []
	q_hat_vals = []
	time_steps = []
	total_rew_vals = []
	
	# Evaluate untrained policy
	all_tot_disc_rews, all_q_hats, episode_lengths, all_tot_rews = eval_policy(policy, eval_env, args.seed, args.eval_episodes, args.discount, args.policy, device, args.cart_mass, args.pole_mass, args.gravity)
	episode_durations.append(episode_lengths)
	true_q_vals.append(all_tot_disc_rews)
	q_hat_vals.append(all_q_hats)
	time_steps.append(0)
	total_rew_vals.append(all_tot_rews)


	state, _ = env.reset(seed=args.seed)
	done = False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

	for t in range(int(args.max_timesteps)):
		
		episode_timesteps += 1

		# Select action randomly or according to policy
		if t < args.start_timesteps:
			action = env.action_space.sample()
		else:
			action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)

		# Perform action
		next_state, reward, terminated, truncated, _ = env.step(action) 
		done = terminated or truncated
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if t >= args.start_timesteps:
			loss_dict = policy.train(replay_buffer, args.batch_size)

			# Log training losses to WandB
			if args.use_wandb and loss_dict is not None:
				wandb_log_dict = {"global_step": t + 1}

				# Add all metrics with train/ prefix
				for key, value in loss_dict.items():
					wandb_log_dict[f"train/{key}"] = value

				wandb.log(wandb_log_dict)

		if done: 
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
   
			# --- Log training reward to WandB ---
			if args.use_wandb:
				wandb.log({
					"train/episode_reward": episode_reward,
					"global_step": t + 1
				})
			
			# Reset environment
			state, _ = env.reset(seed=episode_num)
			done = False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			all_tot_disc_rews, all_q_hats, episode_lengths, all_tot_rews = eval_policy(policy, eval_env, args.seed, args.eval_episodes, args.discount, args.policy, device, args.cart_mass, args.pole_mass, args.gravity)
			episode_durations.append(episode_lengths)
			true_q_vals.append(all_tot_disc_rews)
			q_hat_vals.append(all_q_hats)
			time_steps.append(t+1)
			total_rew_vals.append(all_tot_rews)

			# --- Log evaluation metrics to WandB ---
			if args.use_wandb:
				wandb.log({
					"eval/avg_total_reward": np.mean(all_tot_rews),
					"eval/avg_discounted_reward": np.mean(all_tot_disc_rews),
					"eval/avg_q_hat": np.mean(all_q_hats),
					"eval/avg_episode_length": np.mean(episode_lengths),
					"global_step": t + 1
				})
			
			np.save(f"{folder_name}/true_q_vals.npy", true_q_vals)
			np.save(f"{folder_name}/q_hat_vals.npy", q_hat_vals)
			np.save(f"{folder_name}/episode_durations.npy", episode_durations)
			np.save(f"{folder_name}/time_steps.npy", time_steps)
			np.save(f"{folder_name}/total_rew_vals.npy", total_rew_vals)

			if args.save_model: policy.save(f"./{folder_name}/model")