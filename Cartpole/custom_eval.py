from stable_baselines3.common.callbacks import EvalCallback
import os
import torch
import numpy as np

class EvalWithCriticCallback(EvalCallback):
    def __init__(self, eval_env, gamma, eval_freq=500, n_eval_episodes=5, save_path="./logs/", verbose=1, algo="PPO"):
        super().__init__(
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            log_path=save_path,
            verbose=verbose,
        )
        self.mc_tot_returns = []    # List of lists: episode returns at each eval
        self.mc_disc_returns = []   # List of lists: episode returns at each eval
        self.v_s0_estimates = []    # List of lists: V(s0) estimates at each eval
        self.time_steps_of_eval = []
        self.episode_durations = []

        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        self.gamma = gamma
        self.algo = algo

    def _on_step(self) -> bool:
        result = super()._on_step()

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            tot_returns = []
            disc_returns = []
            critic_vals = []
            eval_time_steps = []
            episode_durations = []

            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                if isinstance(obs, tuple):  # Gym >=0.26 returns (obs, info)
                    obs = obs[0]

                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.model.device)
                with torch.no_grad():
                    if self.algo in ["PPO"]:
                        # https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/policies.html#ActorCriticPolicy.predict_values    
                        v_estimate = self.model.policy.predict_values(obs_tensor).item()
                    elif self.algo in ["RPI", "DQN", "DoubleDQN"]:
                        v_estimate = self.model.q_net(obs_tensor)[0].max().item()
                
                critic_vals.append(v_estimate)

                done = False
                total_reward = 0.0
                disc_return = 0.0
                t = 0
                while not done:
                    if self.algo in ["PPO"]:
                        action, _ = self.model.predict(obs, deterministic=True)
                    
                    elif self.algo in ["RPI", "DQN", "DoubleDQN"]:
                        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.model.device)
                        action = self.model.q_net(obs_tensor).max(1).indices.detach().cpu().numpy()
                    obs, reward, done, info = self.eval_env.step(action)
                    if isinstance(obs, tuple):  # Gym >=0.26
                        obs = obs[0]
                    total_reward += reward
                    disc_return += (self.gamma **t) * reward
                    t += 1
                
                tot_returns.append(total_reward)
                disc_returns.append(disc_return)
                eval_time_steps.append(self.n_calls)
                episode_durations.append(t)

            self.mc_tot_returns.append(tot_returns)
            self.mc_disc_returns.append(disc_returns)
            self.v_s0_estimates.append(critic_vals)
            self.time_steps_of_eval.append(eval_time_steps)
            self.episode_durations.append(episode_durations)

            # Save to .npy files
            np.save(os.path.join(self.save_path, "mc_tot_returns.npy"), np.array(self.mc_tot_returns))
            np.save(os.path.join(self.save_path, "mc_disc_returns.npy"), np.array(self.mc_disc_returns))
            np.save(os.path.join(self.save_path, "v_s0_estimates.npy"), np.array(self.v_s0_estimates))
            np.save(os.path.join(self.save_path, "time_steps_of_eval.npy"), np.array(self.time_steps_of_eval))
            np.save(os.path.join(self.save_path, "episode_durations.npy"), np.array(self.episode_durations))

            # Log to WandB
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({
                        "eval/mean_total_return": np.mean(tot_returns),
                        "eval/mean_discounted_return": np.mean(disc_returns),
                        "eval/mean_v_s0": np.mean(critic_vals),
                        "eval/mean_episode_duration": np.mean(episode_durations),
                        "eval/global_timesteps": self.n_calls
                    })
            except ImportError:
                pass  # wandb not installed, skip logging

            if self.verbose > 0:
                print(f"[Eval] Timesteps: {self.n_calls} Avg Disc Return: {np.mean(disc_returns):.2f}, Avg V(s_0): {np.mean(critic_vals):.4f}")

        return result