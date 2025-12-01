from stable_baselines3.common.callbacks import EvalCallback
import os
import torch
import numpy as np
import torch
import wandb # Import wandb

class EvalWithCriticCallback(EvalCallback):
    def __init__(self, eval_env, gamma, eval_freq=500, n_eval_episodes=5, save_path="./logs/", verbose=1, algo="PPO"):
        super().__init__(
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            log_path=save_path,
            verbose=verbose,
        )
        self.mc_tot_returns = []
        self.mc_disc_returns = []
        self.v_s0_estimates = []
        self.time_steps_of_eval = []
        self.episode_durations = []

        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        self.gamma = gamma
        self.algo = algo

    def _on_step(self) -> bool:
        # The parent's on_step will trigger the evaluation if it's time
        result = super()._on_step()

        # Your custom evaluation logic runs at the specified frequency
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            tot_returns = []
            disc_returns = []
            critic_vals = []
            eval_time_steps = []
            episode_durations = []

            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]

                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.model.device)
                with torch.no_grad():
                    if self.algo in ["PPO"]:
                        v_estimate = self.model.policy.predict_values(obs_tensor).item()
                    elif self.algo in ["DQN"]:
                        v_estimate = self.model.q_net(obs_tensor)[0].max().item()
                critic_vals.append(v_estimate)

                done = False
                total_reward = 0.0
                disc_return = 0.0
                t = 0
                while not done:
                    if self.algo in ["PPO"]:
                        action, _ = self.model.predict(obs, deterministic=True)
                    elif self.algo in ["DQN"]:
                        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.model.device)
                        action = self.model.q_net(obs_tensor).max(1).indices.detach().cpu().numpy()

                    obs, reward, done, info = self.eval_env.step(action)
                    if isinstance(obs, tuple):
                        obs = obs[0]
                    total_reward += reward
                    disc_return += (self.gamma ** t) * reward
                    t += 1

                tot_returns.append(total_reward)
                disc_returns.append(disc_return)
                eval_time_steps.append(self.n_calls)
                episode_durations.append(t)

            # === Add logging to WandB ===
            # This logs the mean of your custom metrics at each evaluation step
            if wandb.run is not None:
                wandb.log({
                    "eval/avg_total_return": np.mean(tot_returns),
                    "eval/avg_discounted_return": np.mean(disc_returns),
                    "eval/avg_v_s0_estimate": np.mean(critic_vals),
                    "eval/avg_episode_duration": np.mean(episode_durations)
                })

            self.mc_tot_returns.append(tot_returns)
            self.mc_disc_returns.append(disc_returns)
            self.v_s0_estimates.append(critic_vals)
            self.time_steps_of_eval.append(eval_time_steps)
            self.episode_durations.append(episode_durations)

            np.save(os.path.join(self.save_path, "mc_tot_returns.npy"), np.array(self.mc_tot_returns, dtype=object))
            np.save(os.path.join(self.save_path, "mc_disc_returns.npy"), np.array(self.mc_disc_returns, dtype=object))
            np.save(os.path.join(self.save_path, "v_s0_estimates.npy"), np.array(self.v_s0_estimates, dtype=object))
            np.save(os.path.join(self.save_path, "time_steps_of_eval.npy"), np.array(self.time_steps_of_eval, dtype=object))
            np.save(os.path.join(self.save_path, "episode_durations.npy"), np.array(self.episode_durations, dtype=object))

            if self.verbose > 0:
                print(f"[Eval] Timesteps: {self.n_calls} Avg Disc Return: {np.mean(disc_returns):.2f}, Avg V(s_0): {np.mean(critic_vals):.4f}")

        return result