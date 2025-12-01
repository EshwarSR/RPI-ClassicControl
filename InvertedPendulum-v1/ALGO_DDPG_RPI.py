import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Implementation of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971
# [Not the implementation used in the TD3 paper]


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, arch=[400, 300]):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, arch[0])
        self.l2 = nn.Linear(arch[0], arch[1])
        self.l3 = nn.Linear(arch[1], action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))



class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, arch=[400, 300]):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim, arch[0])
        self.l2 = nn.Linear(arch[0] + action_dim, arch[1])
        self.l3 = nn.Linear(arch[1], 1)

    def forward(self, state, action):
        q = F.relu(self.l1(state))
        q = F.relu(self.l2(torch.cat([q, action], 1)))
        return self.l3(q)


class DDPG_RPI(object):
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.001,
                 c=1.0,
                 lambda1=10.0,
                 lambda1_b01=100.0,
                 lambda2=1.0,
                 r_min=1.0,
                 device="cpu",
                 arch=[400, 300],
                 max_grad_norm=10.0,
                 enable_clipping=False,
                 penalty_function="relu",
                 use_dynamic_lambda_one=True):

        # self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor = Actor(state_dim, action_dim, max_action, arch).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=1e-4)

        # self.critic = Critic(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim, action_dim, arch).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), weight_decay=1e-2)

        self.discount = discount
        self.tau = tau

        self.c = c
        self.lambda1 = lambda1
        self.lambda1_b01 = lambda1_b01
        self.lambda2 = lambda2
        self.r_min = r_min

        self.max_grad_norm = max_grad_norm
        self.enable_clipping = enable_clipping
        self.penalty_function = penalty_function
        self.use_dynamic_lambda_one = use_dynamic_lambda_one
        self.device = device

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=64):
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(
            batch_size)

        # Compute the target Q value
        target_Q = self.critic_target(
            next_state, self.actor_target(next_state))
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic(state, action)

        # Original Loss:
        # Compute critic loss with conditional lambda based on TD error magnitude
        # critic_loss = F.mse_loss(current_Q, target_Q)

        # RPI modified loss:
        # Calculate TD error
        td_error = current_Q - target_Q

        # Dynamic lambda switching (if enabled)
        if self.use_dynamic_lambda_one:
            # Use lambda1_b01 when |TD error| < 1, use lambda1 when |TD error| >= 1 (component wise)
            lambda_coeffs = torch.where(
                torch.abs(td_error) < 1.0,
                self.lambda1_b01,
                self.lambda1
            )
        else:
            # Always use lambda1
            lambda_coeffs = self.lambda1

        # Compute other terms
        term1 = - self.c * current_Q

        # Compute term2 and term 3 based on penalty function
        if self.penalty_function == "relu":
            term2 = lambda_coeffs * F.relu(td_error)
            term3 = self.lambda2 * F.relu(self.r_min - current_Q)
        elif self.penalty_function == "quadratic":
            term2 = lambda_coeffs * (F.relu(td_error)**2)
            term3 = self.lambda2 * (F.relu(self.r_min - current_Q)**2)
        elif self.penalty_function == "cubic":
            term2 = lambda_coeffs * (F.relu(td_error)**3)
            term3 = self.lambda2 * (F.relu(self.r_min - current_Q)**3)
        else:
            raise ValueError(
                f"Unknown penalty function: {self.penalty_function}")

        critic_loss = torch.mean(term1 + term2 + term3)

        with torch.no_grad():
            # Calculate per-term contribution percentages
            term1_abs_mean = torch.abs(term1).mean().item()
            term2_abs_mean = torch.abs(term2).mean().item()
            term3_abs_mean = torch.abs(term3).mean().item()
            total_abs_contribution = term1_abs_mean + term2_abs_mean + term3_abs_mean

            if total_abs_contribution > 1e-8:  # Avoid division by zero
                term1_percentage = (
                    term1_abs_mean / total_abs_contribution) * 100.0
                term2_percentage = (
                    term2_abs_mean / total_abs_contribution) * 100.0
                term3_percentage = (
                    term3_abs_mean / total_abs_contribution) * 100.0
            else:
                term1_percentage = term2_percentage = term3_percentage = 0.0

            # Store comprehensive metrics for logging (detached to avoid gradient issues)
            loss_components = {
                # Loss terms
                'loss_term1': torch.mean(term1).item(),
                'loss_term2': torch.mean(term2).item(),
                'loss_term3': torch.mean(term3).item(),
                'loss_total': critic_loss.item(),

                # Per-term contribution percentages
                'loss_term1_pct': term1_percentage,
                'loss_term2_pct': term2_percentage,
                'loss_term3_pct': term3_percentage,

                # Q-value statistics
                'q_values_mean': current_Q.mean().item(),

                # Target statistics
                'target_mean': target_Q.mean().item(),

                # TD error statistics
                'td_error_mean': td_error.mean().item(),

                # RPI-specific metrics
                'overestimation_count': (current_Q > target_Q).sum().item(),
                'underestimation_count': (current_Q < target_Q).sum().item(),
                'below_rmin_count': (current_Q < self.r_min).sum().item(),

                # Reward statistics from current batch
                'reward_mean': reward.mean().item(),

                # ReLU activations (how much each penalty term is actually contributing)
                'relu_overestimation_mean': F.relu(current_Q - target_Q).mean().item(),
                'relu_underestimation_mean': F.relu(self.r_min - current_Q).mean().item(),
            }

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        # Clip gradient norm (if enabled)
        if self.enable_clipping:
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.max_grad_norm)

        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        with torch.no_grad():
            loss_components['actor_loss'] = actor_loss.item()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

        return loss_components

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(),
                   filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(),
                   filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(
            torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
