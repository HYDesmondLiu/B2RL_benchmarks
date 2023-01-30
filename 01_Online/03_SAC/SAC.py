# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda")

# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear((state_dim + action_dim), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
        # action rescaling
        #self.register_buffer("action_scale", torch.FloatTensor((max_action + max_action) / 2.0))
        #self.register_buffer("action_bias", torch.FloatTensor((max_action - max_action) / 2.0))
        self.action_scale = 1
        self.action_bias = 0

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def select_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

class SAC(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2
        ):
            self.actor = Actor(state_dim, action_dim, max_action).to(device)
            self.qf1 = SoftQNetwork(state_dim, action_dim).to(device)
            self.qf2 = SoftQNetwork(state_dim, action_dim).to(device)
            self.qf1_target = SoftQNetwork(state_dim, action_dim).to(device)
            self.qf2_target = SoftQNetwork(state_dim, action_dim).to(device)
            self.qf1_target.load_state_dict(self.qf1.state_dict())
            self.qf2_target.load_state_dict(self.qf2.state_dict())
            self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=1e-3)
            self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=3e-4)

            # Automatic entropy tuning
            self.target_entropy = -torch.prod(torch.Tensor(action_dim).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=1e-3)

            self.total_it = 0

    def select_action(self, x):
        x = torch.FloatTensor(x.reshape(1, -1)).to(device)
        action, _, _ = self.actor.select_action(x)
        return action.cpu().data.numpy().flatten()

    def train(self, args, replay_buffer, batch_size=256):
            self.total_it += 1
            # Sample replay buffer 
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

            #action, _, _ = self.actor.select_action(torch.Tensor(state).to(device))
            action, _, _ = self.actor.select_action(state)
            #action = action.detach().cpu().numpy()

            with torch.no_grad():
                next_state_action, next_state_log_pi, _ = self.actor.select_action(next_state)
                qf1_next_target = self.qf1_target(next_state, next_state_action)
                qf2_next_target = self.qf2_target(next_state, next_state_action)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                next_q_value = reward.flatten() + (not_done.flatten()) * args.discount * (min_qf_next_target).view(-1)

            qf1_a_values = self.qf1(state, action).view(-1)
            qf2_a_values = self.qf2(state, action).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            self.q_optimizer.zero_grad()
            qf_loss.backward()
            self.q_optimizer.step()

            if self.total_it % args.policy_freq == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_freq
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = self.actor.select_action(state)
                    qf1_pi = self.qf1(state, pi)
                    qf2_pi = self.qf2(state, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                    actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    with torch.no_grad():
                        _, log_pi, _ = self.actor.select_action(state)
                    alpha_loss = (-self.log_alpha * (log_pi + self.target_entropy)).mean()

                    self.a_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.a_optimizer.step()
                    self.alpha = self.log_alpha.exp().item()

            # update the target networks
            if self.total_it % args.target_network_frequency == 0:
                for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)