import os, pickle, argparse
import math
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F
from optim import ObGD as Optimizer
from time_wrapper import AddTimeInfo
from normalization_wrappers import NormalizeObservation, ScaleReward
from sparse_init import sparse_init
from tmaze import TMazeClassicToy, TMazeClassicActive, TMazeClassicPassive, TMazeClassicEasy
from torch.utils.tensorboard import SummaryWriter
import wandb

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        sparse_init(m.weight, sparsity=0.9)
        m.bias.data.fill_(0.0)
    if isinstance(m, NoisyLinear):
        sparse_init(m.weight_mu, sparsity=0.9)
        # sparse_init(m.weight_sigma, sparsity=0.1)
        m.bias_mu.data.fill_(0.0)
        # m.bias_sigma.data.fill_(0.0)

class StreamQ(nn.Module):
    def __init__(self, n_obs=11, n_actions=3, hidden_size=64, lr=1.0, gamma=0.99, lamda=0.8, kappa_value=2.0):
        super(StreamQ, self).__init__()
        self.n_actions = n_actions
        self.gamma = gamma
        self.fc1_v   = nn.Linear(n_obs, hidden_size)
        self.hidden_v  = nn.Linear(hidden_size, hidden_size)

        self.value_head = NoisyLinear(hidden_size, 1, std_init=1.0)
        self.advantage_head = NoisyLinear(hidden_size, self.n_actions, std_init=1.0)

        self.apply(initialize_weights)
        self.optimizer = Optimizer(list(self.parameters()), lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_value)

    def reset_noise(self):
        self.value_head.reset_noise()
        self.advantage_head.reset_noise()

    def q(self, x):
        x = self.fc1_v(x)
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        x = self.hidden_v(x)
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)

        value = self.value_head(x)
        advantage = self.advantage_head(x)
        x = value + advantage - advantage.mean(dim=0, keepdim=True)
        return x

    def sample_action(self, s):
        if isinstance(s, np.ndarray):
            s = torch.tensor(np.array(s), dtype=torch.float)
        q_values = self.q(s)
        return torch.argmax(q_values, dim=-1).item(), False

    def update_params(self, s, a, r, s_prime, done, is_nongreedy, overshooting_info=False):
        done_mask = 0 if done else 1
        s, a, r, s_prime, done_mask = torch.tensor(np.array(s), dtype=torch.float), torch.tensor([a], dtype=torch.int).squeeze(0), \
                                         torch.tensor(np.array(r)), torch.tensor(np.array(s_prime), dtype=torch.float), \
                                         torch.tensor(np.array(done_mask), dtype=torch.float)
        q_sa = self.q(s)[a]
        max_q_s_prime_a_prime = torch.max(self.q(s_prime), dim=-1).values
        td_target = r + self.gamma * max_q_s_prime_a_prime * done_mask
        delta = td_target - q_sa

        q_output = -q_sa
        self.optimizer.zero_grad()
        q_output.backward()
        self.optimizer.step(delta.item(), reset=(done or is_nongreedy))

        if overshooting_info:
            max_q_s_prime_a_prime = torch.max(self.q(s_prime), dim=-1).values
            td_target = r + self.gamma * max_q_s_prime_a_prime * done_mask
            delta_bar = td_target - self.q(s)[a]
            if torch.sign(delta_bar * delta).item() == -1:
                print("Overshooting Detected!")

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer("bias_epsilon", torch.FloatTensor(out_features))

        # tracking exploration strength
        self._exploration_sum = 0.0
        self._exploration_count = 0

        # factorized gaussian noise
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        # tracking exploration strength
        with torch.no_grad():
            exploration_strength = (self.weight_sigma.abs().mean() + self.bias_sigma.abs().mean()) / 2
            self._exploration_sum += exploration_strength.item()
            self._exploration_count += 1

        return F.linear(input, weight, bias)
    
    def get_and_reset_exploration(self):
        """Return mean exploration strength since last reset and clear counters."""
        if self._exploration_count == 0:
            return 0.0
        avg_strength = self._exploration_sum / self._exploration_count
        self._exploration_sum = 0.0
        self._exploration_count = 0
        return avg_strength

def create_env(env_name, corridor_length=9, render=False):
    if env_name == "TMazeClassicToy":
        env = TMazeClassicToy(corridor_length=corridor_length)
    elif env_name == "TMazeClassicPassive":
        env = TMazeClassicPassive(corridor_length=corridor_length)
    elif env_name == "TMazeClassicActive":
        env = TMazeClassicActive(corridor_length=corridor_length)
    elif env_name == "TMazeClassicEasy":
        env = TMazeClassicEasy(corridor_length=corridor_length)
    else:
        env = gym.make(env_name, render_mode='human', max_episode_steps=10_000) if render else gym.make(env_name, max_episode_steps=10_000)
    return env

def main(env_name, corridor_length, seed, lr, gamma, lamda, total_steps, kappa_value, hidden_size, debug, overshooting_info, render=False, track=False):
    torch.manual_seed(seed); np.random.seed(seed)
    # env = gym.make(env_name, render_mode='human', max_episode_steps=10_000) if render else gym.make(env_name, max_episode_steps=10_000)
    env = create_env(env_name, corridor_length, render)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = ScaleReward(env, gamma=gamma)
    env = NormalizeObservation(env)
    env = AddTimeInfo(env)
    agent = StreamQ(n_obs=env.observation_space.shape[0], n_actions=env.action_space.n, hidden_size=hidden_size, lr=lr, gamma=gamma, lamda=lamda, kappa_value=kappa_value)
    num_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    print("Number of parameters:", num_params)
    save_dir = "data_stream_q_duelingnoisy_{}_h_{}".format(env_name + '-' + str(corridor_length), hidden_size)
    print("Save Directory:", save_dir)
    if debug:
        print("seed: {}".format(seed), "env: {}".format(env_name))
    if track:
        wandb.init(
            project='stream_q',
            entity='phamcnm',
            sync_tensorboard=True,
            name=save_dir,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{save_dir}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    returns, term_time_steps = [], []
    s, _ = env.reset(seed=seed)
    episode_num = 1
    cumulative_return = 0.0
    success, failure = 0, 0
    for t in range(1, total_steps+1):
        a, is_nongreedy = agent.sample_action(s)
        s_prime, r, terminated, truncated, info = env.step(a)
        agent.update_params(s, a, r, s_prime, terminated or truncated, is_nongreedy, overshooting_info)
        agent.reset_noise()
        s = s_prime
        if terminated or truncated:
            total_return = info['episode']['r']
            if total_return >= 1:
                success += 1
                cumulative_return += 1
            else:
                failure += 1
            if t % 1000 == 0:
                writer.add_scalar("charts/cumulative_return", cumulative_return, t)
            writer.add_scalar("charts/episodic_return", total_return, t)
            if debug and total_return > 0:
                print("Episodic Return: {}, Time Step {}, Episode Number {}, Epsilon {}".format(total_return, t, episode_num, agent.epsilon))
            # returns.append(total_return)
            # term_time_steps.append(t)
            terminated, truncated = False, False
            s, _ = env.reset()
            episode_num += 1
        if t % 10000 == 0:
            returns.append(success)
            term_time_steps.append(t)
            value_exploration_strength = agent.value_head.get_and_reset_exploration()
            advantage_exploration_strength = agent.advantage_head.get_and_reset_exploration()
            print("Cumulative Return: {}, Time Step {}, Episode Number {}, Success {}/{}, Exploration {} | {}".format(cumulative_return, t, episode_num, success, success + failure, value_exploration_strength, advantage_exploration_strength))
            success, failure = 0, 0
    env.close()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, "seed_{}.pkl".format(seed)), "wb") as f:
        pickle.dump((returns, term_time_steps, env_name), f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stream Q(λ)')
    parser.add_argument('--env_name', type=str, default='TMazeClassicEasy')
    parser.add_argument('--corridor_length', type=int, default=49, help='Length of the corridor in the TMaze environment')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lamda', type=float, default=0.8)
    parser.add_argument('--kappa_value', type=float, default=2.0)
    parser.add_argument('--total_steps', type=int, default=2_000_000)
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--overshooting_info', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--track', action='store_true')
    args = parser.parse_args()
    main(args.env_name, args.corridor_length, args.seed, args.lr, args.gamma, args.lamda, args.total_steps, args.kappa_value, args.hidden_size, args.debug, args.overshooting_info, args.render)