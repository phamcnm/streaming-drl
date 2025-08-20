import os, pickle, argparse
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

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        sparse_init(m.weight, sparsity=0.9)
        m.bias.data.fill_(0.0)

class StreamQ(nn.Module):
    def __init__(self, n_obs=11, n_actions=3, hidden_size=32, lr=1.0, epsilon_target=0.01, epsilon_start=1.0, exploration_fraction=0.1, final_tau=0.05, initial_tau=1.0, tau_decay=3e-6, total_steps=1_000_000, gamma=0.99, lamda=0.8, kappa_value=2.0):
        super(StreamQ, self).__init__()
        self.n_actions = n_actions
        self.gamma = gamma

        self.epsilon_start = epsilon_start
        self.epsilon_target = epsilon_target
        self.epsilon = epsilon_start
        self.exploration_fraction = exploration_fraction
        self.total_steps = total_steps
        self.final_tau = final_tau
        self.initial_tau = initial_tau
        self.tau_decay = tau_decay
        self.time_step = 0

        self.fc1_v   = nn.Linear(n_obs, hidden_size)
        self.hidden_v  = nn.Linear(hidden_size, hidden_size)
        self.fc_v  = nn.Linear(hidden_size, n_actions)
        self.apply(initialize_weights)
        self.optimizer = Optimizer(list(self.parameters()), lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_value)

    def q(self, x):
        x = self.fc1_v(x)
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        x = self.hidden_v(x)
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        return self.fc_v(x)

    def sample_action(self, s):
        self.time_step += 1
        self.epsilon = linear_schedule(self.epsilon_start, self.epsilon_target, self.exploration_fraction * self.total_steps, self.time_step)
        if isinstance(s, np.ndarray):
            s = torch.tensor(np.array(s), dtype=torch.float)
        if np.random.rand() < self.epsilon:
            q_values = self.q(s)
            greedy_action = torch.argmax(q_values, dim=-1).item()
            random_action = np.random.randint(0, self.n_actions)
            if greedy_action == random_action:
                return random_action, False
            else:
                return random_action, True
        else:
            q_values = self.q(s)
            return torch.argmax(q_values, dim=-1).item(), False
        
    def get_tau(self):
        tau = max(self.final_tau, self.initial_tau * np.exp(-self.tau_decay * self.time_step))
        return tau
        
    def sample_action_softmax(self, s):
        """
        q_values: torch.tensor of shape (n_actions,)
        temperature: float, controls exploration vs exploitation
        """
        self.time_step += 1
        tau = self.get_tau()
        if isinstance(s, np.ndarray):
            s = torch.tensor(np.array(s), dtype=torch.float)
        q_values = self.q(s)
        probs = F.softmax(q_values / tau, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), False

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

def create_env(env_name, corridor_length=9, render=False):
    if env_name == "TMazeClassicToy":
        env = TMazeClassicToy()
    elif env_name == "TMazeClassicPassive":
        env = TMazeClassicPassive()
    elif env_name == "TMazeClassicActive":
        env = TMazeClassicActive()
    elif env_name == "TMazeClassicEasy":
        env = TMazeClassicEasy(corridor_length=corridor_length)
    else:
        env = gym.make(env_name, render_mode='human', max_episode_steps=10_000) if render else gym.make(env_name, max_episode_steps=10_000)
    return env

def main(env_name, corridor_length, seed, lr, gamma, lamda, total_steps, action_selection, epsilon_target, epsilon_start, exploration_fraction, final_tau, initial_tau, tau_decay, kappa_value, hidden_size, debug, overshooting_info, render=False, track=False):
    torch.manual_seed(seed); np.random.seed(seed)
    # env = gym.make(env_name, render_mode='human', max_episode_steps=10_000) if render else gym.make(env_name, max_episode_steps=10_000)
    env = create_env(env_name, corridor_length, render)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = ScaleReward(env, gamma=gamma)
    # env = NormalizeObservation(env)
    env = AddTimeInfo(env)
    agent = StreamQ(n_obs=env.observation_space.shape[0], n_actions=env.action_space.n, hidden_size=hidden_size, lr=lr, gamma=gamma, lamda=lamda, 
                    epsilon_target=epsilon_target, epsilon_start=epsilon_start, exploration_fraction=exploration_fraction, final_tau=final_tau, initial_tau=initial_tau, tau_decay=tau_decay, 
                    total_steps=total_steps, kappa_value=kappa_value)
    num_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    print("Number of parameters:", num_params)
    save_dir_start, save_dir_rate = epsilon_start if action_selection == 'epsilon_greedy' else initial_tau, exploration_fraction if action_selection == 'epsilon_greedy' else tau_decay
    save_dir = "data_stream_q_{}_h_{}_action_{}_start_{}_rate_{}".format(env_name + '-' + str(corridor_length), hidden_size, action_selection, save_dir_start, save_dir_rate)
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
        if action_selection == 'epsilon_greedy':
            a, is_nongreedy = agent.sample_action(s)
        elif action_selection == 'softmax':
            a, is_nongreedy = agent.sample_action_softmax(s)
        s_prime, r, terminated, truncated, info = env.step(a)
        agent.update_params(s, a, r, s_prime, terminated or truncated, is_nongreedy, overshooting_info)
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
            action_selection_status = "%s %f" % ("Epsilon" if action_selection == 'epsilon_greedy' else "Tau", agent.epsilon if action_selection == 'epsilon_greedy' else agent.get_tau())
            print("Cumulative Return: {}, Time Step {}, Episode Number {}, {}, Success {}/{}".format(cumulative_return, t, episode_num, action_selection_status, success, success + failure))
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
    parser.add_argument('--action', type=str, default='softmax', choices=['epsilon_greedy', 'softmax'], help='Action selection strategy')
    parser.add_argument('--epsilon_target', type=float, default=0.01)
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--exploration_fraction', type=float, default=0.5)
    parser.add_argument('--final_tau', type=float, default=0.05)
    parser.add_argument('--initial_tau', type=float, default=0.8)
    parser.add_argument('--tau_decay', type=float, default=3e-6)
    parser.add_argument('--kappa_value', type=float, default=2.0)
    parser.add_argument('--total_steps', type=int, default=4_000_000)
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--overshooting_info', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--track', action='store_true')
    args = parser.parse_args()
    main(args.env_name, args.corridor_length, args.seed, args.lr, args.gamma, args.lamda, args.total_steps, 
         args.action, args.epsilon_target, args.epsilon_start, args.exploration_fraction, args.final_tau, args.initial_tau, args.tau_decay,
         args.kappa_value, args.hidden_size, args.debug, args.overshooting_info, args.render)