import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import os

class TMazeBase(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        episode_length: int = 11,
        corridor_length: int = 10,
        oracle_length: int = 0,
        goal_reward: float = 1.0,
        penalty: float = 0.0,
        distract_reward: float = 0.0,
        ambiguous_position: bool = False,
        expose_goal: bool = False,
        add_timestep: bool = False,
    ):
        super().__init__()
        assert corridor_length >= 1 and episode_length >= 1
        assert penalty <= 0.0

        self.episode_length = episode_length
        self.corridor_length = corridor_length
        self.oracle_length = oracle_length

        self.goal_reward = goal_reward
        self.penalty = penalty
        self.distract_reward = distract_reward

        self.ambiguous_position = ambiguous_position
        self.expose_goal = expose_goal
        self.add_timestep = add_timestep

        self.action_space = gym.spaces.Discrete(4)
        self.action_mapping = [[1, 0], [0, 1], [-1, 0], [0, -1]] # right, up, left, down

        self.tmaze_map = np.zeros(
            (3 + 2, self.oracle_length + self.corridor_length + 1 + 2), dtype=bool
        )
        self.bias_x, self.bias_y = 1, 2
        self.tmaze_map[self.bias_y, self.bias_x : -self.bias_x] = True
        self.tmaze_map[[self.bias_y - 1, self.bias_y + 1], -self.bias_x - 1] = True

        obs_dim = 2 if self.ambiguous_position else 3
        if self.expose_goal:
            assert self.ambiguous_position is False
            obs_dim = 4
        if self.add_timestep:
            obs_dim += 1

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

    def position_encoding(self, x: int, y: int, goal_y: int):
        if x == 0:
            if not self.oracle_visited:
                exposure = goal_y
                self.oracle_visited = True
            else:
                exposure = 0

        if self.ambiguous_position:
            if x == 0:
                return [0, exposure]
            elif x < self.oracle_length + self.corridor_length:
                return [0, 0]
            else:
                return [1, y]
        else:
            if self.expose_goal:
                return [x, y, self.corridor_length == x, goal_y if self.oracle_visited else 0]
            else:
                if x == 0:
                    return [x, y, exposure]
                else:
                    return [x, y, 0]

    def timestep_encoding(self):
        return [self.time_step] if self.add_timestep else []

    def get_obs(self):
        return np.array(
            self.position_encoding(self.x, self.y, self.goal_y) + self.timestep_encoding(),
            dtype=np.float32,
        )

    def reward_fn(self, done: bool, x: int, y: int, goal_y: int):
        if done:
            return float(y == goal_y) * self.goal_reward
        else:
            rew = float(x < self.time_step - self.oracle_length) * self.penalty
            if x == 0:
                return rew + self.distract_reward
            else:
                return rew

    def step(self, action):
        self.time_step += 1
        assert self.action_space.contains(action)

        move_x, move_y = self.action_mapping[action]
        if self.tmaze_map[self.bias_y + self.y + move_y, self.bias_x + self.x + move_x]:
            self.x, self.y = self.x + move_x, self.y + move_y

        terminated = self.time_step >= self.episode_length
        truncated = False
        rew = self.reward_fn(terminated, self.x, self.y, self.goal_y)
        return self.get_obs(), rew, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.x, self.y = self.oracle_length, 0
        self.goal_y = np.random.choice([-1, 1])
        self.oracle_visited = False
        self.time_step = 0
        return self.get_obs(), {}

    def visualize(self, trajectories: np.array, idx: str):
        from utils import logger

        batch_size, seq_length, _ = trajectories.shape
        xs = np.arange(seq_length)

        for traj in trajectories:
            plt.plot(xs, traj[:, 0])

        plt.xlabel("Time Step")
        plt.ylabel("Position X")
        plt.savefig(
            os.path.join(logger.get_dir(), "plt", f"{idx}.png"),
            dpi=200,
            bbox_inches="tight",
            pad_inches=0.1,
        )
        plt.close()

class TMazeClassicToy(TMazeBase):
    def __init__(
        self, corridor_length: int = 3, goal_reward: float = 1.0, penalty: float = 0.0, distract_reward: float = 0.0
    ):
        super().__init__(
            episode_length=corridor_length + 1,
            corridor_length=corridor_length,
            goal_reward=goal_reward,
            penalty=penalty,
            distract_reward=distract_reward,
            expose_goal=True,
            ambiguous_position=False,
            add_timestep=False,
        )

class TMazeClassicEasy(TMazeBase):
    def __init__(
        self, corridor_length: int = 49, goal_reward: float = 1.0, penalty: float = 0.0, distract_reward: float = 0.0
    ):
        super().__init__(
            episode_length=corridor_length + 1,
            corridor_length=corridor_length,
            goal_reward=goal_reward,
            penalty=-1.0/corridor_length,
            distract_reward=distract_reward,
            expose_goal=True,
            ambiguous_position=False,
            add_timestep=False,
        )

class TMazeClassicPassive(TMazeBase):
    def __init__(
        self, corridor_length: int = 10, goal_reward: float = 1.0, penalty: float = 0.0, distract_reward: float = 0.0
    ):
        super().__init__(
            episode_length=corridor_length + 1,
            corridor_length=corridor_length,
            goal_reward=goal_reward,
            penalty=penalty,
            distract_reward=distract_reward,
            expose_goal=False,
            ambiguous_position=True,
            add_timestep=False,
        )


class TMazeClassicActive(TMazeBase):
    def __init__(
        self, corridor_length: int = 10, goal_reward: float = 1.0, penalty: float = 0.0, distract_reward: float = 0.0
    ):
        oracle_length = 1
        super().__init__(
            episode_length=corridor_length + 2 * oracle_length + 1,
            corridor_length=corridor_length,
            oracle_length=oracle_length,
            goal_reward=goal_reward,
            penalty=penalty,
            distract_reward=distract_reward,
            expose_goal=False,
            ambiguous_position=True,
            add_timestep=False,
        )


if __name__ == "__main__":
    env = TMazeClassicActive(corridor_length=10, penalty=-0.1)
    obs, _ = env.reset()
    print(env.time_step, "null", obs)
    done = False
    while not done:
        act = env.action_space.sample()
        if env.time_step == 0:
            act = 2
        elif env.time_step == env.episode_length - 1:
            if obs[2] == -1:
                act = 3
            else:
                act = 1
        else:
            act = 0
        obs, rew, terminated, truncated, info = env.step(act)
        done = terminated
        print(env.time_step, env.action_mapping[act], obs, rew, done)