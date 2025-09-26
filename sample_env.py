# import gym
import gymnasium as gym  # (0828 KSH)
import numpy as np

class EnvSampler:
    def __init__(self, env, max_path_length=1000):
        self.env = env

        self.path_length = 0
        self.current_state = None
        self.max_path_length = max_path_length
        self.path_rewards = []
        self.sum_reward = 0

    def sample(self, agent, eval_t=False, exclude_xy_pos=False): # (0923 KSH - added exclude_xy_pos)
        if self.current_state is None:
            # self.current_state = self.env.reset()
            self.current_state, _ = (
                self.env.reset()
            )  # (0828 KSH - gymnasium API)

        cur_state = self.current_state

        if exclude_xy_pos:  # (0923 KSH - manual exclusion, Ant-v5 specific)
            exc_cur_state = np.delete(cur_state, [0, 1])  # exclude x, y pos
        else:
            exc_cur_state = cur_state

        action = agent.select_action(exc_cur_state, eval_t)
        # next_state, reward, terminal, info = self.env.step(action)
        next_state, reward, term, trunc, info = self.env.step(
            action
        )  # (0828 KSH - gymnasium API)
        self.path_length += 1
        self.sum_reward += reward

        terminal = term or trunc  # (0828 KSH - gymnasium API)

        # TODO: Save the path to the env_pool
        if terminal or self.path_length >= self.max_path_length:
            self.current_state = None
            self.path_length = 0
            self.path_rewards.append(self.sum_reward)
            self.sum_reward = 0
        else:
            self.current_state = next_state

        return cur_state, action, next_state, reward, terminal, info
