import os, sys

package_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(package_dir)

import argparse
import time

# import gym
import gymnasium as gym  # (0828 KSH)
from dm_control import suite  # (1011 KSH)
import torch
import numpy as np
from itertools import count

from gymnasium import Env as gymEnv
from dm_control.rl.control import Environment as dmcEnv

import logging

import os.path as osp
import json

from sac.replay_memory import ReplayMemory
from sac.sac import SAC
from model import EnsembleDynamicsModel
from predict_env import PredictEnv
from sample_env import EnvSampler
from tf_models.constructor import construct_model, format_samples_for_training

# (0828 KSH) Tensorboard logger
from torch.utils.tensorboard import SummaryWriter

import imageio


def readParser():
    parser = argparse.ArgumentParser(description="MBPO")
    parser.add_argument(
        "--env_name",
        default="Hopper-v2",
        help="Mujoco Gym environment (default: Hopper-v2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123456,
        metavar="N",
        help="random seed (default: 123456)",
    )

    parser.add_argument(
        "--use_decay",
        type=bool,
        default=True,
        metavar="G",
        help="discount factor for reward (default: 0.99)",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        metavar="G",
        help="discount factor for reward (default: 0.99)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.005,
        metavar="G",
        help="target smoothing coefficient(τ) (default: 0.005)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.2,
        metavar="G",
        help="Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)",
    )
    parser.add_argument(
        "--policy",
        default="Gaussian",
        help="Policy Type: Gaussian | Deterministic (default: Gaussian)",
    )
    parser.add_argument(
        "--target_update_interval",
        type=int,
        default=1,
        metavar="N",
        help="Value target update per no. of updates per step (default: 1)",
    )
    parser.add_argument(
        "--automatic_entropy_tuning",
        type=bool,
        default=False,
        metavar="G",
        help="Automaically adjust α (default: False)",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=256,
        metavar="N",
        help="hidden size (default: 256)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0003,
        metavar="G",
        help="learning rate (default: 0.0003)",
    )

    parser.add_argument(
        "--num_networks",
        type=int,
        default=7,
        metavar="E",
        help="ensemble size (default: 7)",
    )
    parser.add_argument(
        "--num_elites", type=int, default=5, metavar="E", help="elite size (default: 5)"
    )
    parser.add_argument(
        "--pred_hidden_size",
        type=int,
        default=200,
        metavar="E",
        help="hidden size for predictive model",
    )
    parser.add_argument(
        "--reward_size",
        type=int,
        default=1,
        metavar="E",
        help="environment reward size",
    )

    parser.add_argument(
        "--replay_size",
        type=int,
        default=1000000,
        metavar="N",
        help="size of replay buffer (default: 10000000)",
    )

    parser.add_argument(
        "--model_retain_epochs", type=int, default=1, metavar="A", help="retain epochs"
    )
    parser.add_argument(
        "--model_train_freq",
        type=int,
        default=250,
        metavar="A",
        help="frequency of training",
    )
    parser.add_argument(
        "--rollout_batch_size",
        type=int,
        default=100000,
        metavar="A",
        help="rollout number M",
    )
    parser.add_argument(
        "--epoch_length", type=int, default=1000, metavar="A", help="steps per epoch"
    )
    parser.add_argument(
        "--rollout_min_epoch",
        type=int,
        default=20,
        metavar="A",
        help="rollout min epoch",
    )
    parser.add_argument(
        "--rollout_max_epoch",
        type=int,
        default=150,
        metavar="A",
        help="rollout max epoch",
    )
    parser.add_argument(
        "--rollout_min_length",
        type=int,
        default=1,
        metavar="A",
        help="rollout min length",
    )
    parser.add_argument(
        "--rollout_max_length",
        type=int,
        default=15,
        metavar="A",
        help="rollout max length",
    )
    parser.add_argument(
        "--num_epoch",
        type=int,
        default=1000,
        metavar="A",
        help="total number of epochs",
    )
    parser.add_argument(
        "--min_pool_size", type=int, default=1000, metavar="A", help="minimum pool size"
    )
    parser.add_argument(
        "--real_ratio",
        type=float,
        default=0.05,
        metavar="A",
        help="ratio of env samples / model samples",
    )
    parser.add_argument(
        "--train_every_n_steps",
        type=int,
        default=1,
        metavar="A",
        help="frequency of training policy",
    )
    parser.add_argument(
        "--num_train_repeat",
        type=int,
        default=20,
        metavar="A",
        help="times to training policy per step",
    )
    parser.add_argument(
        "--max_train_repeat_per_step",
        type=int,
        default=5,
        metavar="A",
        help="max training times per step",
    )
    parser.add_argument(
        "--policy_train_batch_size",
        type=int,
        default=256,
        metavar="A",
        help="batch size for training policy",
    )
    parser.add_argument(
        "--init_exploration_steps",
        type=int,
        default=5000,
        metavar="A",
        help="exploration steps initially",
    )
    parser.add_argument(
        "--max_path_length",
        type=int,
        default=1000,
        metavar="A",
        help="max length of path",
    )

    parser.add_argument(
        "--model_type",
        default="tensorflow",
        metavar="A",
        help="predict model -- pytorch or tensorflow",
    )

    parser.add_argument(
        "--cuda", default=True, action="store_true", help="run on CUDA (default: True)"
    )
    return parser.parse_args()


def train(args, env_sampler, predict_env, agent, env_pool, model_pool):
    writer = (
        SummaryWriter(log_dir=args.rl_save_dir)
        if hasattr(args, "rl_save_dir")
        else None
    )  # (0828 KSH)

    total_step = 0
    reward_sum = 0
    rollout_length = 1
    print("Exploring before start...")
    exploration_before_start(args, env_sampler, env_pool, agent)

    for epoch in range(args.num_epoch):
        start_step = total_step
        train_policy_steps = 0
        for i in count():
            cur_step = total_step - start_step

            if cur_step >= args.epoch_length and len(env_pool) > args.min_pool_size:
                break

            if (
                cur_step > 0
                and cur_step % args.model_train_freq == 0
                and args.real_ratio < 1.0
            ):
                train_predict_model(args, env_pool, predict_env, total_step, writer)

                new_rollout_length = set_rollout_length(args, epoch)
                if rollout_length != new_rollout_length:
                    rollout_length = new_rollout_length
                    model_pool = resize_model_pool(args, rollout_length, model_pool)

                print(f"[ Rollout ] length set to {new_rollout_length}")
                rollout_model(
                    args, predict_env, agent, model_pool, env_pool, rollout_length
                )
                print(f"[ Rollout ] finished")

            cur_state, action, next_state, reward, done, info = env_sampler.sample(
                agent, exclude_xy_pos=args.manually_exclude_xy_pos  # (0923 KSH)
            )
            env_pool.push(cur_state, action, reward, next_state, done)

            if len(env_pool) > args.min_pool_size:
                train_policy_steps += train_policy_repeats(
                    args,
                    total_step,
                    train_policy_steps,
                    cur_step,
                    env_pool,
                    model_pool,
                    agent,
                    writer,
                )

            total_step += 1

            if total_step % args.epoch_length == 0:  # Evaluate
                """
                avg_reward_len = min(len(env_sampler.path_rewards), 5)
                avg_reward = sum(env_sampler.path_rewards[-avg_reward_len:]) / avg_reward_len
                logging.info("Step Reward: " + str(total_step) + " " + str(env_sampler.path_rewards[-1]) + " " + str(avg_reward))
                print(total_step, env_sampler.path_rewards[-1], avg_reward)
                """
                # env_sampler.current_state = None
                # sum_reward = 0
                # done = False
                # test_step = 0

                # while (not done) and (test_step != args.max_path_length):
                #     cur_state, action, next_state, reward, done, info = (
                #         env_sampler.sample(agent, eval_t=True)
                #     )
                #     sum_reward += reward
                #     test_step += 1
                # # logger.record_tabular("total_step", total_step)
                # # logger.record_tabular("sum_reward", sum_reward)
                # # logger.dump_tabular()
                # # logging.info("Step Reward: " + str(total_step) + " " + str(sum_reward))

                # 0923 evaluation fix
                env = env_sampler.env
                ep_r_list = []
                for episode in range(args.eval_episode):
                    if isinstance(env, gymEnv):
                        o, _ = env.reset()
                    if isinstance(env, dmcEnv):
                        ts = env.reset()
                        o = np.concat([v for v in ts.observation.values() if v.ndim > 0])

                    if (
                        args.env_name == "Ant-v5" and args.manually_exclude_xy_pos
                    ):  # (0923 KSH - manual exclusion: x, y pos)
                        o = np.delete(o, [0, 1], axis=0)
                    ep_r = 0
                    while True:
                        with torch.no_grad():
                            a = agent.select_action(o, eval=True)
                        if isinstance(env, gymEnv):
                            next_state, reward, term, trunc, info = env.step(a)
                        if isinstance(env, dmcEnv):
                            ts = env.step(a)
                            next_state = np.concat(
                                [v for v in ts.observation.values() if v.ndim > 0]
                            )
                            reward = ts.reward
                            term = ts.last()
                            trunc = False
                            info = {}
                        ep_r += reward
                        o = next_state
                        if (
                            args.env_name == "Ant-v5" and args.manually_exclude_xy_pos
                        ):  # (0923 KSH - manual exclusion: x, y pos)
                            o = np.delete(o, [0, 1], axis=0)
                        done = term or trunc
                        if done:
                            ep_r_list.append(ep_r)
                            break

                avg_return = np.mean(ep_r_list)

                if writer is not None:  # 0828 KSH: tensorboard logging
                    writer.add_scalar("eval/return", avg_return, total_step)
                    # writer.add_scalar("eval/return", sum_reward, total_step)
                    # writer.add_scalar("eval/epoch", epoch, total_step)

                # 0929 store agent checkpoint
                # 1113 store worldmodel
                if (total_step / args.epoch_length) % 5 == 0:
                    model_save_dir = args.rl_save_dir + "/models"
                    os.makedirs(model_save_dir, exist_ok=True)
                    actor_path = model_save_dir + "/actor.pth"
                    critic_path = model_save_dir + "/critic.pth"
                    agent.save_model(
                        env_name=args.env_name,
                        actor_path=actor_path,
                        critic_path=critic_path,
                    )

                    if args.lib == "gym":
                        render_gif_gym(args, agent, total_step)
                    if args.lib == "dmc":
                        render_gif_dmc(args, agent, total_step)

                print(
                    f"==================================================================================[Epoch {epoch+1} / Step {total_step}] return = {avg_return}"
                )

    if writer is not None:  # 0828 KSH
        writer.close()


def render_gif_gym(args, agent: SAC, mainloop_step):
    os.environ["MUJOCO_GL"] = "egl"
    if "MUJOCO_GL" in os.environ:
        print(os.getenv("MUJOCO_GL"))
        print(os.getenv("PYOPENGL_PLATFORM"))

    if args.env_name == "Ant-v5":
        env = gym.make(
            args.env_name,
            exclude_current_positions_from_observation=True,
            include_cfrc_ext_in_observation=args.include_cfrc,
            forward_reward_weight=args.forward_reward_weight,  # default 1.0
            ctrl_cost_weight=args.ctrl_cost_weight,  # default 0.5
            contact_cost_weight=args.contact_cost_weight,  # optional
            healthy_reward=args.healthy_reward,  # optional
            render_mode="rgb_array",
        )
    else:
        env = gym.make(
            args.env_name,
            exclude_current_positions_from_observation=True,
            render_mode="rgb_array",
        )

    frames = []
    seed = 0
    # obs, info = env.reset(seed=seed)
    obs, info = env.reset()

    episode_reward = 0
    episode_length = 0
    done = False

    while not done:
        frame = env.render()
        frames.append(frame)

        if args.manually_exclude_xy_pos:
            obs_exc = obs[2:]
        else:
            obs_exc = obs

        action = agent.select_action(obs_exc, eval=True)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        episode_length += 1

        if done:
            print(f"Episode reward {episode_reward}, length {episode_length}")
            break

    save_dir = args.rl_save_dir + "/rendering/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    gif_name = os.path.join(
        save_dir, f"{args.env_name}_{mainloop_step}step__reward:{episode_reward}.gif"
    )
    imageio.mimsave(gif_name, frames, fps=20)

    env.close()

def render_gif_dmc(args, agent: SAC, mainloop_step):
    # (This should be done before importing dm control -> move to main script)
    # os.environ["MUJOCO_GL"] = "egl"
    # if "MUJOCO_GL" in os.environ:
    #     print(os.getenv("MUJOCO_GL"))
    #     print(os.getenv("PYOPENGL_PLATFORM"))

    seed = 0
    env = suite.load(args.env_name, args.dmc_task, task_kwargs={"random": np.random.RandomState(seed)})

    frames = []
    seed = 0
    time_step = env.reset()

    obs = np.concat([v for v in time_step.observation.values() if v.ndim > 0])

    episode_reward = 0
    episode_length = 0
    done = False

    while not done:
        frame = env.physics.render(camera_id=0, height=480, width=640)
        frames.append(frame)

        if args.manually_exclude_xy_pos:
            obs_exc = obs[2:]
        else:
            obs_exc = obs

        action = agent.select_action(obs_exc, eval=True)

        time_step = env.step(action)

        obs = np.concat([v for v in time_step.observation.values() if v.ndim > 0])
        reward = time_step.reward
        done = time_step.last()
        episode_reward += reward
        episode_length += 1

        if done:
            print(f"Episode reward {episode_reward}, length {episode_length}")
            break

    save_dir = args.rl_save_dir + "/rendering/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    gif_name = os.path.join(
        save_dir, f"{args.env_name}_{args.dmc_task}_{mainloop_step}step__reward:{episode_reward}.gif"
    )
    imageio.mimsave(gif_name, frames, fps=20)


def exploration_before_start(args, env_sampler, env_pool, agent):
    # exclude_xy_pos = True if args.env_name == "Ant-v5" else False  # (0923 KSH)
    exclude_xy_pos = args.manually_exclude_xy_pos

    for i in range(args.init_exploration_steps):
        cur_state, action, next_state, reward, done, info = env_sampler.sample(
            agent, exclude_xy_pos=exclude_xy_pos
        )  # (0923 KSH)
        env_pool.push(cur_state, action, reward, next_state, done)


def set_rollout_length(args, epoch_step):
    rollout_length = min(
        max(
            args.rollout_min_length
            + (epoch_step - args.rollout_min_epoch)
            / (args.rollout_max_epoch - args.rollout_min_epoch)
            * (args.rollout_max_length - args.rollout_min_length),
            args.rollout_min_length,
        ),
        args.rollout_max_length,
    )
    return int(rollout_length)


def train_predict_model(
    args, env_pool, predict_env, mainloop_step, writer: SummaryWriter = None
):  # 0828 KSH: added writer
    # 0905 KSH: added mainloop_step
    # Get all samples from environment
    # state, action, reward, next_state, done = env_pool.sample(len(env_pool))
    state, action, reward, next_state, done = env_pool.sample(args.model_train_sample_size, recent=True)
    delta_state = next_state - state
    inputs = np.concatenate((state, action), axis=-1)
    labels = np.concatenate(
        (np.reshape(reward, (reward.shape[0], -1)), delta_state), axis=-1
    )

    batch_size = (
        args.model_train_batch_size if hasattr(args, "model_train_batch_size") else 256
    )  # 0828
    predict_env.model.train(
        inputs,
        labels,
        mainloop_step=mainloop_step,
        batch_size=batch_size,
        holdout_ratio=0.2,
        writer=writer,
    )  # 0828 KSH: added writer / 0905 KSH: added mainloop_step / 0930: scale data


def resize_model_pool(args, rollout_length, model_pool):
    rollouts_per_epoch = (
        args.rollout_batch_size * args.epoch_length / args.model_train_freq
    )
    model_steps_per_epoch = int(rollout_length * rollouts_per_epoch)
    new_pool_size = args.model_retain_epochs * model_steps_per_epoch

    sample_all = model_pool.return_all()
    new_model_pool = ReplayMemory(new_pool_size)
    new_model_pool.push_batch(sample_all)

    return new_model_pool


def rollout_model(args, predict_env, agent, model_pool, env_pool, rollout_length):
    state, action, reward, next_state, done = env_pool.sample_all_batch(
        args.rollout_batch_size
    )

    # (0828 KSH) For CUDA memory - split samples
    split_size = 250  # 500 hits memory capacity
    for i in range(rollout_length):
        if state.shape[0] == 0:
            break

        next_states_parts = []
        n = state.shape[0]
        for start in range(0, n, split_size):
            end = min(start + split_size, n)

            s_chunk = state[start:end]  # (M, obs_dim)

            if (
                args.env_name == "Ant-v5" and args.manually_exclude_xy_pos
            ):  # (0923 KSH - manual exclusion: x, y pos)
                s_exc_chunk = np.delete(s_chunk, [0, 1], axis=1)
            else:
                s_exc_chunk = s_chunk

            a_chunk = agent.select_action(s_exc_chunk)  # (M, act_dim), returns NumPy
            ns_chunk, r_chunk, term_chunk, info = predict_env.step(s_chunk, a_chunk)
            # ns_chunk: (M, obs_dim), r_chunk: (M, 1) or (M,), term_chunk: (M, 1) or (M,)

            # Push this chunk to the model buffer
            model_pool.push_batch(
                [
                    (s_chunk[j], a_chunk[j], r_chunk[j], ns_chunk[j], term_chunk[j])
                    for j in range(ns_chunk.shape[0])
                ]
            )

            # Collect non-terminal next states to continue rollout
            term_arr = np.asarray(term_chunk)
            if term_arr.ndim > 1:
                nonterm_mask = ~term_arr.squeeze(-1)
            else:
                nonterm_mask = ~term_arr

            if np.any(nonterm_mask):
                next_states_parts.append(ns_chunk[nonterm_mask])

        # If all trajectories terminated, stop early
        if not next_states_parts:
            break

        # Use only non-terminal next states for the next rollout step
        state = np.concatenate(next_states_parts, axis=0)

        # # TODO: Get a batch of actions
        # action = agent.select_action(state)
        # next_states, rewards, terminals, info = predict_env.step(state, action)
        # # TODO: Push a batch of samples
        # model_pool.push_batch(
        #     [
        #         (state[j], action[j], rewards[j], next_states[j], terminals[j])
        #         for j in range(state.shape[0])
        #     ]
        # )
        # nonterm_mask = ~terminals.squeeze(-1)
        # if nonterm_mask.sum() == 0:
        #     break
        # state = next_states[nonterm_mask]


def train_policy_repeats(
    args,
    total_step,
    train_step,
    cur_step,
    env_pool,
    model_pool,
    agent,
    writer=None,  # 0828 KSH
):
    if total_step % args.train_every_n_steps > 0:
        return 0

    if train_step > args.max_train_repeat_per_step * total_step:
        return 0

    for i in range(args.num_train_repeat):
        env_batch_size = int(args.policy_train_batch_size * args.real_ratio)
        model_batch_size = args.policy_train_batch_size - env_batch_size

        env_state, env_action, env_reward, env_next_state, env_done = env_pool.sample(
            int(env_batch_size)
        )

        if args.env_name == "Ant-v5" and args.manually_exclude_xy_pos:  # (0923 KSH)
            env_state = np.delete(env_state, [0, 1], axis=1)
            env_next_state = np.delete(env_next_state, [0, 1], axis=1)

        if model_batch_size > 0 and len(model_pool) > 0:
            model_state, model_action, model_reward, model_next_state, model_done = (
                model_pool.sample_all_batch(int(model_batch_size))
            )

            # 0923 manual exclusion: x, y pos (Ant-v5) / while giving x,y to nominal
            if args.env_name == "Ant-v5" and args.manually_exclude_xy_pos:  # (0923 KSH)
                model_state = np.delete(model_state, [0, 1], axis=1)
                model_next_state = np.delete(model_next_state, [0, 1], axis=1)

            batch_state, batch_action, batch_reward, batch_next_state, batch_done = (
                np.concatenate((env_state, model_state), axis=0),
                np.concatenate((env_action, model_action), axis=0),
                np.concatenate(
                    (np.reshape(env_reward, (env_reward.shape[0], -1)), model_reward),
                    axis=0,
                ),
                np.concatenate((env_next_state, model_next_state), axis=0),
                np.concatenate(
                    (np.reshape(env_done, (env_done.shape[0], -1)), model_done), axis=0
                ),
            )
        else:
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = (
                env_state,
                env_action,
                env_reward,
                env_next_state,
                env_done,
            )

        batch_reward, batch_done = np.squeeze(batch_reward), np.squeeze(batch_done)
        batch_done = (~batch_done).astype(int)
        # (0828 KSH: obtain return values)
        # print("state batch shape:", batch_state.shape)
        # print("action batch shape:", batch_action.shape)
        qf1_loss, qf2_loss, policy_loss, alpha_loss, alpha_tlogs = (
            agent.update_parameters(
                (batch_state, batch_action, batch_reward, batch_next_state, batch_done),
                args.policy_train_batch_size,
                i,
            )
        )
        if (
            cur_step % 100 == 0 and i == args.num_train_repeat - 1
        ):  # (0828 KSH: logging)
            if writer is not None:
                global_step = total_step * args.num_train_repeat + i  # monotonic step

                # writer.add_scalar("sac/qf1_loss", float(qf1_loss), global_step)
                # writer.add_scalar("sac/qf2_loss", float(qf2_loss), global_step)
                writer.add_scalar(
                    "sac/avg_qf_loss", float(qf1_loss + qf2_loss) / 2, global_step
                )
                writer.add_scalar("sac/policy_loss", float(policy_loss), global_step)
                writer.add_scalar("sac/alpha_loss", float(alpha_loss), global_step)
                writer.add_scalar("sac/alpha", float(alpha_tlogs), global_step)

                # (Optional) mix ratio for visibility
                # writer.add_scalar("sac/real_ratio", float(args.real_ratio), global_step)

            print(
                f"[ SAC Agent Update (cur_step {cur_step}) ]\n "
                + f"qf1_loss = {qf1_loss:.4f} | "
                + f"qf2_loss = {qf2_loss:.4f} | "
                + f"policy_loss = {policy_loss:.4f} | "
                + f"alpha_loss = {alpha_loss:.4f} | "
                + f"alpha_tlogs = {alpha_tlogs:.4f}"
            )

    return args.num_train_repeat


# from gym.spaces import Box
from gymnasium.spaces import Box  # (0828 KSH)


class SingleEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super(SingleEnvWrapper, self).__init__(env)
        obs_dim = env.observation_space.shape[0]
        obs_dim += 2
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        torso_height, torso_ang = self.env.sim.data.qpos[
            1:3
        ]  # Need this in the obs for determining when to stop
        obs = np.append(obs, [torso_height, torso_ang])

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        torso_height, torso_ang = self.env.sim.data.qpos[1:3]
        obs = np.append(obs, [torso_height, torso_ang])
        return obs


def main(args=None):
    if args is None:
        args = readParser()

    # Initial environment
    env = gym.make(args.env_name)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # env.seed(args.seed)
    _, _ = env.reset(seed=args.seed)  # (0828 KSH - compatible with latest gymnasium)

    # Intial agent
    agent = SAC(env.observation_space.shape[0], env.action_space, args)

    # Initial ensemble model
    state_size = np.prod(env.observation_space.shape)
    action_size = np.prod(env.action_space.shape)
    if args.model_type == "pytorch":
        env_model = EnsembleDynamicsModel(
            args.num_networks,
            args.num_elites,
            state_size,
            action_size,
            args.reward_size,
            args.pred_hidden_size,
            use_decay=args.use_decay,
        )
    else:
        env_model = construct_model(
            obs_dim=state_size,
            act_dim=action_size,
            hidden_dim=args.pred_hidden_size,
            num_networks=args.num_networks,
            num_elites=args.num_elites,
        )

    # Predict environments
    predict_env = PredictEnv(env_model, args.env_name, args.model_type)

    # Initial pool for env
    env_pool = ReplayMemory(args.replay_size)
    # Initial pool for model
    rollouts_per_epoch = (
        args.rollout_batch_size * args.epoch_length / args.model_train_freq
    )
    model_steps_per_epoch = int(1 * rollouts_per_epoch)
    new_pool_size = args.model_retain_epochs * model_steps_per_epoch
    model_pool = ReplayMemory(new_pool_size)

    # Sampler of environment
    env_sampler = EnvSampler(env, max_path_length=args.max_path_length)

    train(args, env_sampler, predict_env, agent, env_pool, model_pool)


if __name__ == "__main__":
    main()
