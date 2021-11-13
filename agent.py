#
import os
from typing import Callable

import igibson
from igibson.envs.igibson_env import iGibsonEnv
from featureextractor import CustomCombinedExtractor

try:
    import gym
    import torch as th
    import torch.nn as nn
    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.preprocessing import maybe_transpose
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

except ModuleNotFoundError:
    print("stable-baselines3 is not installed. You would need to do: pip install stable-baselines3")
    exit(1)


def main():
    config_file = "turtlebot_point_nav.yaml"
    tensorboard_log_dir = "log_dir"
    num_cpu = 8

    def make_env(rank: int, seed: int = 0) -> Callable:
        def _init() -> iGibsonEnv:
            env = iGibsonEnv(
                config_file=os.path.join(igibson.example_config_path, config_file),
                mode="headless",
                action_timestep=1 / 10.0,
                physics_timestep=1 / 120.0,
            )
            env.seed(seed + rank)
            return env

        set_random_seed(seed)
        return _init

    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    env = VecMonitor(env)

    eval_env = iGibsonEnv(
        config_file=os.path.join(igibson.example_config_path, config_file),
        mode="headless",
        action_timestep=1 / 10.0,
        physics_timestep=1 / 120.0,
    )

    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
    )
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=tensorboard_log_dir, policy_kwargs=policy_kwargs)
    print(model.policy)

    # Random Agent, before training
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Before Training: Mean reward: {mean_reward} +/- {std_reward:.2f}")

    model.learn(1000000)

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
    print(f"After Training: Mean reward: {mean_reward} +/- {std_reward:.2f}")

    model.save("ckpt")
    del model

    model = PPO.load("ckpt")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
    print(f"After Loading: Mean reward: {mean_reward} +/- {std_reward:.2f}")


if __name__ == "__main__":
    main()