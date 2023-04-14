import gymnasium as gym

from ray.tune.logger import pretty_print, UnifiedLogger
from ray.tune.registry import register_env
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.ppo import PPOConfig

from reward_wrapper import RewardWrapper

import yaml

EXP_FOLDER_NAME = './logs/dqn_reward_y_1'
# EXP_FOLDER_NAME = './logs/ppo_reward_v_2'



def load_config(filename):
    with open(filename, "r") as stream:
        return yaml.safe_load(stream)
    
def logger_creator(config):
    return UnifiedLogger(config, EXP_FOLDER_NAME, loggers=None)

def env_creator_v(env_config):
    env = gym.make("MountainCar-v0")
    env = RewardWrapper(env=env, kind='v')
    return env

def env_creator_y(env_config):
    env = gym.make("MountainCar-v0")
    env = RewardWrapper(env=env, kind='y')
    return env


if __name__ == "__main__":
    dqn_config = load_config("dqn_config.yaml")
    ppo_config = load_config("ppo_curiosity_config.yaml")
    register_env('MountainCar_reward_v', env_creator_v)
    algo = (
        DQNConfig()
        .training(**dqn_config['config'])
        .exploration(exploration_config=dqn_config['exploration_config'])
        .rollouts(num_rollout_workers=8)
        .resources(num_gpus=1)
        .framework('torch')
        .environment(env='MountainCar_reward_y')
        .rl_module(_enable_rl_module_api=False)
        .build(logger_creator=logger_creator)
    )
    # algo = (
    #     PPOConfig()
    #     .training(**ppo_config['config'])
    #     .exploration(exploration_config=ppo_config['exploration_config'])
    #     .rollouts(num_rollout_workers=8)
    #     .resources(num_gpus=1)
    #     .framework('torch')
    #     .environment(env='MountainCar_reward_v')
    #     # .rl_module(_enable_rl_module_api=False)
    #     .build(logger_creator=logger_creator)
    # )

    for i in range(1000):
        result = algo.train()
        print(pretty_print(result))
        algo.log_result(result)
        if i % 10 == 0:
            checkpoint_dir = algo.save(checkpoint_dir=EXP_FOLDER_NAME)
            print(f"Checkpoint saved in directory {checkpoint_dir}")