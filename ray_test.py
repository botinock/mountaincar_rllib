import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from ray.rllib.policy import Policy

from moviepy.editor import *
import os
from natsort import natsorted


if __name__ == "__main__":
    # policy = Policy.from_checkpoint('./trained_dqn/policies/default_policy')
    # env = gym.make("MountainCar-v0", render_mode="rgb_array")
    # env = RecordVideo(env, video_folder='./demo', video_length=0, name_prefix='trained_agent', episode_trigger=lambda episode_id: True)
    # for i in range(10):
    #     episode_reward = 0
    #     terminated = truncated = False
    #     obs, *_ = env.reset()
    #     while not (terminated or truncated):
    #         action, *_ = policy.compute_single_action(obs)
    #         obs, reward, terminated, truncated, info = env.step(action)
    #         episode_reward += reward
    #     print(episode_reward)

    video_list = []
    for root, dirs, files in os.walk("./demo"):
        files = natsorted(files)
        for file in files:
            if os.path.splitext(file)[1] == '.mp4':
                filePath = os.path.join(root, file)
                video = VideoFileClip(filePath)
                video_list.append(video)

    final_clip = concatenate_videoclips(video_list)
    final_clip.to_videofile("demo/trained_demo.mp4", fps=60, remove_temp=False)
