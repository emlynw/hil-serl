import os
import jax
import jax.numpy as jnp
import numpy as np
import gymnasium as gym
from gym.wrappers import TimeLimit

from franka_env.envs.wrappers import (
    MultiCameraBinaryRewardClassifierWrapper,
)
from franka_env.envs.franka_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from experiments.strawb_sim.wrappers import Quat2EulerWrapper, ActionState, VideoRecorder, ExplorationMemory
from gym_INB0104 import envs

class TrainConfig(DefaultTrainingConfig):
    image_keys = ["wrist1", "wrist2"]
    classifier_keys = ["wrist1", "wrist2"]
    proprio_keys = ["panda/tcp_pos", "panda/tcp_orientation", "panda/gripper_pos", "panda/gripper_vec", "exploration"]
    buffer_period = 1000
    checkpoint_period = 10_000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-fixed-gripper"

    def get_environment(self, fake_env=False, save_video=False, video_dir='', classifier=False, obs_horizon=1):
        env = gym.make("gym_INB0104/ReachIKDeltaStrawbHangingEnv", width=256, height=256, cameras=["wrist1", "wrist2"], randomize_domain=True, ee_dof=4)
        env = TimeLimit(env, max_episode_steps=100)
        if save_video:
            env = VideoRecorder(env, video_dir, crop_resolution=256, resize_resolution=224, fps=10, record_every=2)
        # if not fake_env:
        #     env = SpacemouseIntervention(env)
        env = ExplorationMemory(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ActionState(env)
        env = ChunkingWrapper(env, obs_horizon=obs_horizon, act_exec_horizon=None)
        if classifier:
            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                # added check for z position to further robustify classifier, but should work without as well
                return int(sigmoid(classifier(obs)) > 0.85 and obs['state'][0, 6] > 0.04)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        return env