import os
import jax
import jax.numpy as jnp
import numpy as np
import gymnasium as gym
from gym.wrappers import TimeLimit
from gamepad_wrapper import GamepadIntervention

from franka_env.envs.wrappers import (
    MultiCameraBinaryRewardClassifierWrapper,
)
from franka_env.envs.franka_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from experiments.strawb_real.wrappers import Quat2EulerWrapper, ActionState, VideoRecorderReal, ExplorationMemory
from franka_ros2_gym import envs

class TrainConfig(DefaultTrainingConfig):
    image_keys = ["wrist1", "wrist2"]
    classifier_keys = ["wrist1", "wrist2"]
    proprio_keys = ["panda/tcp_pos", "panda/tcp_orientation", "panda/tcp_vel", "panda/gripper_pos", "panda/gripper_vec", "exploration"]
    buffer_period = 1000
    checkpoint_period = 20_000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"

    def get_environment(self, fake_env=False, save_video=False, video_dir='', classifier=False, obs_horizon=1):
        env = gym.make("franka_ros2_gym/ReachIKDeltaRealStrawbEnv", pos_scale = 0.2, rot_scale=1.0, cameras=self.image_keys, width=128, height=128, randomize_domain=True, ee_dof=6)
        env = TimeLimit(env, max_episode_steps=100)
        if save_video:
            for image_name in self.image_keys:
                env = VideoRecorderReal(env, video_dir, camera_name=image_name, crop_resolution=256, resize_resolution=224, fps=10, record_every=2)
        if not fake_env:
            env = GamepadIntervention(env)
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
                return int(sigmoid(classifier(obs)) > 0.85)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        return env