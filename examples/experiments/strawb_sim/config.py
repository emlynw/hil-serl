import os
import jax
import jax.numpy as jnp
import numpy as np
import gymnasium as gym
from gym.wrappers import TimeLimit

from franka_env.envs.wrappers import (
    MultiCameraBinaryRewardClassifierWrapper,
    Quat2EulerWrapper,
)
from franka_env.envs.franka_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from experiments.strawb_sim.wrappers import ActionState, VideoRecorderReal, ExplorationMemory, CustomPixelObservation, RotateImage, GripperPenaltyWrapper
from experiments.strawb_sim.relative_env import RelativeFrame
from experiments.strawb_real.gamepad_wrapper import GamepadIntervention

from fruit_gym import envs

class TrainConfig(DefaultTrainingConfig):
    image_keys = ["wrist1", "wrist2"]
    classifier_keys = ["wrist1", "wrist2"]
    proprio_keys = ["tcp_pose", "tcp_vel", "gripper_pos"]
    buffer_period = 1000
    checkpoint_period = 5000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"
    video_res = 480
    state_res = 128
    replay_buffer_capacity: int = 80_000

    def get_environment(self, fake_env=False, save_video=False, video_dir='', video_res=video_res, state_res=state_res, classifier=False, xirl=False, obs_horizon=1):
        env = gym.make("PickStrawbEnv", cameras=self.image_keys, width=video_res, height=video_res, randomize_domain=True, reward_type="dense", ee_dof=6, gripper_pause=True)
        env = TimeLimit(env, max_episode_steps=150)
        if not fake_env:
            env = GamepadIntervention(env)
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = RotateImage(env, pixel_key="wrist1")
        if save_video:
            for image_name in self.image_keys:
                crop_res = env.observation_space[image_name].shape[0]
                env = VideoRecorderReal(env, video_dir, camera_name=image_name, crop_resolution=crop_res, resize_resolution=video_res, fps=10, record_every=1)
        for image_name in self.image_keys:
            crop_res = env.observation_space[image_name].shape[0]
            env = CustomPixelObservation(env, pixel_key=image_name, crop_resolution=crop_res, resize_resolution=state_res)
        env = ChunkingWrapper(env, obs_horizon=obs_horizon, act_exec_horizon=None)

        if classifier:
            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("/home/emlyn/rl_franka/hil-serl/examples/classifier_ckpt/"),
            )

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                # added check for z position to further robustify classifier, but should work without as well
                logits = jnp.squeeze(classifier(obs))
                confidence = sigmoid(logits)
                return int(sigmoid(logits) > 0.85), confidence

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        # elif xirl:
        #     # env = RotateImage(env, pixel_key="wrist1")
        #     env = xirlResnet18RewardWrapper(env, image_key="wrist1")
        env = GripperPenaltyWrapper(env, penalty=-0.02)
            
        return env