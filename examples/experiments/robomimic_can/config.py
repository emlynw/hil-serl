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
# from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from experiments.robomimic_can.serl_obs_wrapper import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from experiments.robomimic_can.wrappers import VideoRecorderReal, CustomPixelObservation, RotateImage, GripperPenaltyWrapper, DoFConverterGymWrapper
from experiments.robomimic_can.relative_env import RelativeFrame
from experiments.robomimic_can.gamepad_wrapper import GamepadIntervention
from experiments.robomimic_can.robomimic_gym_wrapper import RobomimicGymWrapper


# -----------------------------------------------------------------------------
#  extra imports you need once, at top of the file
# -----------------------------------------------------------------------------
import robomimic.utils.env_utils as EnvUtils
from robomimic.utils.env_utils import create_env, create_env_from_metadata
from robomimic.utils.file_utils import get_env_metadata_from_dataset
import robomimic.utils.obs_utils as ObsUtils
from robomimic.config import config_factory

# -----------------------------------------------------------------------------
#  replace TrainConfig.get_environment with the version below
# -----------------------------------------------------------------------------
class TrainConfig(DefaultTrainingConfig):
    # … (unchanged attributes) …

    def get_environment(
        self,
        fake_env=False,
        save_video=True,
        video_dir="",
        video_res=480,
        state_res=128,
        obs_horizon=1,
        classifier=False,
        xirl=False
    ):
        buffer_period = 1000
        checkpoint_period = 1000
        steps_per_update = 50
        encoder_type = "resnet-pretrained"
        setup_mode = "single-arm-learned-gripper"
        video_res = 480
        state_res = 128
        # ------------------------------------------------------------------
        # 1. Build a robomimic env  (example: PickPlaceCan w/ Panda, images)
        # ------------------------------------------------------------------

        # default BC config
        config = config_factory(algo_name="bc")

        config.observation.modalities.obs.rgb = [
            "agentview_image",
            "robot0_eye_in_hand_image",
        ]

        print(f"low dim: {config.observation.modalities.obs.low_dim}")

        # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
        ObsUtils.initialize_obs_utils_with_config(config)

        rm_env = EnvUtils.create_env(
            env_type=1,
            env_name="PickPlaceCan",
            render=False,
            render_offscreen=True,
            use_image_obs=True,
            robots='Panda',
            camera_names = ['agentview', 'robot0_eye_in_hand'],
        )

        # Optional: reduce DoF to match your policy (here 4 EE DoF + gripper)
        # rm_env = DoFConverterWrapper(rm_env, ee_dof=4)

        # ------------------------------------------------------------------
        # 2.  Export to Gym so all HIL‑SERL wrappers work unchanged
        #     –– we expose only the two image keys + the low‑dim you need
        # ------------------------------------------------------------------
        gym_keys = ["robot0_eye_in_hand_image", "agentview_image",
                    "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
        image_keys = ["robot0_eye_in_hand_image", "agentview_image"]
        self.image_keys = image_keys
        proprio_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
        env = RobomimicGymWrapper(rm_env, keys=gym_keys, flatten_obs=False)



        # ------------------------------------------------------------------
        # 3.  The rest of the wrapper chain is **identical** to the original
        # ------------------------------------------------------------------
        env = TimeLimit(env, max_episode_steps=300)

        env = DoFConverterGymWrapper(env, ee_dof=3)

        if not fake_env:
            env = GamepadIntervention(env, ee_dof=3)

        # Your original Gym wrappers continue to function
        # env = RelativeFrame(env)
        # env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, image_keys=image_keys, proprio_keys=proprio_keys)


        # ------------------------------------------------------------------
        # 4. Video + pixel down‑sampling wrappers
        # ------------------------------------------------------------------
        if save_video:
            for image_name in ["robot0_eye_in_hand_image", "agentview_image"]:
                crop = env.observation_space[image_name].shape[0]
                print(f"crop: {crop}")
                env  = VideoRecorderReal(
                    env,
                    video_dir,
                    camera_name=image_name,
                    crop_resolution=crop,
                    resize_resolution=video_res,
                    fps=10,
                    record_every=1,
                )

        for image_name in ["robot0_eye_in_hand_image", "agentview_image"]:
            crop = env.observation_space[image_name].shape[0]
            env  = CustomPixelObservation(
                env,
                pixel_key=image_name,
                crop_resolution=crop,
                resize_resolution=state_res,
            )

        env = ChunkingWrapper(env, obs_horizon=obs_horizon, act_exec_horizon=None)

        # # ------------------------------------------------------------------
        # # 6. Gripper penalty (unchanged)
        # # ------------------------------------------------------------------
        # env = GripperPenaltyWrapper(env, penalty=-0.02)

        return env
