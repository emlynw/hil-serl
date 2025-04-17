# ---------------------------------------------------------------------------
#  Imports – unchanged
# ---------------------------------------------------------------------------
import os, time, datetime, copy, pickle as pkl
from tqdm import tqdm
import numpy as np
import cv2, tkinter as tk
from absl import app, flags
import jax, jax.numpy as jnp
from pynput import keyboard
import gymnasium as gym
from gym.wrappers import TimeLimit

from franka_env.envs.wrappers import (
    MultiCameraBinaryRewardClassifierWrapper,
    Quat2EulerWrapper,
)
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from experiments.strawb_sim.wrappers import (
    ActionState, VideoRecorderReal, ExplorationMemory,
    CustomPixelObservation, RotateImage, GripperPenaltyWrapper,
)
from experiments.strawb_sim.relative_env import RelativeFrame
from experiments.strawb_real.gamepad_wrapper import GamepadIntervention

from fruit_gym import envs
from experiments.mappings import CONFIG_MAPPING

# ---------------------------------------------------------------------------
#  Flags
# ---------------------------------------------------------------------------
FLAGS = flags.FLAGS
flags.DEFINE_string ("exp_name",         None, "Name of experiment folder")
flags.DEFINE_integer("successes_needed", 6,   "Successful demos to collect")
flags.DEFINE_list   ("camera_keys", ["robot0_eye_in_hand_image", "agentview_image"],
                     "Comma‑separated camera obs keys to display / record")

reset_key = False
def on_press(key):
    global reset_key
    if str(key) == "Key.enter":
        reset_key = True

# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    cam_keys = FLAGS.camera_keys            # e.g. ["agentview", "wrist_cam"]

    # ------------------------------------------------------------------ #
    # 1) build config and env
    # ------------------------------------------------------------------ #
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    config.image_keys = cam_keys            # overwrite default camera list
    env = config.get_environment(
        fake_env=False,
        save_video=True,
        video_res=480,
        state_res=480,
        video_dir="./demo_videos",
        classifier=True,
    )

    # ------------- window placement (unchanged) ------------------------ #
    waitkey            = 10
    resize_resolution  = (640, 640)
    window_width       = resize_resolution[0]
    window_height      = resize_resolution[1] * 2
    root               = tk.Tk()
    x_pos  = (root.winfo_screenwidth()  - window_width)  // 2
    y_pos  = (root.winfo_screenheight() - window_height) // 2
    root.destroy()

    cv2.namedWindow("Wrist Views", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Wrist Views", window_width, window_height)
    cv2.moveWindow ("Wrist Views", x_pos, y_pos)

    # ------------------------------------------------------------------ #
    # 2) Rollout collection loop (only cam key strings changed)
    # ------------------------------------------------------------------ #
    obs, info = env.reset()
    print("Reset done")

    transitions, trajectory = [], []
    success_count, returns, episode_step = 0, 0, 0
    success_needed          = FLAGS.successes_needed
    episode_success         = False
    pbar = tqdm(total=success_needed)
    reward = 0.0

    while success_count < success_needed:
        # -------- Camera preview ---------------------------------------
        img2 = cv2.cvtColor(obs[cam_keys[1]][0], cv2.COLOR_RGB2BGR)
        img1 = cv2.cvtColor(obs[cam_keys[0]][0], cv2.COLOR_RGB2BGR)
        img2 = cv2.resize(img2, resize_resolution)
        img1 = cv2.resize(img1, resize_resolution)
        combined = np.vstack((img2, img1))
        cv2.putText(
            combined, f"Reward: {reward:.2f}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA,
        )
        cv2.imshow("Wrist Views", combined)
        cv2.waitKey(waitkey)

        # -------- Env step, logging ------------------------------------
        actions = np.zeros(env.action_space.sample().shape)
        if "intervene_action" in info:
            actions = info["intervene_action"]
        next_obs, rew, done, truncated, info = env.step(actions)
        returns += rew
        if "intervene_action" in info:
            actions = info["intervene_action"]
        if rew == 1.0:
            episode_success = True
            done = True

        if episode_step > 0:      # skip very first transition
            trajectory.append(
                dict(
                    observations   = obs,
                    actions        = actions,
                    next_observations = next_obs,
                    rewards        = rew,
                    masks          = 1.0 - done,
                    dones          = done,
                    infos          = info,
                )
            )

        pbar.set_description(f"Return: {returns:.1f}")
        obs          = next_obs
        episode_step += 1

        # -------- Episode end ------------------------------------------
        if done or truncated:
            if episode_success:
                transitions.extend(copy.deepcopy(trajectory))
                success_count += 1
                pbar.update(1)

                if not os.path.exists("./demo_data"):
                    os.makedirs("./demo_data")
                uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"./demo_data/{FLAGS.exp_name}_{success_needed}_demos_{uuid}.pkl"
                with open(filename, "wb") as f:
                    pkl.dump(transitions, f)
                print(f"Saved {len(transitions)} transitions to {filename}")
                done = False

            # hard reset for next episode
            obs, info = env.reset()
            trajectory.clear()
            transitions.clear()
            returns, episode_step = 0, 0
            episode_success       = False

            # show first frame of new episode
            img2 = cv2.cvtColor(obs[cam_keys[1]][0], cv2.COLOR_RGB2BGR)
            img1 = cv2.cvtColor(obs[cam_keys[0]][0], cv2.COLOR_RGB2BGR)
            img2 = cv2.resize(img2, resize_resolution)
            img1 = cv2.resize(img1, resize_resolution)
            combined = np.vstack((img2, img1))
            cv2.putText(
                combined, f"Reward: {reward:.2f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA,
            )
            cv2.imshow("Wrist Views", combined)
            cv2.waitKey(0)


if __name__ == "__main__":
    app.run(main)
