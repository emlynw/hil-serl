#!/usr/bin/env python3

import copy
import os
import datetime
import numpy as np
import pickle as pkl
import cv2
import tkinter as tk
from tqdm import tqdm

from absl import app, flags
from pynput import keyboard

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("episodes_needed", 20, "Number of total episodes to collect.")

def main(_):
    # 1. Make sure config exists
    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    config = CONFIG_MAPPING[FLAGS.exp_name]()

    # 2. Create the environment
    env = config.get_environment(
        fake_env=False,
        save_video=True,
        video_res=480,
        state_res=480,
        video_dir="./classifier_videos",
        classifier=False,
        xirl=False
    )

    # 3. Window/camera display setup
    waitkey = 10
    resize_resolution = (640, 640)
    window_width = resize_resolution[0]
    window_height = resize_resolution[1] * 2  # stack vertically

    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()

    x_pos = (screen_width - window_width) // 2
    y_pos = (screen_height - window_height) // 2

    cv2.namedWindow("Wrist Views", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Wrist Views", window_width, window_height)
    cv2.moveWindow("Wrist Views", x_pos, y_pos)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (0, 0, 255)  # red
    line_type = cv2.LINE_AA

    # 4. Where we save the per-episode .pkl files
    output_dir = "./classifier_data_valid"
    os.makedirs(output_dir, exist_ok=True)

    # 5. Episode counters
    total_episodes_needed = FLAGS.episodes_needed
    current_episode_index = 0
    step_reward = 0

    while current_episode_index < total_episodes_needed:
        # Reset environment
        obs, _ = env.reset()

        print("Press any key to start episode")
        wrist2 = cv2.cvtColor(obs["wrist2"][0], cv2.COLOR_RGB2BGR)
        wrist2 = cv2.resize(wrist2, resize_resolution)
        wrist1 = cv2.cvtColor(obs["wrist1"][0], cv2.COLOR_RGB2BGR)
        wrist1 = cv2.resize(wrist1, resize_resolution)
        combined = np.vstack((wrist2, wrist1))
        cv2.imshow("Wrist Views", combined)
        cv2.waitKey(0)
        done = False
        truncated = False

        # Collect all transitions in this list
        episode_transitions = []

        print(f"\nStarting episode {current_episode_index+1}/{total_episodes_needed}")

        while not done and not truncated:
            # Show camera images
            wrist2 = cv2.cvtColor(obs["wrist2"][0], cv2.COLOR_RGB2BGR)
            wrist2 = cv2.resize(wrist2, resize_resolution)
            wrist1 = cv2.cvtColor(obs["wrist1"][0], cv2.COLOR_RGB2BGR)
            wrist1 = cv2.resize(wrist1, resize_resolution)
            combined = np.vstack((wrist2, wrist1))
            cv2.putText(
                combined,
                f"Reward: {step_reward:.2f}",
                (10, 30),
                font, 1.0, text_color, 2, line_type
            )
            cv2.imshow("Wrist Views", combined)
            cv2.waitKey(waitkey)

            # Either zero action or environment-provided intervention
            actions = np.zeros(env.action_space.sample().shape)
            next_obs, env_reward, done, truncated, info = env.step(actions)

            if "intervene_action" in info:
                actions = info["intervene_action"]

            # Override the environment's reward using success_key
            # If success_key is True => reward = 1.0, else = 0.0
            if info.get("success_key", False):
                step_reward = 1.0
            else:
                step_reward = 0.0

            # Build the transition
            transition = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=step_reward,  # Overridden reward
                masks=1.0 - done,
                dones=done,
            )

            # Save this step
            episode_transitions.append(copy.deepcopy(transition))

            # Move forward
            obs = next_obs

        # End of episode
        current_episode_index += 1
        print(
            f"Episode {current_episode_index} ended with {len(episode_transitions)} transitions."
        )

        # 6. Save this entire episode as a single .pkl file
        uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(output_dir, f"episode_{uuid}.pkl")

        with open(filename, "wb") as f:
            pkl.dump(episode_transitions, f)

        print(f"Saved episode to {filename}")

    print("\nCollection complete!")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    app.run(main)
