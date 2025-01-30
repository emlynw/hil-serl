import copy
import os
from tqdm import tqdm
import numpy as np
import pickle as pkl
import datetime
from absl import app, flags
from pynput import keyboard
import cv2
import tkinter as tk

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 100, "Number of successful transitions to collect.")

def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(
        fake_env=False,
        save_video=True,
        video_res=480,
        state_res=256,
        video_dir="./classifier_videos",
        classifier=False
    )

    waitkey = 10
    # Calculate window dimensions and position
    resize_resolution = (480, 480)
    window_width = resize_resolution[0]
    window_height = resize_resolution[1] * 2  # Double height for vertical stack
    
    # Get screen dimensions
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()

    # Calculate centered position
    x_pos = (screen_width - window_width) // 2
    y_pos = (screen_height - window_height) // 2

    # Create and configure window
    cv2.namedWindow("Wrist Views", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Wrist Views", window_width, window_height)
    cv2.moveWindow("Wrist Views", x_pos, y_pos)

    obs, _ = env.reset()
    successes = []
    failures = []

    success_needed = FLAGS.successes_needed
    successful_episodes = 0
    failed_episodes = 0
    successful_episodes_needed = 10
    failed_episodes_needed = 10
    episode_success = False
    pbar1 = tqdm(total=successful_episodes_needed)
    pbar2 = tqdm(total=failed_episodes_needed)

    # Track how many steps we have in the current episode
    episode_step = 0

    while (failed_episodes < failed_episodes_needed):
        # Show camera images
        wrist2 = cv2.cvtColor(obs["wrist2"][0], cv2.COLOR_RGB2BGR)
        wrist2 = cv2.resize(wrist2, resize_resolution)
        wrist1 = cv2.rotate(obs['wrist1'][0], cv2.ROTATE_180)
        wrist1 = cv2.cvtColor(wrist1, cv2.COLOR_RGB2BGR)
        wrist1 = cv2.resize(wrist1, resize_resolution)

        combined = np.vstack((wrist2, wrist1))
        cv2.imshow("Wrist Views", combined)
        cv2.waitKey(waitkey)

        actions = np.zeros(env.action_space.sample().shape) 
        next_obs, rew, done, truncated, info = env.step(actions)

        if "intervene_action" in info:
            actions = info["intervene_action"]

        # Build the transition
        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - done,
                dones=done,
            )
        )

        obs = next_obs

        # Only save transitions if episode_step > 0
        # i.e. skip the first transition after reset
        if episode_step > 0:
            if info['success_key']:
                successes.append(transition)
                episode_success = True
            else:
                failures.append(transition)

        episode_step += 1

        if done or truncated:
            if episode_success:
                successful_episodes += 1
                pbar1.update(1)
            else:
                failed_episodes += 1
                pbar2.update(1)

            print(
                f"Episode complete. Successful episodes: {successful_episodes}, "
                f"Failed episodes: {failed_episodes}"
            )
            print(f"Successful transitions so far: {len(successes)}, "
                  f"Failed transitions: {len(failures)}")
            
            # Save the data
            if not os.path.exists("./classifier_data"):
                os.makedirs("./classifier_data")

            uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_name = f"./classifier_data/{FLAGS.exp_name}_{success_needed}_success_images_{uuid}.pkl"
            with open(file_name, "wb") as f:
                pkl.dump(successes, f)
            print(f"saved {len(successes)} successful transitions to {file_name}")

            file_name = f"./classifier_data/{FLAGS.exp_name}_failure_images_{uuid}.pkl"
            with open(file_name, "wb") as f:
                pkl.dump(failures, f)
            print(f"saved {len(failures)} failure transitions to {file_name}")

            # Reset flags for next episode
            episode_success = False
            episode_step = 0  # reset step counter
            successes = []
            failures = []

            obs, _ = env.reset()

            print("\nPress Any key to start a new episode")
            # Show reset screen
            wrist2 = cv2.cvtColor(obs["wrist2"][0], cv2.COLOR_RGB2BGR)
            wrist2 = cv2.resize(wrist2, (480, 480))
            wrist1 = cv2.rotate(obs['wrist1'][0], cv2.ROTATE_180)
            wrist1 = cv2.cvtColor(wrist1, cv2.COLOR_RGB2BGR)
            wrist1 = cv2.resize(wrist1, (480, 480))
            combined = np.vstack((wrist2, wrist1))
            cv2.imshow("Wrist Views", combined)
            cv2.waitKey(0)

if __name__ == "__main__":
    app.run(main)
