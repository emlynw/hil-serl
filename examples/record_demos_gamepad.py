import os
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags
import time
import cv2
import tkinter as tk
from pynput import keyboard

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 20, "Number of successful demos to collect.")

reset_key = False
def on_press(key):
    global reset_key
    try:
        if str(key) == 'Key.enter':
            reset_key = True
    except AttributeError:
        pass

def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(
        fake_env=False,
        save_video=True,
        video_res=480,
        state_res=256,
        video_dir='./demo_videos',
        classifier=True
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
    
    obs, info = env.reset()
    print("Reset done")

    transitions = []
    success_count = 0
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    trajectory = []
    returns = 0
    episode_success = False

    # Track the step count within the current episode
    episode_step = 0

    while success_count < success_needed:
        # ---------------------------------------------------------------------
        # Show camera views
        # ---------------------------------------------------------------------
        wrist2 = cv2.cvtColor(obs["wrist2"][0], cv2.COLOR_RGB2BGR)
        wrist2 = cv2.resize(wrist2, resize_resolution)
        wrist1 = cv2.rotate(obs["wrist1"][0], cv2.ROTATE_180)
        wrist1 = cv2.cvtColor(wrist1, cv2.COLOR_RGB2BGR)
        wrist1 = cv2.resize(wrist1, resize_resolution)

        combined = np.vstack((wrist2, wrist1))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            combined,
            f"Confidence: {info['confidence']:.2f}",
            (10, 30),  # x,y
            font,
            1.0,       # font scale
            (0, 0, 255),  # color (B,G,R) = red text
            2,         # thickness
            cv2.LINE_AA
        )
        cv2.imshow("Wrist Views", combined)
        cv2.waitKey(waitkey)

        # ---------------------------------------------------------------------
        # Environment step
        # ---------------------------------------------------------------------
        actions = np.zeros(env.action_space.sample().shape)
        next_obs, rew, done, truncated, info = env.step(actions)
        returns += rew

        if "intervene_action" in info:
            actions = info["intervene_action"]

        if info.get("succeed", False):
            episode_success = True
            print("Success")

        # Build transition
        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - done,
                dones=done,
                infos=info,
            )
        )

        # Only append to the trajectory if episode_step > 0
        # This ensures we skip the very first transition after reset.
        if episode_step > 0:
            trajectory.append(transition)

        pbar.set_description(f"Return: {returns}")

        obs = next_obs

        episode_step += 1  # increment step count after the first real step

        # ---------------------------------------------------------------------
        # End of episode
        # ---------------------------------------------------------------------
        if done or truncated:
            if episode_success:
                print("\nEpisode complete. Press Any key to start a new episode")
                # Display new reset screen
                wrist2 = cv2.cvtColor(obs["wrist2"][0], cv2.COLOR_RGB2BGR)
                wrist2 = cv2.resize(wrist2, (480, 480))
                wrist1 = cv2.rotate(obs["wrist1"][0], cv2.ROTATE_180)
                wrist1 = cv2.cvtColor(wrist1, cv2.COLOR_RGB2BGR)
                wrist1 = cv2.resize(wrist1, (480, 480))
                combined = np.vstack((wrist2, wrist1))
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    combined,
                    f"Confidence: {info['confidence']:.2f}",
                    (10, 30),  # x,y
                    font,
                    1.0,       # font scale
                    (0, 0, 255),  # color (B,G,R) = red text
                    2,         # thickness
                    cv2.LINE_AA
                )
                cv2.imshow("Wrist Views", combined)
                cv2.waitKey(0)
                # If this episode was successful, copy its entire trajectory
                # to the main transitions buffer
                for trans in trajectory:
                    transitions.append(copy.deepcopy(trans))
                success_count += 1
                pbar.update(1)

                # Save transitions if you want immediate saving upon success
                if not os.path.exists("./demo_data_test"):
                    os.makedirs("./demo_data_test")
                uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                file_name = f"./demo_data_test/{FLAGS.exp_name}_{success_needed}_demos_{uuid}.pkl"
                with open(file_name, "wb") as f:
                    pkl.dump(transitions, f)
                print(f"saved {len(transitions)} transitions to {file_name}")

            # Reset for the next episode
            trajectory = []
            transitions = []
            returns = 0
            episode_success = False
            episode_step = 0  # reset step count
            obs, info = env.reset()

            print("\nEpisode complete. Press Any key to start a new episode")
            # Display new reset screen
            wrist2 = cv2.cvtColor(obs["wrist2"][0], cv2.COLOR_RGB2BGR)
            wrist2 = cv2.resize(wrist2, (480, 480))
            wrist1 = cv2.rotate(obs["wrist1"][0], cv2.ROTATE_180)
            wrist1 = cv2.cvtColor(wrist1, cv2.COLOR_RGB2BGR)
            wrist1 = cv2.resize(wrist1, (480, 480))
            combined = np.vstack((wrist2, wrist1))
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                combined,
                f"Confidence: {info['confidence']:.2f}",
                (10, 30),  # x,y
                font,
                1.0,       # font scale
                (0, 0, 255),  # color (B,G,R) = red text
                2,         # thickness
                cv2.LINE_AA
            )
            cv2.imshow("Wrist Views", combined)
            cv2.waitKey(0)

if __name__ == "__main__":
    app.run(main)
