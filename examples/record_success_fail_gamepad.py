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

reset_key = False
def on_press(key):
    global reset_key
    try:
        if str(key) == 'Key.enter':
            reset_key = True
    except AttributeError:
        pass

def main(_):
    global reset_key
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    resize_resolution = (480, 480)
    env = config.get_environment(fake_env=False, save_video=True, video_res=480, state_res=224, video_dir="./videos", classifier=False)
    waitkey=10
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
    pbar = tqdm(total=success_needed)
    
    while len(successes) < success_needed:
        wrist2 = cv2.cvtColor(obs["wrist2"][0], cv2.COLOR_RGB2BGR)
        wrist2 = cv2.resize(wrist2, resize_resolution)
        wrist1 = cv2.rotate(obs['wrist1'][0], cv2.ROTATE_180)
        wrist1 = cv2.cvtColor(wrist1, cv2.COLOR_RGB2BGR)
        wrist1 = cv2.resize(wrist1, resize_resolution)
        # Combine images vertically
        combined = np.vstack((wrist2, wrist1))
        cv2.imshow("Wrist Views", combined)
        cv2.waitKey(waitkey)

        actions = np.zeros(env.action_space.sample().shape) 
        next_obs, rew, done, truncated, info = env.step(actions)
        if "intervene_action" in info:
            actions = info["intervene_action"]

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
        if info['success_key']:
            successes.append(transition)
            pbar.update(1)
        else:
            failures.append(transition)

        if done or truncated:
            obs, _ = env.reset()
            print("\nEpisode complete. Press Any key to start a new episode")
            # Show reset screen
            wrist2 = cv2.cvtColor(obs["wrist2"][0], cv2.COLOR_RGB2BGR)
            wrist2 = cv2.resize(wrist2, (480, 480))
            wrist1 = cv2.rotate(obs['wrist1'][0], cv2.ROTATE_180)
            wrist1 = cv2.cvtColor(wrist1, cv2.COLOR_RGB2BGR)
            wrist1 = cv2.resize(wrist1, (480, 480))
            combined = np.vstack((wrist2, wrist1))
            cv2.imshow("Wrist Views", combined)
            cv2.waitKey(0)

    if not os.path.exists("./classifier_data"):
        os.makedirs("./classifier_data")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./classifier_data/{FLAGS.exp_name}_{success_needed}_success_images_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(successes, f)
        print(f"saved {success_needed} successful transitions to {file_name}")

    file_name = f"./classifier_data/{FLAGS.exp_name}_failure_images_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(failures, f)
        print(f"saved {len(failures)} failure transitions to {file_name}")
        
if __name__ == "__main__":
    app.run(main)
