#!/usr/bin/env python3

import os
import glob
import pickle as pkl
import cv2
import numpy as np
import tkinter as tk

EPISODE_DIR = "/home/emlyn/rl_franka/hil-serl/examples/experiments/strawb_real/xirl_classifier/demo_buffer"  # where your 'episode_YYYY-MM-DD_HH-MM-SS.pkl' files live
EPISODE_OUT_DIR = EPISODE_DIR

def load_episodes():
    """
    Loads all .pkl files named 'episode_*.pkl' from EPISODE_DIR.
    Each file is expected to be a list of transitions, e.g.:
       [ { "observations":..., "actions":..., "rewards":..., ... }, ... ]

    Returns:
      episodes: a list of dicts, each dict is:
         {
           "filename": str (original pkl path),
           "transitions": [transition, transition, ...]
         }
    """
    os.makedirs(EPISODE_OUT_DIR, exist_ok=True)

    episode_files = glob.glob(os.path.join(EPISODE_DIR, "*.pkl"))

    episodes = []
    for fname in episode_files:
        with open(fname, "rb") as f:
            transitions = pkl.load(f)  # Now just a list
        episodes.append({
            "filename": fname,
            "transitions": transitions
        })
    return episodes


def get_obs_image(obs, key="wrist1"):
    """
    Extract a single image from obs[key].
    If shape is [1, H, W, 3], take obs[key][0].
    Otherwise, if [H, W, 3], just return obs[key].
    """
    img = obs[key]
    if img.ndim == 4:
        img = img[0]
    return img


def show_transition_images(transition, label_str):
    """
    Displays wrist1/wrist2 images in a vertical stack,
    overlaying the label_str (current reward) in the top-left corner.
    """
    next_obs = transition["next_observations"]
    # Adjust if your keys differ. 
    wrist1 = get_obs_image(next_obs, "wrist1")
    wrist2 = get_obs_image(next_obs, "wrist2")

    # Convert from RGB to BGR for OpenCV display
    wrist1_bgr = cv2.cvtColor(wrist1, cv2.COLOR_RGB2BGR)
    wrist2_bgr = cv2.cvtColor(wrist2, cv2.COLOR_RGB2BGR)

    # Resize for display
    h, w = 480, 480
    wrist1_bgr = cv2.resize(wrist1_bgr, (w, h))
    wrist2_bgr = cv2.resize(wrist2_bgr, (w, h))

    combined = np.vstack((wrist1_bgr, wrist2_bgr))

    # Draw the label in the top-left corner
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        combined,
        f"Current Reward: {label_str}",
        (10, 30),  # x,y
        font,
        1.0,
        (0, 0, 255),  # BGR => Red text
        2,
        cv2.LINE_AA
    )
    cv2.imshow("Relabel Viewer", combined)


def main():
    # Create a centered OpenCV window
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()

    window_width = 480
    window_height = 480 * 2
    x_pos = (screen_width - window_width) // 2
    y_pos = (screen_height - window_height) // 2

    cv2.namedWindow("Relabel Viewer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Relabel Viewer", window_width, window_height)
    cv2.moveWindow("Relabel Viewer", x_pos, y_pos)

    # 1) Load episodes
    episodes = load_episodes()
    if not episodes:
        print(f"No files found in {EPISODE_DIR} named 'episode_*.pkl'. Exiting.")
        return

    print(f"Found {len(episodes)} episodes in {EPISODE_DIR}.\n")

    # Build a "global index" of (episode_idx, transition_idx) for transitions that are currently successful (reward==1.0).
    global_indices = []
    for e_idx, ep in enumerate(episodes):
        for t_idx, transition in enumerate(ep["transitions"]):
            if transition.get("rewards", 0.0) == 1.0:
                global_indices.append((e_idx, t_idx))

    if not global_indices:
        print("No successful transitions (reward==1.0) found. Exiting.")
        return

    # We'll iterate over the filtered transitions in a linear pass
    g_ptr = 0
    last_idx = len(global_indices) - 1

    print("Controls:")
    print("  [s] => keep as success (reward=1.0)")
    print("  [f] => mark as failure and delete the transition")
    print("  [n or Enter] => next transition (keep reward as-is)")
    print("  [b] => go back one transition")
    print("  [q] => quit early\n")

    # Interactive loop: only show transitions that are currently successful.
    while 0 <= g_ptr <= last_idx:
        e_idx, t_idx = global_indices[g_ptr]
        transition = episodes[e_idx]["transitions"][t_idx]

        # Show the current reward
        curr_reward = transition.get("rewards", 0.0)
        show_transition_images(transition, f"{curr_reward:.1f}")

        key = cv2.waitKey(0)
        if key == ord('s'):
            # Keep as success => reward remains 1.0
            transition["rewards"] = 1.0
            g_ptr += 1
        elif key == ord('f'):
            # Mark as failure and delete the transition:
            # Remove this transition from the episode.
            episodes[e_idx]["transitions"].pop(t_idx)
            # Remove the corresponding global index.
            global_indices.pop(g_ptr)
            last_idx -= 1
            # Adjust subsequent indices for the same episode.
            for i in range(g_ptr, len(global_indices)):
                if global_indices[i][0] == e_idx and global_indices[i][1] > t_idx:
                    global_indices[i] = (e_idx, global_indices[i][1] - 1)
            if g_ptr > last_idx:
                g_ptr = last_idx
        elif key == ord('n') or key == 13:  # 13 = Enter
            # Next transition, keep reward as is
            g_ptr += 1
        elif key == ord('b'):
            # Go back one transition
            g_ptr -= 1
            if g_ptr < 0:
                g_ptr = 0
        elif key == ord('q'):
            print("Quitting early.")
            break
        else:
            # Unrecognized key => skip to next
            g_ptr += 1

    cv2.destroyAllWindows()

    # 2) Save each episode’s updated list of transitions in EPISODE_OUT_DIR.
    # All transitions are saved (even those not shown in the interactive loop), with only their rewards modified.
    print("\nSaving updated episodes (all transitions, with modified rewards)...")
    for ep in episodes:
        old_file = os.path.basename(ep["filename"])  # e.g. "episode_2025-02-05_14-21-59.pkl"
        new_path = os.path.join(EPISODE_OUT_DIR, old_file)
        with open(new_path, "wb") as f:
            pkl.dump(ep["transitions"], f)
        print(f"Wrote {len(ep['transitions'])} transitions => {new_path}")

    print("\nAll done! Exiting.")


if __name__ == "__main__":
    main()
