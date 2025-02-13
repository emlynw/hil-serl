#!/usr/bin/env python3

import os
import glob
import pickle as pkl
import cv2
import numpy as np
import tkinter as tk

EPISODE_DIR = "./classifier_data_episodes_valid"  # directory containing your 'episode_*.pkl' files
EPISODE_OUT_DIR = "./classifier_data_episodes_valid_shortened"  # where updated episodes will be saved

def load_episodes():
    """
    Loads all .pkl files named 'episode_*.pkl' from EPISODE_DIR.
    Each file is expected to contain a list of transitions (dicts).
    
    Returns:
      episodes: a list of dicts:
         {
           "filename": str (original pkl file path),
           "transitions": [transition, transition, ...]
         }
    """
    os.makedirs(EPISODE_OUT_DIR, exist_ok=True)

    episode_files = glob.glob(os.path.join(EPISODE_DIR, "*.pkl"))
    episodes = []
    for fname in episode_files:
        with open(fname, "rb") as f:
            transitions = pkl.load(f)
        episodes.append({
            "filename": fname,
            "transitions": transitions
        })
    return episodes

def get_obs_image(obs, key="wrist1"):
    """
    Extract a single image from obs[key].
    If the image has shape [1, H, W, 3], returns obs[key][0].
    Otherwise, if [H, W, 3], returns obs[key] directly.
    """
    img = obs[key]
    if img.ndim == 4:
        img = img[0]
    return img

def show_transition_images(transition, label_str):
    """
    Displays the wrist1 and wrist2 images (vertically stacked)
    with the label (e.g. current reward) overlaid in the top-left.
    """
    next_obs = transition["next_observations"]
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

    # Overlay the label text (current reward)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        combined,
        f"Current Reward: {label_str}",
        (10, 30),
        font,
        1.0,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )
    cv2.imshow("Relabel Viewer", combined)

def rebuild_global_indices(episodes):
    """
    Rebuilds a global list of (episode_index, transition_index) for all episodes.
    """
    global_indices = []
    for e_idx, ep in enumerate(episodes):
        for t_idx in range(len(ep["transitions"])):
            global_indices.append((e_idx, t_idx))
    return global_indices

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

    # Load episodes
    episodes = load_episodes()
    if not episodes:
        print(f"No episode files found in {EPISODE_DIR}. Exiting.")
        return
    print(f"Found {len(episodes)} episodes in {EPISODE_DIR}.\n")

    # Build the initial global index list: list of (episode_index, transition_index)
    global_indices = rebuild_global_indices(episodes)
    g_ptr = 0
    last_idx = len(global_indices) - 1

    print("Controls:")
    print("  [s] => set reward=1.0 (SUCCESS) and delete all later transitions in that episode")
    print("  [f] => set reward=0.0 (FAILURE)")
    print("  [n or Enter] => next transition (leave reward unchanged)")
    print("  [b] => go back one transition")
    print("  [q] => quit early\n")

    # Interactive loop over all transitions (using global_indices)
    while 0 <= g_ptr <= last_idx:
        e_idx, t_idx = global_indices[g_ptr]
        transition = episodes[e_idx]["transitions"][t_idx]

        # Show current reward on the display
        curr_reward = transition.get("rewards", 0.0)
        show_transition_images(transition, f"{curr_reward:.1f}")

        key = cv2.waitKey(0)
        if key == ord('s'):
            # Mark current transition as success
            transition["rewards"] = 1.0
            # Delete all transitions after this one in the same episode
            current_ep = episodes[e_idx]["transitions"]
            if t_idx + 1 < len(current_ep):
                episodes[e_idx]["transitions"] = current_ep[:t_idx+1]
                print(f"Truncated episode {e_idx}: kept {t_idx+1} transitions (reward 1.0 set).")
                # Rebuild global indices since the episode changed
                global_indices = rebuild_global_indices(episodes)
                last_idx = len(global_indices) - 1
                # If the current index was the last for this episode, g_ptr will automatically move to next episode
            g_ptr += 1

        elif key == ord('f'):
            # Mark as failure (reward 0.0)
            transition["rewards"] = 0.0
            g_ptr += 1

        elif key == ord('n') or key == 13:  # 13 is Enter
            # Next transition; leave reward as-is.
            g_ptr += 1

        elif key == ord('b'):
            # Go back one transition.
            g_ptr -= 1
            if g_ptr < 0:
                g_ptr = 0

        elif key == ord('q'):
            print("Quitting early.")
            break

        else:
            # For any unrecognized key, just move forward.
            g_ptr += 1

    cv2.destroyAllWindows()

    # Save updated episodes into EPISODE_OUT_DIR
    print("\nSaving updated episodes...")
    for ep in episodes:
        old_file = os.path.basename(ep["filename"])
        new_path = os.path.join(EPISODE_OUT_DIR, old_file)
        with open(new_path, "wb") as f:
            pkl.dump(ep["transitions"], f)
        print(f"Wrote {len(ep['transitions'])} transitions => {new_path}")

    print("\nAll done! Exiting.")

if __name__ == "__main__":
    main()
