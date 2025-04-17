#!/usr/bin/env python3
import os
import glob
import pickle as pkl
import cv2
import numpy as np

# Directory containing the episode pickle files.
EPISODE_DIR = "./demo_fails"
# Output directory for the images.
OUTPUT_DIR = "./can_fails"

def load_episodes():
    """
    Loads all .pkl files from EPISODE_DIR.
    Each file is expected to be a list of transitions (dictionaries),
    with each transition containing a 'next_observations' key.
    
    Returns:
      episodes: a list of dicts, each dict is:
         {
           "filename": str (full path),
           "transitions": [transition, transition, ...]
         }
    """
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
    Extract an image from obs[key]. If the image has shape [1, H, W, 3],
    return the first element; otherwise, if already [H, W, 3], return it directly.
    """
    img = obs[key]
    if img.ndim == 4:
        img = img[0]
    return img

def main():
    episodes = load_episodes()
    if not episodes:
        print(f"No episodes found in {EPISODE_DIR}. Exiting.")
        return

    print(f"Found {len(episodes)} episodes.")

    # For each episode, assign an index (e.g., 0, 1, 2, ...)
    for ep_idx, ep in enumerate(episodes):
        print(f"Processing episode {ep_idx} from file {os.path.basename(ep['filename'])}...")
        # For this episode, we will loop over transitions.
        # For each camera encountered, we will create (if necessary) the folder:
        # OUTPUT_DIR/<camera>/<ep_idx>/
        # and then save images sequentially.
        # We'll keep a per-camera counter for numbering the images.
        camera_counters = {}

        for transition in ep["transitions"]:
            next_obs = transition.get("next_observations", {})
            for cam_key, image in next_obs.items():
                if not isinstance(image, np.ndarray):
                    continue
                img = get_obs_image(next_obs, key=cam_key)
                # Ensure that the top-level folder for this camera exists.
                cam_folder = os.path.join(OUTPUT_DIR, cam_key)
                os.makedirs(cam_folder, exist_ok=True)
                # Create an episode folder inside the camera folder.
                ep_folder = os.path.join(cam_folder, str(ep_idx))
                os.makedirs(ep_folder, exist_ok=True)
                # Initialize the counter if not already.
                if cam_key not in camera_counters:
                    camera_counters[cam_key] = 0
                # Increment counter for this camera.
                camera_counters[cam_key] += 1
                img_filename = os.path.join(ep_folder, f"{camera_counters[cam_key]}.png")
                # If needed, convert from RGB to BGR (uncomment if required):
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_filename, img)
        print(f"Episode {ep_idx} processed.")

    print("All episodes processed and images saved.")

if __name__ == "__main__":
    main()
