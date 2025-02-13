#!/usr/bin/env python3

import os
import glob
import pickle as pkl

EPISODE_DIR = "./classifier_data_valid"
# Optional: set a different directory if you don't want to overwrite:
# EPISODE_OUT_DIR = "./classifier_data_episodes_trimmed"
EPISODE_OUT_DIR = EPISODE_DIR # Overwrite in place, by default

def main():
    os.makedirs(EPISODE_OUT_DIR, exist_ok=True)

    # Find all files named "episode_*.pkl" in EPISODE_DIR
    episode_files = glob.glob(os.path.join(EPISODE_DIR, "episode_*.pkl"))
    if not episode_files:
        print(f"No files found in {EPISODE_DIR} matching 'episode_*.pkl'. Exiting.")
        return

    print(f"Found {len(episode_files)} episode files in {EPISODE_DIR}.\n")

    for file_path in episode_files:
        with open(file_path, "rb") as f:
            transitions = pkl.load(f)

        if not isinstance(transitions, list):
            print(f"Skipping {file_path}: does not contain a list of transitions.")
            continue
        
        num_original = len(transitions)
        if num_original <= 1:
            print(f"Skipping {file_path}: only {num_original} transition(s), cannot remove the first one meaningfully.")
            continue
        
        # Remove the first transition
        truncated = transitions[1:]
        num_after = len(truncated)

        # Write out the truncated list. Overwrite in place or in a new folder.
        new_basename = os.path.basename(file_path)  # e.g. "episode_2025-02-05_14-21-59.pkl"
        new_path = os.path.join(EPISODE_OUT_DIR, new_basename)
        with open(new_path, "wb") as f:
            pkl.dump(truncated, f)

        print(f"{file_path}: removed 1st transition (from {num_original} down to {num_after}), saved => {new_path}")

    print("\nDone trimming episodes.")

if __name__ == "__main__":
    main()
