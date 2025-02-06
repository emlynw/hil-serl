#!/usr/bin/env python3

import os
import glob
import pickle as pkl

def main():
    # Directory containing per-episode PKLs
    input_dir = "./classifier_data_episodes_fail"
    output_dir = "./classifier_data_fails"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Output paths for the success/failure transitions
    success_output = os.path.join(output_dir, "all_successes.pkl")
    fail_output = os.path.join(output_dir, "all_failures.pkl")

    # Gather all .pkl files (e.g. "episode_YYYY-MM-DD_HH-MM-SS.pkl")
    episode_files = glob.glob(os.path.join(input_dir, "*.pkl"))
    print(f"Found {len(episode_files)} per-episode PKL files in {input_dir}")

    all_successes = []
    all_failures = []

    for file_path in episode_files:
        with open(file_path, "rb") as f:
            data = pkl.load(f)

        # If your files are a list of transitions directly:
        #   transitions = data
        # If your files are a dict with "episode_transitions" as a list:
        #   transitions = data.get("episode_transitions", [])
        # Adjust as appropriate. Example below assumes the latter:
        transitions = data
        if not isinstance(transitions, list):
            print(f"Warning: {file_path} doesn't contain a list of transitions.")
            continue

        # Classify each transition individually
        for t in transitions:
            # If you have exactly reward=1.0 => success
            # (You could also do > 0 if you have other positive rewards, etc.)
            if t["rewards"] == 1.0:
                all_successes.append(t)
            else:
                all_failures.append(t)

    # Now dump all the success/failure transitions
    with open(success_output, "wb") as f:
        pkl.dump(all_successes, f)

    with open(fail_output, "wb") as f:
        pkl.dump(all_failures, f)

    print(f"\nWrote {len(all_successes)} transitions to {success_output}")
    print(f"Wrote {len(all_failures)} transitions to {fail_output}")
    print("Done!")

if __name__ == "__main__":
    main()
