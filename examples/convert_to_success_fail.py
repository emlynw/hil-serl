#!/usr/bin/env python3

import os
import glob
import pickle as pkl

def main():
    # Directory containing per-episode PKLs
    input_dir = "./classifier_data_episodes"
    output_dir = "./classifer_data_success_fail"

    # Output paths for the merged success/failure transitions
    success_output = os.path.join(output_dir, "all_successes.pkl")
    fail_output = os.path.join(output_dir, "all_failures.pkl")

    # Gather all .pkl files that look like "episode_*.pkl"
    episode_files = glob.glob(os.path.join(input_dir, "episode_*.pkl"))
    print(f"Found {len(episode_files)} per-episode PKL files in {input_dir}")

    all_successes = []
    all_failures = []

    for file_path in episode_files:
        with open(file_path, "rb") as f:
            data = pkl.load(f)
        transitions = data.get("episode_transitions", [])

        # Check if any transition in this episode has reward == 1.0
        any_reward_one = any(t["rewards"] == 1.0 for t in transitions)

        # If yes, we treat this entire episode as "success"
        if any_reward_one:
            all_successes.extend(transitions)
        else:
            all_failures.extend(transitions)

    # Now dump the combined transitions
    with open(success_output, "wb") as f:
        pkl.dump(all_successes, f)
    with open(fail_output, "wb") as f:
        pkl.dump(all_failures, f)

    print(f"\nWrote {len(all_successes)} transitions to {success_output}")
    print(f"Wrote {len(all_failures)} transitions to {fail_output}")
    print("Done!")

if __name__ == "__main__":
    main()
