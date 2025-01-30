#!/usr/bin/env python3

import glob
import os
import pickle as pkl
import numpy as np

def main():
    # Path where your .pkl files are stored
    DEMO_DIR = "./demo_data"
    # Collect all pkl files
    pkl_files = glob.glob(os.path.join(DEMO_DIR, "*.pkl"))

    total_transitions = 0
    count_exceeding = 0

    for file_path in pkl_files:
        with open(file_path, "rb") as f:
            transitions = pkl.load(f)

        for transition in transitions:
            total_transitions += 1
            actions = transition["actions"]
            # Check if any of the first 6 action dimensions exceed 0.25 in absolute value
            if np.any(np.abs(actions[:6]) > 0.25):
                count_exceeding += 1

    print(f"Scanned {total_transitions} transitions.")
    print(f"Number with |action[:6]| > 0.25 in at least one dimension: {count_exceeding}")


if __name__ == "__main__":
    main()
