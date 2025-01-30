#!/usr/bin/env python3

import os
import glob
import pickle as pkl
import numpy as np

SOURCE_DIR = "./demo_data"       # <-- Where your original demos reside
OUTPUT_DIR = "./demo_data_rescaled"   # <-- Where to save the rescaled files
BOUNDS = 0.25                         # <-- Max absolute value allowed for first 6 actions
SCALE_FACTOR = 4.0                    # <-- Multiply the first 6 actions by 4

def main():
    # Create output directory if needed
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all pkl files in the source directory
    pkl_files = glob.glob(os.path.join(SOURCE_DIR, "*.pkl"))
    print(f"Found {len(pkl_files)} .pkl file(s) in {SOURCE_DIR}.")

    for file_path in pkl_files:
        with open(file_path, "rb") as f:
            transitions = pkl.load(f)

        cleaned_rescaled = []
        removed_count = 0

        for t in transitions:
            # Check if the first 6 dimensions of action are within ±0.25
            action = t["actions"]
            if np.any(np.abs(action[:6]) > BOUNDS):
                # Skip this transition
                removed_count += 1
                continue

            # Otherwise rescale the first 6 dimensions by *4
            action[:6] = action[:6] * SCALE_FACTOR

            # Append back to the new transitions list
            cleaned_rescaled.append(t)

        # Save to a new file in OUTPUT_DIR with similar name
        base_name = os.path.basename(file_path)  # e.g. 'my_demos.pkl'
        new_file = os.path.join(OUTPUT_DIR, base_name.replace(".pkl", "_rescaled.pkl"))
        with open(new_file, "wb") as f:
            pkl.dump(cleaned_rescaled, f)

        print(
            f"Processed: {file_path}\n"
            f"  Original transitions: {len(transitions)}\n"
            f"  Removed transitions:  {removed_count}\n"
            f"  Final transitions:    {len(cleaned_rescaled)}\n"
            f"  Saved to: {new_file}\n"
        )

if __name__ == "__main__":
    main()
