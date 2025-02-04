#!/usr/bin/env python3
import os
import argparse
import pickle
import numpy as np
import cv2
from tqdm import tqdm

def rotate_wrist1_images(transitions):
    """
    For each transition, rotate the 'wrist1' image by 180 degrees if it has shape (1, H, W, 3).
    """
    for t in transitions:
        # Current observation
        if "observations" in t and "wrist1" in t["observations"]:
            wrist1_img = t["observations"]["wrist1"]
            # Check if shape is (1, H, W, 3)
            if (wrist1_img.ndim == 4 and
                wrist1_img.shape[0] == 1 and
                wrist1_img.shape[3] == 3):
                # Remove the leading batch dimension -> shape (H, W, 3)
                squeezed = wrist1_img[0]
                # Rotate by 180 degrees
                rotated = cv2.rotate(squeezed, cv2.ROTATE_180)
                # Re-add the batch dimension -> shape (1, H, W, 3)
                t["observations"]["wrist1"] = np.expand_dims(rotated, axis=0)

        # Next observation
        if "next_observations" in t and "wrist1" in t["next_observations"]:
            next_wrist1_img = t["next_observations"]["wrist1"]
            if (next_wrist1_img.ndim == 4 and
                next_wrist1_img.shape[0] == 1 and
                next_wrist1_img.shape[3] == 3):
                squeezed_next = next_wrist1_img[0]
                rotated_next = cv2.rotate(squeezed_next, cv2.ROTATE_180)
                t["next_observations"]["wrist1"] = np.expand_dims(rotated_next, axis=0)
    return transitions

def main():
    input_dir = '/home/emlyn/rl_franka/hil-serl/examples/classifier_data_128'
    output_dir = '/home/emlyn/rl_franka/hil-serl/examples/classifier_data_new'
    os.makedirs(output_dir, exist_ok=True)

    # Collect all .pkl files
    pkl_files = [f for f in os.listdir(input_dir) if f.endswith(".pkl")]

    for pkl_file in tqdm(pkl_files, desc="Processing PKLs"):
        in_path = os.path.join(input_dir, pkl_file)
        out_path = os.path.join(output_dir, pkl_file)

        # Load transitions from the pkl file
        with open(in_path, "rb") as f:
            transitions = pickle.load(f)

        # Rotate all wrist1 images
        transitions = rotate_wrist1_images(transitions)

        # Save updated transitions to output_dir
        with open(out_path, "wb") as f:
            pickle.dump(transitions, f)

    print("Done rotating 'wrist1' images by 180 degrees.")

if __name__ == "__main__":
    main()
