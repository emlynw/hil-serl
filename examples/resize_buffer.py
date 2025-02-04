#!/usr/bin/env python3
import os
import pickle
import argparse

import cv2
import numpy as np
from tqdm import tqdm

def resize_if_needed(img, new_size=(128, 128)):
    """
    Resizes an image from (1, 256, 256, 3) to (1, 128, 128, 3) if it detects a 256×256 shape.
    Otherwise, returns the original image array unmodified.
    """
    # We expect a shape of (1, H, W, 3) where H and W may be 256.
    if (img.ndim == 4 and
        img.shape[0] == 1 and
        img.shape[1] == 256 and
        img.shape[2] == 256 and
        img.shape[3] == 3):
        # Remove the leading batch dimension -> shape (256, 256, 3)
        image_2d = img[0]

        # Resize to 128×128
        resized = cv2.resize(
            image_2d, 
            dsize=new_size, 
            interpolation=cv2.INTER_AREA
        )

        # Re-add the leading batch dimension -> shape (1, 128, 128, 3)
        return np.expand_dims(resized, axis=0)
    else:
        # If it doesn't match the expected shape, do nothing
        return img

def process_transitions(transitions):
    """
    Iterates over each transition in the list, resizing all images
    found in `observations` and `next_observations`.
    """
    for t in transitions:
        # Access the current obs and next_obs dicts
        obs = t["observations"]
        next_obs = t["next_observations"]

        # Resize each key if it matches the 256×256 pattern
        for k in obs.keys():
            obs[k] = resize_if_needed(obs[k])

        for k in next_obs.keys():
            next_obs[k] = resize_if_needed(next_obs[k])

    return transitions

def main():
    
    input_dir = '/home/emlyn/rl_franka/hil-serl/examples/classifier_data'
    output_dir = '/home/emlyn/rl_franka/hil-serl/examples/classifier_data_128'
    os.makedirs(output_dir, exist_ok=True)

    # Find all .pkl files in the input directory
    pkl_files = [f for f in os.listdir(input_dir) if f.endswith(".pkl")]

    for pkl_file in tqdm(pkl_files, desc="Processing PKLs"):
        in_path = os.path.join(input_dir, pkl_file)
        out_path = os.path.join(output_dir, pkl_file)

        # Load the transitions list from the .pkl
        with open(in_path, "rb") as f:
            transitions = pickle.load(f)

        # Resize all relevant images
        transitions = process_transitions(transitions)

        # Save the modified transitions to output_dir with the same filename
        with open(out_path, "wb") as f:
            pickle.dump(transitions, f)

    print("Done resizing all data.")

if __name__ == "__main__":
    main()
