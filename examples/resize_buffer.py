#!/usr/bin/env python3
import os
import pickle
import argparse

import cv2
import numpy as np
from tqdm import tqdm

def resize_image(img, new_size=(128, 128)):
    """
    Resizes `img` (if 3D or 4D) to `new_size`.
    
    - If shape is (1, H, W, 3), remove the leading batch dimension, resize, then re-add.
    - If shape is (H, W, 3), just resize.
    - Otherwise, return as is.
    """
    if img.ndim == 4 and img.shape[0] == 1 and img.shape[-1] == 3:
        # shape = (1, H, W, 3)
        h, w = img.shape[1], img.shape[2]
        # Remove leading batch dimension => shape (H, W, 3)
        image_2d = img[0]
        # Resize
        resized_2d = cv2.resize(image_2d, dsize=new_size, interpolation=cv2.INTER_AREA)
        # Re-add leading batch dimension => shape (1, newH, newW, 3)
        return np.expand_dims(resized_2d, axis=0)

    elif img.ndim == 3 and img.shape[-1] == 3:
        # shape = (H, W, 3)
        return cv2.resize(img, dsize=new_size, interpolation=cv2.INTER_AREA)

    # If it's not (H, W, 3) or (1, H, W, 3), just return unchanged
    return img

def process_transitions(transitions, new_size=(128, 128)):
    """
    Iterates over each transition in the list, resizing all images
    found in `observations` and `next_observations`.
    """
    for t in transitions:
        obs = t["observations"]
        next_obs = t["next_observations"]

        # Resize each key in observations
        for k in obs.keys():
            obs[k] = resize_image(obs[k], new_size)

        # Resize each key in next_observations
        for k in next_obs.keys():
            next_obs[k] = resize_image(next_obs[k], new_size)

    return transitions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="./classifier_data_fails", help="Path to input .pkl files.")
    parser.add_argument("--output_dir", default="./classifier_data_fails", help="Where to save resized .pkl files.")
    parser.add_argument("--width", type=int, default=128, help="Target width for resized images.")
    parser.add_argument("--height", type=int, default=128, help="Target height for resized images.")
    args = parser.parse_args()

    # Make sure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Gather all .pkl files from input_dir
    pkl_files = [f for f in os.listdir(args.input_dir) if f.endswith(".pkl")]

    # The new size to which we want to resize (width, height)
    new_size = (args.width, args.height)

    for pkl_file in tqdm(pkl_files, desc="Processing PKLs"):
        in_path = os.path.join(args.input_dir, pkl_file)
        out_path = os.path.join(args.output_dir, pkl_file)

        # Load transitions
        with open(in_path, "rb") as f:
            transitions = pickle.load(f)

        # Resize
        transitions = process_transitions(transitions, new_size=new_size)

        # Save to output_dir with same filename
        with open(out_path, "wb") as f:
            pickle.dump(transitions, f)

    print("Done resizing all data.")

if __name__ == "__main__":
    main()
