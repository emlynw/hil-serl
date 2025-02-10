#!/usr/bin/env python3
import os
import argparse
import pickle
import numpy as np

from tqdm import tqdm

def remove_last_seven_state_elements(transitions):
    """
    For each transition, remove the last 1 elements of 'observations["state"]'
    and 'next_observations["state"]' if they are shaped (1, N).
    """
    for t in transitions:
        # Current observation
        if "observations" in t and "state" in t["observations"]:
            state_arr = t["observations"]["state"]  # shape (1, N)
            if state_arr.ndim == 2 and state_arr.shape[0] == 1 and state_arr.shape[1] >= 1:
                squeezed = state_arr[0]              # shape (N,)
                sliced = squeezed[:-1]              # remove the last 1 elements
                t["observations"]["state"] = np.expand_dims(sliced, axis=0)

        # Next observation
        if "next_observations" in t and "state" in t["next_observations"]:
            next_state_arr = t["next_observations"]["state"]  # shape (1, N)
            if (next_state_arr.ndim == 2 and 
                next_state_arr.shape[0] == 1 and 
                next_state_arr.shape[1] >= 1):
                squeezed_next = next_state_arr[0]  # shape (N,)
                sliced_next = squeezed_next[:-1]   # remove the last 1 elements
                t["next_observations"]["state"] = np.expand_dims(sliced_next, axis=0)
    
    return transitions

def main():
    input_dir = '/home/emlyn/rl_franka/hil-serl/examples/classifier_data'
    output_dir = '/home/emlyn/rl_franka/hil-serl/examples/classifier_data_new'
    os.makedirs(output_dir, exist_ok=True)

    pkl_files = [f for f in os.listdir(input_dir) if f.endswith(".pkl")]

    for pkl_file in tqdm(pkl_files, desc="Processing PKLs"):
        in_path = os.path.join(input_dir, pkl_file)
        out_path = os.path.join(output_dir, pkl_file)

        # Load the transitions from the .pkl
        with open(in_path, "rb") as f:
            transitions = pickle.load(f)

        # Remove the last seven elements from 'observations["state"]' / 'next_observations["state"]'
        transitions = remove_last_seven_state_elements(transitions)

        # Save the modified transitions
        with open(out_path, "wb") as f:
            pickle.dump(transitions, f)

    print("Done removing the last 1 elements from obs['state'] with shape (1, N).")

if __name__ == "__main__":
    main()
