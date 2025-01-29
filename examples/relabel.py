#!/usr/bin/env python3

import os
import glob
import pickle as pkl
import cv2
import numpy as np
from tqdm import tqdm
import tkinter as tk

DATA_DIR = "./classifier_data"

def load_labelled_data():
    """
    Loads all 'success' transitions from *success*.pkl
    and all 'failure' transitions from *failure*.pkl.

    Returns a list of [transition, original_file, current_label].
    current_label is "S" or "F" to start, based on the file name.
    """
    success_files = glob.glob(os.path.join(DATA_DIR, "*success*.pkl"))
    failure_files = glob.glob(os.path.join(DATA_DIR, "*failure*.pkl"))

    labelled_data = []  # list of [transition, orig_file, current_label]

    # Load success transitions
    for sf in success_files:
        with open(sf, "rb") as f:
            data = pkl.load(f)
        for t in data:
            labelled_data.append([t, sf, "S"])  # start labeled success

    # Load failure transitions
    for ff in failure_files:
        with open(ff, "rb") as f:
            data = pkl.load(f)
        for t in data:
            labelled_data.append([t, ff, "F"])  # start labeled failure

    return labelled_data

def show_transition_images(transition, label_str):
    """
    Displays the wrist1/wrist2 images in a vertical stack,
    and draws the current label on the image.
    """
    obs = transition["observations"]

    # Extract images. Commonly: obs["wrist1"] shape = [1, H, W, 3]
    # If so, we index obs["wrist1"][0].
    # If it is just [H, W, 3], we skip indexing.
    def get_image(img):
        return img[0] if img.ndim == 4 else img

    wrist1 = get_image(obs["wrist1"])
    wrist2 = get_image(obs["wrist2"])

    # Convert from RGB to BGR for OpenCV
    wrist1_bgr = cv2.cvtColor(wrist1, cv2.COLOR_RGB2BGR)
    wrist2_bgr = cv2.cvtColor(wrist2, cv2.COLOR_RGB2BGR)

    # Resize for display
    h, w = 480, 480
    wrist1_bgr = cv2.resize(wrist1_bgr, (w, h))
    wrist2_bgr = cv2.resize(wrist2_bgr, (w, h))

    # Stack vertically
    combined = np.vstack((wrist1_bgr, wrist2_bgr))

    # Draw the current label in the top-left corner
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        combined,
        f"Label: {label_str}",
        (10, 30),  # x,y
        font,
        1.0,       # font scale
        (0, 0, 255),  # color (B,G,R) = red text
        2,         # thickness
        cv2.LINE_AA
    )

    cv2.imshow("Relabel Viewer", combined)

def main():
    # Create a centered OpenCV window
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()

    window_width = 480
    window_height = 480 * 2
    x_pos = (screen_width - window_width) // 2
    y_pos = (screen_height - window_height) // 2

    cv2.namedWindow("Relabel Viewer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Relabel Viewer", window_width, window_height)
    cv2.moveWindow("Relabel Viewer", x_pos, y_pos)

    # 1) Load data
    labelled_data = load_labelled_data()
    print(f"Loaded {len(labelled_data)} total transitions from {DATA_DIR}.")

    # 2) Interactive labeling
    print("\nControls:")
    print("  [s] => label current transition as SUCCESS (S)")
    print("  [f] => label current transition as FAILURE (F)")
    print("  [n or Enter] => next (keep current label)")
    print("  [b] => go back one transition")
    print("  [q] => quit early\n")

    idx = 0
    while 0 <= idx < len(labelled_data):
        transition, orig_file, curr_label = labelled_data[idx]
        show_transition_images(transition, curr_label)
        key = cv2.waitKey(0)

        if key == ord('s'):
            labelled_data[idx][2] = "S"  # set label to success
            idx += 1
        elif key == ord('f'):
            labelled_data[idx][2] = "F"  # set label to failure
            idx += 1
        elif key == ord('n') or key == 13:  # 13 = Enter
            # skip/next
            idx += 1
        elif key == ord('b'):
            # Go back one transition
            idx -= 1
            if idx < 0:
                idx = 0  # clamp
        elif key == ord('q'):
            # quit
            print("Quitting early.")
            break
        else:
            # unrecognized key => skip
            idx += 1

    cv2.destroyAllWindows()

    # 3) Build final success/failure lists based on current_label
    final_success = []
    final_failure = []
    for trans, orig_f, curr_label in labelled_data:
        if curr_label == "S":
            final_success.append(trans)
        else:
            final_failure.append(trans)

    # 4) Write to new pickle files (do not overwrite original)
    os.makedirs(DATA_DIR, exist_ok=True)
    success_path = os.path.join(DATA_DIR, "relabelled_success.pkl")
    failure_path = os.path.join(DATA_DIR, "relabelled_failure.pkl")

    with open(success_path, "wb") as f:
        pkl.dump(final_success, f)
    with open(failure_path, "wb") as f:
        pkl.dump(final_failure, f)

    print(f"\nFinal labeled success: {len(final_success)} transitions => {success_path}")
    print(f"Final labeled failure: {len(final_failure)} transitions => {failure_path}")
    print("Done.")

if __name__ == "__main__":
    main()
