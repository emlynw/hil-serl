#!/usr/bin/env python3

import os
import glob
import pickle as pkl
import cv2
import numpy as np

BUFFER_PATH = "/home/emlyn/rl_franka/hil-serl/examples/demo_data_128"  # update accordingly

def main():
    # Collect all .pkl files in BUFFER_PATH
    pkl_files = glob.glob(os.path.join(BUFFER_PATH, "*.pkl"))
    pkl_files.sort()  # optional sort if you want them in order

    success_count = 0

    for pkl_file in pkl_files:
        print(f"Loading: {pkl_file}")
        with open(pkl_file, "rb") as f:
            transitions = pkl.load(f)

        print(f"Total transitions in {pkl_file}: {len(transitions)}")

        for i, transition in enumerate(transitions):
            obs = transition["observations"]
            next_obs = transition["next_observations"]
            reward = transition["rewards"]
            mask = transition["masks"]
            action = transition["actions"]
            info = transition["infos"]
            if "grasp_penalty" in info:
                grasp_penalty = info["grasp_penalty"]
            else:
                grasp_penalty = None

            # We'll try to access the 'state' if it exists
            # (adjust as needed based on your data structure)
            if "state" in obs:
                state_info = obs["state"]
            else:
                state_info = None

            # Extract images for a 2x2 grid
            # top-left  = obs['wrist2']  [0]
            # bottom-left  = obs['wrist1'][0]
            # top-right = next_obs['wrist2']  [0]
            # bottom-right = next_obs['wrist1'][0]
            obs_wrist2 = obs["wrist2"][0]  # shape (256,256,3) in RGB
            obs_wrist1 = obs["wrist1"][0]
            nxt_wrist2 = next_obs["wrist2"][0]
            nxt_wrist1 = next_obs["wrist1"][0]

            # Convert from RGB -> BGR for OpenCV
            obs_wrist2_bgr = cv2.cvtColor(obs_wrist2, cv2.COLOR_RGB2BGR)
            obs_wrist1_bgr = cv2.cvtColor(obs_wrist1, cv2.COLOR_RGB2BGR)
            nxt_wrist2_bgr = cv2.cvtColor(nxt_wrist2, cv2.COLOR_RGB2BGR)
            nxt_wrist1_bgr = cv2.cvtColor(nxt_wrist1, cv2.COLOR_RGB2BGR)

            # Build the 2x2 grid
            # top row: obs_wrist2_bgr (left), nxt_wrist2_bgr (right)
            top_row = cv2.hconcat([obs_wrist2_bgr, nxt_wrist2_bgr])
            # bottom row: obs_wrist1_bgr (left), nxt_wrist1_bgr (right)
            bottom_row = cv2.hconcat([obs_wrist1_bgr, nxt_wrist1_bgr])
            # final 2x2
            grid_image = cv2.vconcat([top_row, bottom_row])

            # Optionally overlay text: reward, partial state, etc.
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_color = (0, 0, 255)  # red
            line_type = cv2.LINE_AA

            if reward !=0:
                success_count+=1
            # Show reward in top-left corner
            cv2.putText(
                grid_image,
                f"Reward: {reward:.2f}",
                (10, 30),
                font, 1.0, text_color, 2, line_type
            )

            # If state_info is available, show the first few elements
            if state_info is not None and len(state_info.shape) > 1:
                # typically shape is (1, some_dim), let's just show the first 3 or so
                state_text = np.array2string(state_info[0], precision=3, suppress_small=True)
                cv2.putText(
                    grid_image,
                    f"State: {state_text}",
                    (10, 70),
                    font, 0.4, text_color, 1, line_type
                )

            action = np.array2string(action, precision=2),
            cv2.putText(
                grid_image,
                f"Action: {action}",
                (10, 110),
                font, 0.4, text_color, 1, line_type
            )

            cv2.putText(
                grid_image,
                f"Mask: {mask}",
                (10, 150),
                font, 0.4, text_color, 1, line_type
            )

            
            cv2.putText(
                grid_image,
                f"Grasp Penalty: {grasp_penalty}",
                (10, 190),
                font, 0.4, text_color, 1, line_type
            )

            # Display in a named window
            cv2.imshow("Buffer Viewer", grid_image)
            # Wait for a key press:
            key = cv2.waitKey(0)
            if key == 27:  # ESC pressed => quit early
                print("User hit ESC. Exiting.")
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
