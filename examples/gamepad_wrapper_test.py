import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import cv2
from franka_ros2_gym import envs
import numpy as np
from experiments.strawb_real.gamepad_wrapper import GamepadIntervention
from experiments.strawb_real.resnet_wrapper import ResNet10Wrapper
from experiments.strawb_real.wrappers import Quat2EulerWrapper, ActionState, VideoRecorderReal, ExplorationMemory
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from experiments.mappings import CONFIG_MAPPING
import tkinter as tk
import time

def main():
    exp_name = "strawb_real"
    config = CONFIG_MAPPING[exp_name]()

    env = config.get_environment(fake_env=False, save_video=False, video_res=480, state_res=256, video_dir="./videos", classifier=False)
    env = ResNet10Wrapper(env)
     
    waitkey = 10
    # Calculate window dimensions and position
    resize_resolution = (480, 480)
    window_width = resize_resolution[0]
    window_height = resize_resolution[1] * 2  # Double height for vertical stack
    
    # Get screen dimensions
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()

    # Calculate centered position
    x_pos = (screen_width - window_width) // 2
    y_pos = (screen_height - window_height) // 2

    # Create and configure window
    cv2.namedWindow("Wrist Views", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Wrist Views", window_width, window_height)
    cv2.moveWindow("Wrist Views", x_pos, y_pos)

    while True:
        # reset the environment
        i=0
        terminated = False
        truncated = False
        obs, info = env.reset()
        rotate = True
        
        while not terminated and not truncated:
            wrist2 = cv2.cvtColor(obs["wrist2"][0], cv2.COLOR_RGB2BGR)
            wrist2 = cv2.resize(wrist2, (480, 480))
            wrist1 = cv2.rotate(obs['wrist1'][0], cv2.ROTATE_180)
            wrist1 = cv2.cvtColor(wrist1, cv2.COLOR_RGB2BGR)
            wrist1 = cv2.resize(wrist1, (480, 480))
            combined = np.vstack((wrist2, wrist1))
            cv2.imshow("Wrist Views", combined)
            cv2.waitKey(waitkey)
            
    
            action = np.zeros_like(env.action_space.sample())
            if "intervene_action" in info:
                action = info['intervene_action']
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"xyz: {obs['state'][0][14:17]}")
            i+=1
        
if __name__ == "__main__":
    main()