import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import cv2
from franka_ros2_gym import envs
import numpy as np
from gamepad_wrapper import GamepadIntervention
from wrappers import Quat2EulerWrapper, ActionState, VideoRecorderReal, ExplorationMemory
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from resnet_wrapper import ResNet10Wrapper
from experiments.mappings import CONFIG_MAPPING

def main():
    render_mode = "rgb_array"
    exp_name = "strawb_real"
    depth = False
    save_video = True
    image_keys = ['wrist1', 'wrist2']
    video_dir = "./videos"
    obs_horizon=1
    proprio_keys = ["tcp_pos", "tcp_orientation", "tcp_vel", "gripper_pos", "gripper_vec", "exploration"]
    config = CONFIG_MAPPING[exp_name]()

    env = config.get_environment(fake_env=False, save_video=True, video_res=480, state_res=256, video_dir="./videos", classifier=True)


    print(env.observation_space['state'].shape)
     
    waitkey = 10
    resize_resolution = (480, 480)

    while True:
        # reset the environment
        i=0
        terminated = False
        truncated = False
        obs, info = env.reset()
        rotate = True
        
        while not terminated and not truncated:
            print(obs['state'].shape)
            print(i)
            wrist2 = obs["wrist2"][0]
            cv2.imshow("wrist2", cv2.resize(cv2.cvtColor(wrist2, cv2.COLOR_RGB2BGR), resize_resolution))
            wrist1 = cv2.rotate(obs['wrist1'][0], cv2.ROTATE_180)
            cv2.imshow("wrist1", cv2.resize(cv2.cvtColor(wrist1, cv2.COLOR_RGB2BGR), resize_resolution))
            if depth:
                wrist2_depth = obs["wrist2_depth"]
                cv2.imshow("wrist2_depth", cv2.resize(wrist2_depth, resize_resolution))
                wrist1_depth = cv2.rotate(obs['wrist1_depth'], cv2.ROTATE_180)
                cv2.imshow("wrist1_depth", cv2.resize(wrist1_depth, resize_resolution))

            cv2.waitKey(waitkey)
            
    
            action = np.zeros_like(env.action_space.sample())
            if "intervene_action" in info:
                action = info['intervene_action']
            
            obs, reward, terminated, truncated, info = env.step(action)
            print(info["success_key"])
            i+=1
        
if __name__ == "__main__":
    main()