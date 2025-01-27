import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import cv2
from franka_ros2_gym import envs
import numpy as np
from gamepad_wrapper import GamepadIntervention
from wrappers import Quat2EulerWrapper, ActionState, VideoRecorderReal, ExplorationMemory
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper

def main():
    render_mode = "rgb_array"
    depth = False
    save_video = True
    image_keys = ['wrist1', 'wrist2']
    video_dir = "./videos"
    obs_horizon=1
    proprio_keys = ["panda/tcp_pos", "panda/tcp_orientation", "panda/tcp_vel", "panda/gripper_pos", "panda/gripper_vec", "exploration"]
    

    env = gym.make("franka_ros2_gym/ReachIKDeltaRealStrawbEnv", pos_scale = 0.2, rot_scale=1.0, cameras=image_keys, randomize_domain=False, ee_dof=6)
    env = TimeLimit(env, max_episode_steps=300)
    env = GamepadIntervention(env)
    env = ExplorationMemory(env)
    env = Quat2EulerWrapper(env)
    env = SERLObsWrapper(env, proprio_keys=proprio_keys)
    if save_video:
        for image_name in image_keys:
            env = VideoRecorderReal(env, video_dir, camera_name=image_name, crop_resolution=480, resize_resolution=480, fps=10, record_every=1)
    env = ActionState(env)
    env = ChunkingWrapper(env, obs_horizon=obs_horizon, act_exec_horizon=None)
     
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
            i+=1
        
if __name__ == "__main__":
    main()
