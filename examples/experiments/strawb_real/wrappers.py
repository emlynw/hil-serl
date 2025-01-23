import gymnasium as gym
from gymnasium.spaces import Box, Dict
import numpy as np
from collections import deque
import cv2
import imageio
import os
from scipy.spatial.transform import Rotation as R


class PixelFrameStack(gym.Wrapper):
    def __init__(self, env, num_frames, stack_key='pixels'):
        super().__init__(env)
        self._num_frames = num_frames
        self.stack_key = stack_key
        self._frames = deque([], maxlen=num_frames)
        pixels_shape = env.observation_space[stack_key].shape
        

        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self.observation_space[stack_key] = Box(low=0, high=255, shape=(num_frames*pixels_shape[-1], *pixels_shape[:-1]), dtype=np.uint8)

    def _transform_observation(self, obs):
        assert len(self._frames) == self._num_frames
        obs[self.stack_key] = np.concatenate(list(self._frames), axis=0)
        return obs

    def _extract_pixels(self, obs):
        pixels = obs[self.stack_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        obs, info = self.env.reset()
        pixels = self._extract_pixels(obs)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        pixels = self._extract_pixels(obs)
        self._frames.append(pixels)
        return self._transform_observation(obs), reward, terminated, truncated, info
    
class StateFrameStack(gym.Wrapper):
    def __init__(self, env, num_frames, stack_key='state', flatten=True):
        super().__init__(env)
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)   
        self.stack_key = stack_key
        self.flatten = flatten

        shape = self.env.observation_space[stack_key].shape
        if isinstance(shape, int):
            shape = (shape,)  # Convert to a tuple for consistency
        else:
            shape = shape  # If it's already a tuple or list, keep it as is
        if flatten: 
            self.observation_space[stack_key] = Box(low=-np.inf, high=np.inf, shape=(num_frames * shape[-1],), dtype=np.float32)
        else:
            self.observation_space[stack_key] = Box(low=-np.inf, high=np.inf, shape=(num_frames, *shape), dtype=np.float32)

    def _transform_observation(self):
        assert len(self._frames) == self._num_frames
        obs = np.array(self._frames)
        if self.flatten:
            obs = obs.flatten()
        return obs

    def reset(self):
        obs, info = self.env.reset()
        for _ in range(self._num_frames):
            self._frames.append(obs[self.stack_key])
        obs[self.stack_key] = self._transform_observation()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs[self.stack_key])
        obs[self.stack_key] = self._transform_observation()
        return obs, reward, terminated, truncated, info

class CustomPixelObservation(gym.ObservationWrapper):
  """Resize the observation to a given resolution"""
  def __init__(self, env, pixel_key='pixels', crop_resolution=None, resize_resolution=None):
    super().__init__(env)
    if isinstance(resize_resolution, int):
      resize_resolution = (resize_resolution, resize_resolution)
    if isinstance(crop_resolution, int):
      crop_resolution = (crop_resolution, crop_resolution)
    self.pixel_key = pixel_key
    self.crop_resolution = crop_resolution
    self.resize_resolution = resize_resolution
    self.observation_space[pixel_key] = Box(low=0, high=255, shape=(*self.resize_resolution, 3), dtype=np.uint8)
    
  def observation(self, observation):
    if self.crop_resolution is not None:
      if observation[self.pixel_key].shape[:2] != self.crop_resolution:
        center = observation[self.pixel_key].shape
        x = center[1]/2 - self.crop_resolution[1]/2
        y = center[0]/2 - self.crop_resolution[0]/2
        observation[self.pixel_key]= observation[self.pixel_key][int(y):int(y+self.crop_resolution[0]), int(x):int(x+self.crop_resolution[1])]
    if self.resize_resolution is not None:
      if observation[self.pixel_key].shape[:2] != self.resize_resolution:
        observation[self.pixel_key] = cv2.resize(
            observation[self.pixel_key],
            dsize=self.resize_resolution,
            interpolation=cv2.INTER_CUBIC,
        )
    return observation    

class VideoRecorder(gym.Wrapper):
  """Wrapper for rendering and saving rollouts to disk.
  Reference: https://github.com/ikostrikov/jaxrl/
  """

  def __init__(
      self,
      env,
      save_dir,
      crop_resolution,
      resize_resolution,
      fps = 20,
      current_episode=0,
      record_every=2,
  ):
    super().__init__(env)

    self.save_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)
    num_vids = len(os.listdir(save_dir))
    current_episode = num_vids*record_every

    if isinstance(resize_resolution, int):
      self.resize_resolution = (resize_resolution, resize_resolution)
    if isinstance(crop_resolution, int):
      self.crop_resolution = (crop_resolution, crop_resolution)

    self.resize_h, self.resize_w = self.resize_resolution
    self.crop_h, self.crop_w = self.crop_resolution
    self.fps = fps
    self.enabled = True
    self.current_episode = current_episode
    self.record_every = record_every
    self.frames = []

  def step(self, action):
    observation, reward, terminated, truncated, info = self.env.step(action)
    if self.current_episode % self.record_every == 0:
      frame = self.env.render()[1]
      if self.crop_resolution is not None:
        # Crop
        if frame.shape[:2] != (self.crop_h, self.crop_w):
          center = frame.shape
          x = center[1]/2 - self.crop_w/2
          y = center[0]/2 - self.crop_h/2
          frame = frame[int(y):int(y+self.crop_h), int(x):int(x+self.crop_w)]
      if self.resize_resolution is not None:
        if frame.shape[:2] != (self.resize_h, self.resize_w):
          frame = cv2.resize(
              frame,
              dsize=(self.resize_h, self.resize_w),
              interpolation=cv2.INTER_CUBIC,
          )
      # Write rewards on the frame
      cv2.putText(
          frame,
          f"{reward:.3f}",
          (10, 40),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.5,
          (0, 255, 0),
          1,
          cv2.LINE_AA,
      )
      # Save
      self.frames.append(frame)
    if terminated or truncated:
      if self.current_episode % self.record_every == 0:
        filename = os.path.join(self.save_dir, f"{self.current_episode}.mp4")
        imageio.mimsave(filename, self.frames, fps=self.fps)
        self.frames = []
      self.current_episode += 1
    return observation, reward, terminated, truncated, info
  
class VideoRecorderReal(gym.Wrapper):
    """Wrapper for rendering and saving rollouts to disk from a specific camera."""

    def __init__(
        self,
        env,
        save_dir,
        crop_resolution,
        resize_resolution,
        camera_name="wrist2",
        fps=10,
        current_episode=0,
        record_every=2,
    ):
        super().__init__(env)

        self.save_dir = save_dir
        self.camera_name = camera_name
        os.makedirs(save_dir, exist_ok=True)
        num_vids = len([f for f in os.listdir(save_dir) if f.endswith(camera_name)])
        current_episode = num_vids * record_every

        if isinstance(resize_resolution, int):
            self.resize_resolution = (resize_resolution, resize_resolution)
        if isinstance(crop_resolution, int):
            self.crop_resolution = (crop_resolution, crop_resolution)

        self.resize_h, self.resize_w = self.resize_resolution
        self.crop_h, self.crop_w = self.crop_resolution
        self.fps = fps
        self.enabled = True
        self.current_episode = current_episode
        self.record_every = record_every
        self.frames = []

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        if self.current_episode % self.record_every == 0:
            frame = observation[self.camera_name]
            
            if self.crop_resolution is not None:
                if frame.shape[:2] != (self.crop_h, self.crop_w):
                    center = frame.shape
                    x = center[1] // 2 - self.crop_w // 2
                    y = center[0] // 2 - self.crop_h // 2
                    frame = frame[int(y):int(y + self.crop_h), int(x):int(x + self.crop_w)]

            if self.resize_resolution is not None:
                if frame.shape[:2] != (self.resize_h, self.resize_w):
                    frame = cv2.resize(
                        frame,
                        dsize=(self.resize_w, self.resize_h),
                        interpolation=cv2.INTER_CUBIC,
                    )

            cv2.putText(
                frame,
                f"{reward:.3f}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

            self.frames.append(frame)

        if terminated or truncated:
            if self.current_episode % self.record_every == 0:
                filename = os.path.join(self.save_dir, f"{self.current_episode}_{self.camera_name}.mp4")
                imageio.mimsave(filename, self.frames, fps=self.fps)
                self.frames = []

            self.current_episode += 1

        return observation, reward, terminated, truncated, info
  
class ActionRepeat(gym.Wrapper):
  """Repeat the agent's action N times in the environment.
  Reference: https://github.com/ikostrikov/jaxrl/
  """

  def __init__(self, env, repeat):
    """Constructor.
    Args:
      env: A gym env.
      repeat: The number of times to repeat the action per single underlying env
        step.
    """
    super().__init__(env)

    assert repeat > 1, "repeat should be greater than 1."
    self._repeat = repeat

  def step(self, action):
    total_reward = 0.0
    for _ in range(self._repeat):
      observation, reward, terminated, truncated, info = self.env.step(action)
      total_reward += reward
      if terminated or truncated:
        break
    return observation, total_reward, terminated, truncated, info
  
class FrankaObservation(gym.ObservationWrapper):
  """Resize the observation to a given resolution"""
  def __init__(self, env, camera_name='front'):
    super().__init__(env)
    self.camera_name = camera_name
    pixel_space = self.observation_space['images'][camera_name]
    self.state_keys = ['panda/tcp_pos', 'panda/tcp_orientation', 'panda/gripper_pos', 'panda/gripper_vec']
    state_dim = 0
    for key in self.state_keys:
      state_dim += self.observation_space['state'][key].shape[0]
    state_space = Box(-np.inf, np.inf, shape=(state_dim,), dtype=np.float32)
    self.observation_space = Dict({'pixels': pixel_space, 'state': state_space})
    
  def observation(self, observation):
    pixels = observation['images'][self.camera_name]
    state = np.concatenate([observation['state'][key] for key in self.state_keys])
    observation = {}
    observation['pixels'] = pixels
    observation['state'] = state
    return observation    
  
class FrankaDualCamObservation(gym.ObservationWrapper):
  """Resize the observation to a given resolution"""
  def __init__(self, env, camera1_name='wrist1', camera2_name='wrist2'):
    super().__init__(env)
    self.camera1_name = camera1_name
    self.camera2_name = camera2_name
    img1_space = self.observation_space['images'][camera1_name]
    img2_space = self.observation_space['images'][camera2_name]
    self.state_keys = ['panda/tcp_pos', 'panda/tcp_orientation', 'panda/gripper_pos', 'panda/gripper_vec']
    state_dim = 0
    for key in self.state_keys:
      state_dim += self.observation_space['state'][key].shape[0]
    state_space = Box(-np.inf, np.inf, shape=(state_dim,), dtype=np.float32)
    self.observation_space = Dict({'img1': img1_space, 'img2': img2_space, 'state': state_space})
    
  def observation(self, observation):
    img1 = observation['images'][self.camera1_name]
    img2 = observation['images'][self.camera2_name]
    state = np.concatenate([observation['state'][key] for key in self.state_keys])
    observation = {}
    observation['img1'] = img1
    observation['img2'] = img2
    observation['state'] = state
    return observation  
  
class ActionState(gym.Wrapper):
    # Add previous action to the state
    def __init__(self, env, state_key='state', action_key='action'):
        super().__init__(env)
        self.action_key = action_key
        self.state_key = state_key
        self.action_dim = env.action_space.shape[0]
        self.state_dim = env.observation_space[state_key].shape[0]
        self.observation_space[state_key] = Box(low=-np.inf, high=np.inf, shape=(self.state_dim + self.action_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset()
        self.action = np.zeros(self.action_dim)
        obs[self.state_key] = np.concatenate([obs[self.state_key], self.action])
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs[self.state_key] = np.concatenate([obs[self.state_key], action])
        return obs, reward, terminated, truncated, info
    
class Quat2EulerWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to euler angles
    """

    def __init__(self, env):
        super().__init__(env)
        assert env.observation_space["state"]["panda/tcp_orientation"].shape == (4,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["panda/tcp_orientation"] = gym.spaces.Box(
            -np.inf, np.inf, shape=(3,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_orientation = observation["state"]["panda/tcp_orientation"]
        observation["state"]["panda/tcp_orientation"] = R.from_quat(tcp_orientation).as_euler("xyz")
        return observation
    
class ExplorationMemory(gym.Wrapper):
    # Add max and min xyz to the state
    def __init__(self, env, state_key='state', ee_key='panda/tcp_pos', exploration_key='exploration'):
        super().__init__(env)
        self.state_key = state_key
        self.ee_key = ee_key
        self.exploration_key = exploration_key

        # Update observation space to include the 'exploration' key
        original_state_space = self.observation_space[state_key]
        exploration_space = Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            **self.observation_space.spaces,
            state_key: gym.spaces.Dict({
                **original_state_space.spaces,
                exploration_key: exploration_space,
            })
        })

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset()
        self.min_xyz = obs[self.state_key][self.ee_key]
        self.max_xyz = obs[self.state_key][self.ee_key]
        obs[self.state_key][self.exploration_key] = np.concatenate([self.min_xyz, self.max_xyz])
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.min_xyz = np.minimum(self.min_xyz, obs[self.state_key][self.ee_key])
        self.max_xyz = np.maximum(self.max_xyz, obs[self.state_key][self.ee_key])
        obs[self.state_key][self.exploration_key] = np.concatenate([self.min_xyz, self.max_xyz])
        return obs, reward, terminated, truncated, info