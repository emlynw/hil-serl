import torch
import torch.nn as nn
import torchvision.transforms as transforms
from experiments.strawb_real.models import Resnet18LinearEncoderNet
import gym
import numpy as np
import pickle
import cv2

class xirlResnet18RewardWrapper(gym.Wrapper):
    def __init__(self, env, image_key="wrist1", device=None):
        super().__init__(env)
        self.image_key = image_key
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        goal_emb_path = "/home/emlyn/xirl_results/pretrain_runs/dataset=strawb_pick_algo=xirl_embodiment=human/goal_emb.pkl"
        distance_scale_path = "/home/emlyn/xirl_results/pretrain_runs/dataset=strawb_pick_algo=xirl_embodiment=human/distance_scale.pkl"
        xirl_resnet_18 = torch.load('/home/emlyn/xirl_results/pretrain_runs/dataset=strawb_pick_algo=xirl_embodiment=human/checkpoints/501.ckpt')
        with open(goal_emb_path, "rb") as fp:
            self.goal_emb = pickle.load(fp)
        with open(distance_scale_path, "rb") as fp:
            self.distance_scale = pickle.load(fp)
        model = Resnet18LinearEncoderNet(embedding_size=32, num_ctx_frames=1,
                                normalize_embeddings=False, learnable_temp=False)
        model.load_state_dict(xirl_resnet_18['model'])
        model.to(self.device).eval()
        self.model = model
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        orig_reward = reward
        pixels = obs[self.image_key].copy()
        pixels = pixels[0]
        pixels = cv2.rotate(pixels, cv2.ROTATE_180)
        pixels = np.transpose(pixels, (2, 0, 1))
        pixels_shape = pixels.shape
        pixels = torch.from_numpy(pixels.reshape(1 ,1 ,*pixels_shape)).float()
        pixels = pixels / 255.0
        pixels = pixels.to(self.device)
        with torch.no_grad():
            obs_emb = self.model.infer(pixels).embs
        obs_emb = obs_emb.cpu().numpy()
        dist = np.linalg.norm(obs_emb - self.goal_emb)
        reward = -dist * self.distance_scale
        info['orig_reward'] = orig_reward
        return obs, reward, terminated, truncated, info

