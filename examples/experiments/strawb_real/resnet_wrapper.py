import gymnasium as gym
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import numpy as np
from serl_launcher.vision.resnet_v1 import PreTrainedResNetEncoder, resnetv1_configs
from serl_launcher.vision.data_augmentations import batched_random_crop
from serl_launcher.utils.train_utils import load_resnet10_params
import pickle as pkl
import os
import requests
from tqdm import tqdm
from typing import Dict, Iterable, Optional, Tuple
from einops import rearrange, repeat
import time
import cv2

class EncodingWrapper(nn.Module):
    """
    Encodes observations into a single flat encoding, adding additional
    functionality for adding proprioception and stopping the gradient.

    Args:
        encoder: The encoder network.
        use_proprio: Whether to concatenate proprioception (after encoding).
    """

    encoder: nn.Module

    @nn.compact
    def __call__(
        self,
        images: jnp.ndarray,
        train=False,
        stop_gradient=True,
    ) -> jnp.ndarray:
        # encode images with encoder
        
        images = self.encoder(images, train=train)

        if stop_gradient:
            images = jax.lax.stop_gradient(images)

        return images

class ResNet10Wrapper(gym.ObservationWrapper):
    def __init__(self, env, image_keys=["wrist1", "wrist2"], seed=0, augment=True, embedding_key="embedding"):
        """
        A wrapper to encode images using a ResNet-10 model and add embeddings to observations.

        Args:
            env: Base gym environment.
            image_keys: List of keys in the observation containing image data.
            pretrained: Whether to load pretrained weights for ResNet-10.
        """
        super().__init__(env)
        self.image_keys = image_keys
        self.rng = jax.random.key(seed)
        self.augment = augment
        self.embedding_key = embedding_key

        # Instantiate the ResNet-10 encoder
        pretrained_encoder = resnetv1_configs["resnetv1-10-frozen"](
            pre_pooling=True,
            name="pretrained_encoder",
        )

        self.encoder_def = EncodingWrapper(pretrained_encoder)            

        # Initialize encoder parameters
        if augment:
            dummy_observation = jnp.zeros((2*len(self.image_keys), *env.observation_space[image_keys[0]].shape[1:]))
        else:
            dummy_observation = jnp.zeros((len(self.image_keys), *env.observation_space[image_keys[0]].shape[1:]))

        encoder_params = self.encoder_def.init(self.rng, dummy_observation)
        
        self.encoder_params = self._load_resnet10_params(encoder_params)

        self._jit_encode = jax.jit(lambda params, obs: self.encoder_def.apply(params, obs, train=False))


        # Extend the observation space to include embeddings
        embedding_dim = self._jit_encode(self.encoder_params, dummy_observation).shape
        new_spaces = self.observation_space.spaces.copy()
        new_spaces[embedding_key] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(*embedding_dim,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(new_spaces)

    def _load_resnet10_params(self, encoder_params):
        """
        Load pretrained ResNet-10 parameters into the encoder.
        """
        file_name = "resnet10_params.pkl"
        file_path = os.path.expanduser("~/.serl/")
        os.makedirs(file_path, exist_ok=True)
        file_path = os.path.join(file_path, file_name)

        # Download pretrained weights if necessary
        if not os.path.exists(file_path):
            url = f"https://github.com/rail-berkeley/serl/releases/download/resnet10/{file_name}"
            print(f"Downloading file from {url}")
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024
            with open(file_path, "wb") as f:
                for data in response.iter_content(block_size):
                    f.write(data)
            print("Download complete!")

        # Load pretrained weights
        with open(file_path, "rb") as f:
            pretrained_params = pkl.load(f)

        del pretrained_params['output_head']
        param_count = sum(x.size for x in jax.tree_leaves(pretrained_params))
        print(
            f"Loaded {param_count/1e6}M parameters from ResNet-10 pretrained on ImageNet-1K"
        )

        new_encoder_params = encoder_params['params'][f"encoder"]
        if "pretrained_encoder" in new_encoder_params:
            new_encoder_params = new_encoder_params["pretrained_encoder"]
        for k in new_encoder_params:
            if k in pretrained_params:
                new_encoder_params[k] = pretrained_params[k]
                print(f"replaced {k} in pretrained_encoder")

        return encoder_params

    def observation(self, observation):
        """
        Add ResNet-10 embeddings to the observation dictionary.
        """
        
        self.rng, subrng = jax.random.split(self.rng)
        
        # Stack images for batch processing
        images = jnp.concatenate([observation[k] for k in self.image_keys], axis=0)
        if self.augment:
            aug_images = batched_random_crop(images, subrng, padding=4, num_batch_dims=1)
            images = jnp.concatenate([images, aug_images], axis=0)
        
        # Single device transfer for all images
        resnet_start_time = time.time()
        obs = jax.device_put(images)
        embeddings = self._jit_encode(self.encoder_params, obs)
        
        # Single host transfer for all embeddings
        embeddings_np = jax.device_get(embeddings)        
        observation[self.embedding_key] = embeddings_np
            
        return observation
