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
    use_proprio: bool
    proprio_latent_dim: int = 64
    enable_stacking: bool = False
    image_keys: Iterable[str] = ("image",)

    @nn.compact
    def __call__(
        self,
        observations: Dict[str, jnp.ndarray],
        train=False,
        stop_gradient=False,
        is_encoded=False,
    ) -> jnp.ndarray:
        # encode images with encoder
        encoded = []
        for image_key in self.image_keys:
            image = observations[image_key]
            if not is_encoded:
                if self.enable_stacking:
                    # Combine stacking and channels into a single dimension
                    if len(image.shape) == 4:
                        image = rearrange(image, "T H W C -> H W (T C)")
                    if len(image.shape) == 5:
                        image = rearrange(image, "B T H W C -> B H W (T C)")

            image = self.encoder[image_key](image, train=train, encode=not is_encoded)

            if stop_gradient:
                image = jax.lax.stop_gradient(image)

            encoded.append(image)

        return encoded

class ResNet10Wrapper(gym.ObservationWrapper):
    def __init__(self, env, image_keys=("wrist1", "wrist2"), pooling_method="spatial_learned_embeddings", pretrained=True, seed=0):
        """
        A wrapper to encode images using a ResNet-10 model and add embeddings to observations.

        Args:
            env: Base gym environment.
            image_keys: List of keys in the observation containing image data.
            pooling_method: Pooling method for the ResNet-10 encoder.
            pretrained: Whether to load pretrained weights for ResNet-10.
        """
        super().__init__(env)
        self.image_keys = image_keys
        self.rng = jax.random.key(1)
        print(f"rng: {self.rng}")

        # Instantiate the ResNet-10 encoder
        pretrained_encoder = resnetv1_configs["resnetv1-10-frozen"](
            pre_pooling=True,
            name="pretrained_encoder",
        )
        encoders = {
            image_key: PreTrainedResNetEncoder(
                pooling_method=pooling_method,
                num_spatial_blocks=8,
                bottleneck_dim=256,
                pretrained_encoder=pretrained_encoder,
                name=f"encoder_{image_key}",
            )
            for image_key in image_keys
        }

        self.encoder_def = EncodingWrapper(encoders, use_proprio=False, enable_stacking=True, image_keys=image_keys)

        # Initialize encoder parameters
        dummy_observation = {
            image_key: jnp.zeros((128, 128, 3)) for image_key in image_keys
        }

        # Example: Accessing `params` without unfreezing
        encoder_params = self.encoder_def.init(self.rng, dummy_observation)

        # Load pretrained weights if specified
        if pretrained:
            self.encoder_params = self._load_resnet10_params(encoder_params, image_keys)

        self._jit_encode = jax.jit(lambda params, obs: self.encoder_def.apply(params, obs, train=False))

        # Extend the observation space to include embeddings
        embedding_dim = 256  # Match ResNet-10 bottleneck_dim
        new_spaces = self.observation_space.spaces.copy()
        for image_key in image_keys:
            embedding_key = f"embedding_{image_key}"
            new_spaces[embedding_key] = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(embedding_dim,), dtype=np.float32
            )
        self.observation_space = gym.spaces.Dict(new_spaces)

    def _load_resnet10_params(self, encoder_params, image_keys):
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

        for image_key in image_keys:
            new_encoder_params = encoder_params['params'][f"encoder_{image_key}"]
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
        # Create a dictionary of just the images we need to process
        image_dict = {k: observation[k] for k in self.image_keys}
        aug_images = {f"aug_{k}":batched_random_crop(observation[k], subrng, padding=4, num_batch_dims=1) for k in self.image_keys}  
        image_dict.update(aug_images)  
        # Single device transfer for all images
        obs = jax.device_put(image_dict)
        embeddings = self._jit_encode(self.encoder_params, obs)
        
        # Single host transfer for all embeddings
        embeddings_np = jax.device_get(embeddings)
        
        for image_key, embedding in zip(self.image_keys, embeddings_np):
            observation[f"embedding_{image_key}"] = embedding
            
        return observation
