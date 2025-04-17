# serl_obs_wrappers_robomimic.py
import gymnasium as gym
import numpy as np
from copy import deepcopy

# -----------------------------------------------------------------------------#
#  tiny helpers (same signature as SERL’s originals)
# -----------------------------------------------------------------------------#
def flatten_space(space: gym.spaces.Dict) -> gym.spaces.Box:
    """Create a single Box that is the concatenation of all Dict sub‑spaces."""
    lows, highs = [], []
    for sub in space.spaces.values():
        assert isinstance(sub, gym.spaces.Box)
        lows.append(sub.low.reshape(-1))
        highs.append(sub.high.reshape(-1))
    low  = np.concatenate(lows, dtype=np.float32)
    high = np.concatenate(highs, dtype=np.float32)
    return gym.spaces.Box(low=low, high=high, dtype=np.float32)


def flatten(space: gym.spaces.Dict, sample: dict) -> np.ndarray:
    """Flatten a single dict sample according to `space` ordering."""
    parts = [np.asarray(sample[k], dtype=np.float32).ravel() for k in space.spaces.keys()]
    return np.concatenate(parts, dtype=np.float32)


# -----------------------------------------------------------------------------#
#  Robomimic‑friendly SERL observation wrapper
# -----------------------------------------------------------------------------#
class SERLObsWrapper(gym.ObservationWrapper):
    """
    ‑ For robomimic envs where *all* observations live in one dict.
    ‑ Separates `image_keys` out untouched;
      everything else is flattened into a single "state" vector.
    """

    def __init__(self, env, image_keys, proprio_keys=None):
        """
        Args
        ----
        env          : (gym.Env or compatible) a RobomimicGymWrapper instance.
        image_keys   : list[str]  keys whose values are image tensors (H,W,C)
        proprio_keys : list[str] or None
            Which non‑image keys to include in the flattened state.  If None,
            we use **all** keys that are *not* in `image_keys`.
        """
        super().__init__(env)

        self.image_keys = set(image_keys)

        # Split original observation space
        full_space = env.observation_space
        assert isinstance(full_space, gym.spaces.Dict)

        # ------------------- build proprio (non‑image) sub‑space -----------
        if proprio_keys is None:
            proprio_keys = [k for k in full_space.spaces.keys() if k not in self.image_keys]
        self.proprio_keys = proprio_keys

        self.proprio_space = gym.spaces.Dict(
            {k: deepcopy(full_space[k]) for k in self.proprio_keys}
        )

        # ------------------- new wrapped observation space -----------------
        img_spaces = {}
        for k in self.image_keys:
            chw_shape   = deepcopy(full_space[k]).shape
            hwc_shape   = self._chw_to_hwc_shape(chw_shape)            # <‑‑ only change
            img_spaces[k] = gym.spaces.Box(
                low   = 0,
                high  = 255,
                shape = hwc_shape,                                # H,W,C (or T,H,W,C)
                dtype = np.uint8,
            )

        self.observation_space = gym.spaces.Dict(
            {"state": flatten_space(self.proprio_space), **img_spaces}
        )

    def _chw_to_hwc_shape(self, shape):
        """
        (C,H,W)   -> (H,W,C)
        (T,C,H,W) -> (T,H,W,C)
        """
        if len(shape) == 3:          # C,H,W
            c, h, w = shape
            return (h, w, c)
        elif len(shape) == 4:        # T,C,H,W
            t, c, h, w = shape
            return (t, h, w, c)
        else:
            raise ValueError("unexpected image shape " + str(shape))


    # ------------------------------------------------------------------ #
    #  Observation conversion
    # ------------------------------------------------------------------ #
    def observation(self, obs):
        """
        Convert robomimic obs‑dict → dict{ "state": flat, img1, img2, … }.
        """
        state_dict = {k: obs[k] for k in self.proprio_keys}
        flat_state = flatten(self.proprio_space, state_dict)

        new_obs = {"state": flat_state}
        for k in self.image_keys:
            new_obs[k] = self._to_uint8_hwc(obs[k])
        return new_obs

    # Gymnasium’s ObservationWrapper already calls self.observation()
    # inside step / reset, but we want to preserve the two‑value reset API.
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info
    
    def _to_uint8_hwc(self, arr):
        # (T,)C,H,W  → (T,)H,W,C  + cast to uint8
        if arr.ndim == 3:          # C,H,W
            arr = arr.transpose(1,2,0)
        elif arr.ndim == 4:        # T,C,H,W
            arr = arr.transpose(0,2,3,1)
        if arr.dtype != np.uint8:
            arr = (arr * 255).clip(0,255).astype(np.uint8)
        return arr