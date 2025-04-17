# robomimic_gym_wrapper.py
import numpy as np
from collections import OrderedDict

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from robomimic.envs.wrappers import EnvWrapper          # robomimic base wrapper
from robomimic.envs.env_base import EnvBase             # robomimic env interface


class RobomimicGymWrapper(EnvWrapper, gym.Env):
    """
    Thin adapter that makes any robomimic `EnvBase` look like a Gymnasium env.

    Typical usage
    -------------
    >>> from robomimic.utils.env_utils import create_env
    >>> rm_env = create_env(env_type=1, env_name="Lift", render=False)
    >>> gym_env = RobomimicGymWrapper(rm_env, keys=None, flatten_obs=False)
    >>> obs, _ = gym_env.reset()
    >>> obs.shape or obs.keys()
    """

    metadata = {"render_modes": []}   # robomimic handles its own rendering
    render_mode = None

    # ------------------------------------------------------------------ #
    def __init__(self, env: EnvBase, keys=None, flatten_obs: bool = False):
        super().__init__(env=env)                  # keep robomimic wrapper chain
        assert isinstance(self.env, EnvBase), "Must wrap a robomimic EnvBase"

        # ------------- which observation keys are exposed to Gym --------
        if keys is None:
            keys = list(self.env.reset().keys())   # expose *everything*
        self.keys = keys
        self.flatten_obs = flatten_obs

        # ------------ build gym‑style observation & action spaces -------
        rm_obs = self.env.reset()                  # OrderedDict of np arrays

        if self.flatten_obs:
            flat = self._flatten_obs(rm_obs)
            high = np.inf * np.ones_like(flat, dtype=np.float32)
            low  = -high
            self.observation_space = spaces.Box(low, high, dtype=np.float32)
        else:
            self.observation_space = spaces.Dict({
                k: self._space_from_sample(rm_obs[k]) for k in self.keys if k in rm_obs
            })

        # Action space: robomimic keeps values in ‑1 … 1 by default
        act_dim = self.env.action_dimension
        self.action_space = spaces.Box(
            low=-np.ones(act_dim, dtype=np.float32),
            high=np.ones(act_dim, dtype=np.float32),
            dtype=np.float32,
        )

        # misc gym bookkeeping
        self.reward_range = (-np.inf, np.inf)
        self.spec = None

    # ------------------------------------------------------------------ #
    #  Gym‑compat helpers
    # ------------------------------------------------------------------ #
    def _space_from_sample(self, sample: np.ndarray):
        """Return a Box space that matches a sample ndarray."""
        if np.issubdtype(sample.dtype, np.integer):
            low, high = np.iinfo(sample.dtype).min, np.iinfo(sample.dtype).max
        else:
            low, high = -np.inf, np.inf
        return spaces.Box(low=low, high=high, shape=sample.shape, dtype=sample.dtype)

    def _flatten_obs(self, obs_dict: OrderedDict):
        """Concatenate selected keys into a 1‑D float32 array."""
        parts = [np.asarray(obs_dict[k]).ravel() for k in self.keys if k in obs_dict]
        return np.concatenate(parts, dtype=np.float32)

    def _filter_obs(self, obs_dict: OrderedDict):
        """Return a dict with only the requested keys."""
        return {k: obs_dict[k] for k in self.keys if k in obs_dict}

    # ------------------------------------------------------------------ #
    #  Gym API
    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(int(seed))
        obs_dict = self.env.reset()
        obs = self._flatten_obs(obs_dict) if self.flatten_obs else self._filter_obs(obs_dict)
        return obs, {}

    def step(self, action):
        # robomimic returns (obs_dict, reward, done, info)
        obs_dict, reward, done, info = self.env.step(np.asarray(action, dtype=np.float32))
        obs = self._flatten_obs(obs_dict) if self.flatten_obs else self._filter_obs(obs_dict)
        terminated, truncated = bool(done), False   # robomimic has single done flag
        return obs, float(reward), terminated, truncated, info

    # optional convenience
    def close(self):
        self.env.close()
