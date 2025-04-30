# Load trained model and run inference in the robomimic can env
#!/usr/bin/env python3
import argparse
import os
import time

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]    = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]   = "0.5"

import cv2
import jax
import jax.numpy as jnp
import numpy as np
from absl import flags  # for compatibility with SERL agent API
from flax.training import checkpoints

from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
    make_sac_pixel_agent_hybrid_dual_arm,
)
from experiments.mappings import CONFIG_MAPPING


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name",        required=True,
                   help="Name of experiment (must be in CONFIG_MAPPING)")
    p.add_argument("--checkpoint_path", required=True,
                   help="Directory containing `checkpoint_<step>` files")
    p.add_argument("--step",    type=int, default=None,
                   help="Which checkpoint to restore (default=latest)")
    p.add_argument("--episodes",type=int, default=10,
                   help="How many episodes to roll out")
    p.add_argument("--render", action="store_true",
                   help="If set, show frames in a window")
    return p.parse_args()


def main():
    args = parse_args()

    # 1) Build config & environment in inference mode (no gamepad)
    assert args.exp_name in CONFIG_MAPPING, f"Unknown exp_name {args.exp_name}"
    config = CONFIG_MAPPING[args.exp_name]()
    env = config.get_environment(
        fake_env=True,         # no gamepad intervention
        save_video=False,
        video_res=480,
        state_res=128,
        classifier=False,
        xirl=False,
        obs_horizon=1,
    )

    # 2) Instantiate agent with same setup_mode as training
    setup = config.setup_mode
    sample_obs    = env.observation_space.sample()
    sample_action = env.action_space.sample()
    if setup in ("single-arm-fixed-gripper", "dual-arm-fixed-gripper"):
        AgentCls = make_sac_pixel_agent
    elif setup == "single-arm-learned-gripper":
        AgentCls = make_sac_pixel_agent_hybrid_single_arm
    elif setup == "dual-arm-learned-gripper":
        AgentCls = make_sac_pixel_agent_hybrid_dual_arm
    else:
        raise ValueError(f"Unsupported setup_mode {setup}")

    agent = AgentCls(
        seed        = 0,
        sample_obs  = sample_obs,
        sample_action = sample_action,
        image_keys  = config.image_keys,
        encoder_type= config.encoder_type,
        discount    = config.discount,
    )

    # 3) Restore checkpoint
    #    pass `step=args.step` to restore_checkpoint to load a specific step,
    #    or leave it off to get the latest.
    agent_state = checkpoints.restore_checkpoint(
        ckpt_dir  = os.path.abspath(args.checkpoint_path),
        target    = agent.state,
        step      = args.step,
    )
    agent = agent.replace(state=agent_state)

    # 4) Roll out episodes
    total_success = 0
    total_length  = 0

    # optional CV window
    if args.render:
        cv2.namedWindow("policy", cv2.WINDOW_NORMAL)

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        ep_len = 0
        start = time.time()

        # while not done:
        for i in range(50):
            # greedy / deterministic action
            jax_obs = jax.device_put(obs)
            a = agent.sample_actions(
                observations = jax_obs,
                argmax       = True,
                seed         = jax.random.PRNGKey(int(time.time() * 1e6) % (2**31)),
            )
            a = np.asarray(jax.device_get(a))

            obs, reward, done, truncated, info = env.step(a)
            ep_len += 1

            if args.render:
                # pick one camera to display, e.g. agentview_image or first in config.image_keys
                img_key = config.image_keys[0]
                img = obs[img_key]
                # if shape (1,H,W,3) → drop batch
                if img.ndim == 4 and img.shape[0] == 1:
                    img = img[0]
                # HWC, uint8?
                if img.dtype != np.uint8:
                    img = (img * 255).clip(0,255).astype(np.uint8)
                # Resize to 480x480
                img = cv2.resize(img, (480, 480), interpolation=cv2.INTER_LINEAR)
                # Write reward on top of image
                cv2.putText(
                    img,
                    f"reward: {reward:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("policy", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(1)
                if key == ord("q"):
                    done = True

        dur = time.time() - start
        total_success += float(reward == 1.0)
        total_length  += ep_len
        print(f"Episode {ep+1:2d}: length={ep_len:3d}, success={int(reward==1.0)}, time={dur:.2f}s")

    # 5) Print summary
    succ_rate = total_success / args.episodes
    avg_len   = total_length  / args.episodes
    print(f"\n>>> Success rate: {succ_rate:.2%}")
    print(f">>> Average episode length: {avg_len:.1f} steps")

    if args.render:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
