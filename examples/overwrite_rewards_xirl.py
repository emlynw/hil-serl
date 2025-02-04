import os
import pickle
import numpy as np
import torch
import cv2

from experiments.strawb_real.models import Resnet18LinearEncoderNet

def load_xirl_model(device):
    """
    Loads the ResNet18-based XIRL model along with the goal embedding and distance scale.
    Adjust paths as needed.
    """
    # Paths
    goal_emb_path = "/home/emlyn/xirl_results/pretrain_runs/dataset=strawb_pick_128_algo=xirl_embodiment=human/goal_emb.pkl"
    distance_scale_path = "/home/emlyn/xirl_results/pretrain_runs/dataset=strawb_pick_128_algo=xirl_embodiment=human/distance_scale.pkl"
    ckpt_path = '/home/emlyn/xirl_results/pretrain_runs/dataset=strawb_pick_128_algo=xirl_embodiment=human/checkpoints/800.ckpt'

    # Load embeddings
    with open(goal_emb_path, "rb") as fp:
        goal_emb = pickle.load(fp)
    with open(distance_scale_path, "rb") as fp:
        distance_scale = pickle.load(fp)

    # Load the checkpoint (contains state_dict for the model)
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Build the model
    model = Resnet18LinearEncoderNet(
        embedding_size=128, 
        num_ctx_frames=1,
        normalize_embeddings=False, 
        learnable_temp=False
    )
    model.load_state_dict(checkpoint['model'])
    model.to(device).eval()

    return model, goal_emb, distance_scale


def compute_xirl_reward(
    image: np.ndarray,
    model: Resnet18LinearEncoderNet,
    goal_emb: np.ndarray,
    distance_scale: float,
    device: str
) -> float:
    """
    Given an image of shape (1, H, W, 3),
    rotate by 180, embed it with the XIRL model,
    then compute the distance-based reward.
    """
    # Remove the leading batch dimension if shape is (1, H, W, 3).
    if image.shape[0] == 1:
        image = image[0]  # shape now (H, W, 3)

    # Rotate 180 degrees
    image = cv2.rotate(image, cv2.ROTATE_180)

    # Convert from (H, W, 3) to (3, H, W) for PyTorch
    image = np.transpose(image, (2, 0, 1))  # shape = (3, H, W)

    # Add a batch dimension for PyTorch: (1, 3, H, W) => but we also want (B, T, C, H, W).
    # Since num_ctx_frames=1, we can treat T=1 => shape (1, 1, 3, H, W).
    image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)  # (1, 1, 3, H, W)
    image_tensor = image_tensor / 255.0  # Normalize to [0,1]
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        out = model.infer(image_tensor)
        obs_emb = out.embs.cpu().numpy()  # shape (1, D)

    # Compute L2 distance, then convert to negative reward scaled by distance_scale
    dist = np.linalg.norm(obs_emb - goal_emb)
    return float(-dist * distance_scale)


def rewrite_rewards_in_pkl(
    input_dir: str,
    output_dir: str,
    image_key: str = "wrist1",
):
    """
    Load all .pkl demo files in `input_dir`, recompute rewards using the XIRL model,
    then save updated transitions to `output_dir`.
    """
    # Decide on device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    model, goal_emb, distance_scale = load_xirl_model(device)

    # Make sure output_dir exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all pkl files in input_dir
    for fname in os.listdir(input_dir):
        if fname.endswith(".pkl"):
            in_path = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, fname)

            print(f"Processing {in_path}...")

            # Load transitions
            with open(in_path, "rb") as f:
                transitions = pickle.load(f)

            # transitions is assumed to be a list of dicts with structure:
            # {
            #   'observations': { image_key: np.ndarray of shape (1, H, W, 3), ... },
            #   'rewards': float,
            #   ...
            # }
            # We'll update transitions[i]['rewards'] with the new XIRL-based reward.
            for t in transitions:
                obs = t["observations"]
                if image_key not in obs:
                    # If the key doesn't exist, skip
                    continue

                # Compute new reward
                new_reward = compute_xirl_reward(
                    obs[image_key],
                    model,
                    goal_emb,
                    distance_scale,
                    device
                )
                t["rewards"] = new_reward

            # Save the updated transitions
            with open(out_path, "wb") as f:
                pickle.dump(transitions, f)

            print(f"Saved updated transitions to {out_path}")

    print("Done rewriting rewards.")


if __name__ == "__main__":
    # Example usage:
    # python rewrite_rewards.py
    # Make sure to adjust input_dir and output_dir to your actual paths.

    input_dir = "/home/emlyn/rl_franka/hil-serl/examples/demo_data_new"     # directory containing original .pkl files
    output_dir = "/home/emlyn/rl_franka/hil-serl/examples/demo_data_new_xirl"    # directory to save updated .pkl files

    rewrite_rewards_in_pkl(
        input_dir=input_dir,
        output_dir=output_dir,
        image_key="wrist1",  # or any other key used in your transitions
    )
