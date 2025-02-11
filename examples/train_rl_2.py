#!/usr/bin/env python3

import glob
import time
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import os
import re
import copy
import pickle as pkl
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from natsort import natsorted
from copy import deepcopy
import cv2

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.utils.train_utils import _unpack

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
    make_sac_pixel_agent_hybrid_dual_arm,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 5, "Number of trajectories to evaluate.")
flags.DEFINE_boolean("save_video", False, "Save video.")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging


devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))

import os
import re

def get_latest_checkpoint_step(checkpoint_path):
    """
    Get the highest step number from valid checkpoint folders.

    Args:
        checkpoint_path (str): Path to the directory containing checkpoint folders.

    Returns:
        int: The highest step number, or None if no valid checkpoint folders are found.
    """
    # Define the naming pattern for valid checkpoint folders
    step_pattern = re.compile(r"^checkpoint_(\d+)$")

    # List all valid checkpoint folders in the directory
    step_numbers = []
    for folder in os.listdir(checkpoint_path):
        if os.path.isdir(os.path.join(checkpoint_path, folder)):
            match = step_pattern.match(folder)
            if match:
                step_numbers.append(int(match.group(1)))  # Extract the step number

    # If no valid checkpoint folders are found
    if not step_numbers:
        print("No valid checkpoint folders found.")
        return None

    # Return the highest step number
    return max(step_numbers)



##############################################################################


def actor(agent, data_store, env, sampling_rng):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    
    start_step = (
        int(os.path.basename(natsorted(glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")))[-1])[12:-4]) + 1
        if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path)
        else 0
    )

    datastore_dict = {
        "actor_env": data_store,
    }

    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_stores=datastore_dict,
        wait_for_server=True,
        timeout_ms=3000,
    )

    eval_env = config.get_environment(
        fake_env=FLAGS.learner,
        save_video=True,
        video_dir= os.path.join(FLAGS.checkpoint_path, "eval_videos"),
        classifier=False,
    )

    # Function to update the agent with new params
    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    transitions = []

    obs, _ = env.reset()
    done = False

    # training loop
    timer = Timer()
    running_return = 0.0
    prev_latest_step = 0

    pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True)
    for step in pbar:
        timer.tick("total")
        # Check if there is a file in the checkpont path
        if step % config.eval_period == 0:
            latest_step = get_latest_checkpoint_step(FLAGS.checkpoint_path)
            if latest_step is not None and latest_step > prev_latest_step:
                print(f"evaluating at step {step}...")
                eval_agent = deepcopy(agent) # create a new agent for evaluation
                success_counter = 0
                time_list = []

                ckpt = checkpoints.restore_checkpoint(
                    os.path.abspath(FLAGS.checkpoint_path),
                    eval_agent.state,
                    step=latest_step,
                )
                eval_agent = eval_agent.replace(state=ckpt)

                for episode in range(FLAGS.eval_n_trajs):
                    obs, _ = eval_env.reset()
                    done = False
                    truncated = False
                    start_time = time.time()
                    while not (done or truncated):
                        sampling_rng, key = jax.random.split(sampling_rng)
                        actions = eval_agent.sample_actions(
                            observations=jax.device_put(obs),
                            argmax=True,
                            seed=key
                        )
                        actions = np.asarray(jax.device_get(actions))

                        next_obs, reward, done, truncated, info = eval_env.step(actions)
                        obs = next_obs

                        if done:
                            if reward:
                                dt = time.time() - start_time
                                time_list.append(dt)
                                print(dt)

                            success_counter += reward
                            print(reward)
                            print(f"{success_counter}/{episode + 1}")

                print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
                print(f"average time: {np.mean(time_list)}")
                prev_latest_step = latest_step

        with timer.context("sample_actions"):
            if step < config.random_steps:
                actions = env.action_space.sample()
            else:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    seed=key,
                    argmax=False,
                )
                actions = np.asarray(jax.device_get(actions))

        # Step environment
        with timer.context("step_env"):
            next_obs, reward, done, truncated, info = env.step(actions)
            if "left" in info:
                info.pop("left")
            if "right" in info:
                info.pop("right")

            running_return += reward
            transition = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - done,
                dones=done,
            )
            if 'grasp_penalty' in info:
                transition['grasp_penalty']= info['grasp_penalty']
            data_store.insert(transition)
            transitions.append(copy.deepcopy(transition))

            obs = next_obs
            if done or truncated:
                stats = {"environment": info}  # send stats to the learner to log
                client.request("send-stats", stats)
                pbar.set_description(f"last return: {running_return}")
                running_return = 0.0
                client.update()
                obs, _ = env.reset()

        if step > 0 and config.buffer_period > 0 and step % config.buffer_period == 0:
            # dump to pickle file
            buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer")
            if not os.path.exists(buffer_path):
                os.makedirs(buffer_path)
            with open(os.path.join(buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                pkl.dump(transitions, f)
                transitions = []

        timer.tock("total")

        if step % config.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)


##############################################################################

def gather_intermediate_outputs(nested: dict, key_name: str) -> list:
    """
    Recursively search the nested dictionary for all values stored under the key 'dense_outputs'.
    
    Parameters:
      nested (dict): The dictionary to search.
    
    Returns:
      A list of all values found for the key 'dense_outputs' (flattened).
    """
    outputs = []
    for key, value in nested.items():
        if isinstance(value, dict):
            if key_name in value:
                # Append all values stored under 'dense_outputs'
                outputs.extend(value[key_name])
            else:
                outputs.extend(gather_intermediate_outputs(value, key_name))
    return outputs

def compute_dormant_ratio_for_network(
    apply_fn,
    params,
    inputs,
    percentage_threshold=0.025,
    key_name: str = 'dense_outputs'
):
    """
    Calls apply_fn with mutable intermediates and then measures how many neurons are
    'dormant' in each DenseWithLogging layer (i.e. those stored under 'dense_outputs').
    
    Returns the global ratio of (dormant / total).
    """
    # Run the forward pass, capturing intermediates.
    y, mutables = apply_fn({'params': params}, inputs, mutable=['intermediates'], train=False)
    
    # Recursively gather all intermediate values stored under the key 'dense_outputs'
    intermediate_outputs = gather_intermediate_outputs(mutables["intermediates"], key_name)
    
    total_neurons = 0
    dormant_neurons = 0

    for layer_out in intermediate_outputs:
        # layer_out is expected to have shape (batch_size, hidden_dim).
        mean_abs = jnp.abs(layer_out).mean(axis=0)       # shape (hidden_dim,)
        average_across_neurons = mean_abs.mean()
        dormant_mask = mean_abs < (average_across_neurons * percentage_threshold)
        num_dormant = dormant_mask.sum()
        total_neurons_in_layer = layer_out.shape[1]
        dormant_neurons += num_dormant
        total_neurons += total_neurons_in_layer

    ratio = dormant_neurons / (total_neurons + 1e-8)
    return ratio


def compute_all_dormant_ratios(agent, batch, threshold=0.025):
    """
    Computes the dormant neuron ratio for both dense outputs and spatial outputs.
    """

    batch = _unpack(batch)
    observations = batch["observations"]
    next_observations = batch["next_observations"]
    actions = batch["actions"]

    def actor_apply_fn(variables, inputs, mutable, train):
        # Because `agent.forward_policy` normally does “apply_fn({'params':...}, obs)”
        # we replicate that here. But we do the direct call to “agent.state.apply_fn”:
        return agent.state.apply_fn(
            variables,
            next_observations,  # your input
            name="actor",
            train=False,
            mutable=mutable
        )

    actor_dense_ratio = compute_dormant_ratio_for_network(
        actor_apply_fn,
        agent.state.params,     # actor's parameters
        inputs=None,            # inputs captured via closure (if needed)
        percentage_threshold=threshold,
        key_name='dense_outputs'
    )

    # Compute ratio for Spatial outputs.
    actor_spatial_ratio = compute_dormant_ratio_for_network(
        actor_apply_fn,
        agent.state.params,
        inputs=None,
        percentage_threshold=threshold,
        key_name='spatial_outputs'
    )

    def critic_apply_fn(variables, inputs, mutable, train):
        return agent.state.apply_fn(
            variables,
            observations,
            actions[..., :-1],    # or whatever your code uses for continuous part
            name="critic",
            train=False,
            mutable=mutable
        )

    critic_dense_ratio = compute_dormant_ratio_for_network(
        critic_apply_fn,
        agent.state.params,
        inputs=None,
        percentage_threshold=threshold,
        key_name="dense_outputs"
    )

    critic_spatial_ratio = compute_dormant_ratio_for_network(
        critic_apply_fn,
        agent.state.params,
        inputs=None,
        percentage_threshold=threshold,
        key_name="spatial_outputs"
    )

    def grasp_critic_apply_fn(variables, inputs, mutable, train):
        return agent.state.apply_fn(
            variables,
            observations,
            name="grasp_critic",
            train=train,
            mutable=mutable
        )

    grasp_dense_ratio = compute_dormant_ratio_for_network(
        grasp_critic_apply_fn,
        agent.state.params,
        inputs=None,
        percentage_threshold=threshold,
        key_name="dense_outputs"
    )

    grasp_spatial_ratio = compute_dormant_ratio_for_network(
        grasp_critic_apply_fn,
        agent.state.params,
        inputs=None,
        percentage_threshold=threshold,
        key_name="spatial_outputs"
    )

    return {
        "dormant_ratio_actor_dense": actor_dense_ratio,
        "dormant_ratio_actor_spatial": actor_spatial_ratio,
        "dormant_ratio_critic_dense": critic_dense_ratio,
        "dormant_ratio_critic_spatial": critic_spatial_ratio,
        "dormant_ratio_grasp_critic_dense": grasp_dense_ratio,
        "dormant_ratio_grasp_critic_spatial": grasp_spatial_ratio,
    }


def learner(rng, agent, replay_buffer, wandb_logger=None):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    start_step = (
        int(os.path.basename(checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path)))[11:])
        + 1
        if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path)
        else 0
    )
    step = start_step

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=step)
        return {}  # not expecting a response

    # Create server
    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.start(threaded=True)

    # Loop to wait until replay_buffer is filled
    pbar = tqdm.tqdm(
        total=config.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < config.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
    pbar.close()

    # send the initial network to the actor
    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )

    # wait till the replay buffer is filled with enough data
    timer = Timer()
    
    if isinstance(agent, SACAgent):
        train_critic_networks_to_update = frozenset({"critic"})
        train_networks_to_update = frozenset({"critic", "actor", "temperature"})
    else:
        train_critic_networks_to_update = frozenset({"critic", "grasp_critic"})
        train_networks_to_update = frozenset({"critic", "grasp_critic", "actor", "temperature"})

    for step in tqdm.tqdm(
        range(start_step, config.max_steps), dynamic_ncols=True, desc="learner"
    ):
        # run n-1 critic updates and 1 critic + actor update.
        # This makes training on GPU faster by reducing the large batch transfer time from CPU to GPU
        for critic_step in range(config.cta_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)

            with timer.context("train_critics"):
                agent, critics_info = agent.update(
                    batch,
                    networks_to_update=train_critic_networks_to_update,
                )

        with timer.context("train"):
            batch = next(replay_iterator)
            agent, update_info = agent.update(
                batch,
                networks_to_update=train_networks_to_update,
            )
        # publish the updated network
        if step > 0 and step % (config.steps_per_update) == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        if step % config.steps_per_log_dormant == 0 and wandb_logger:
                dormant_ratios = compute_all_dormant_ratios(agent, batch, threshold=0.025)
                wandb_logger.log(dormant_ratios, step=step)

        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)

        if (
            step > 0
            and config.checkpoint_period
            and step % config.checkpoint_period == 0
        ):
            print_green(f"Saving checkpoint at step {step}")
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path), agent.state, step=step, keep=100
            )


##############################################################################


def main(_):
    global config
    config = CONFIG_MAPPING[FLAGS.exp_name]()

    assert config.batch_size % num_devices == 0
    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    env = config.get_environment(
        fake_env=FLAGS.learner,
        save_video=FLAGS.save_video,
        classifier=False,
    )
    env = RecordEpisodeStatistics(env)

    rng, sampling_rng = jax.random.split(rng)
    
    if config.setup_mode == 'single-arm-fixed-gripper' or config.setup_mode == 'dual-arm-fixed-gripper':   
        agent: SACAgent = make_sac_pixel_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = False
    elif config.setup_mode == 'single-arm-learned-gripper':
        agent: SACAgentHybridSingleArm = make_sac_pixel_agent_hybrid_single_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True
    elif config.setup_mode == 'dual-arm-learned-gripper':
        agent: SACAgentHybridDualArm = make_sac_pixel_agent_hybrid_dual_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True
    else:
        raise NotImplementedError(f"Unknown setup mode: {config.setup_mode}")

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent = jax.device_put(
        jax.tree_map(jnp.array, agent), sharding.replicate()
    )

    if FLAGS.checkpoint_path is not None and os.path.exists(FLAGS.checkpoint_path):
        input("Checkpoint path already exists. Press Enter to resume training.")
        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path),
            agent.state,
        )
        agent = agent.replace(state=ckpt)
        ckpt_number = os.path.basename(
            checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        )[11:]
        print_green(f"Loaded previous checkpoint at step {ckpt_number}.")

    def create_replay_buffer_and_wandb_logger():
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )
        # set up wandb and logging
        wandb_logger = make_wandb_logger(
            project="hil-serl",
            description=FLAGS.exp_name,
            debug=FLAGS.debug,
        )
        return replay_buffer, wandb_logger

    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        replay_buffer, wandb_logger = create_replay_buffer_and_wandb_logger()

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "buffer")
        ):
            for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        replay_buffer.insert(transition)
            print_green(
                f"Loaded previous buffer data. Replay buffer size: {len(replay_buffer)}"
            )

        # learner loop
        print_green("starting learner loop")
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            wandb_logger=wandb_logger,
        )

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(50000)  # the queue size on the actor

        # actor loop
        print_green("starting actor loop")
        actor(
            agent,
            data_store,
            env,
            sampling_rng,
        )

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)
