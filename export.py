
import gymnasium as gym
import numpy as np
import rl_zoo3
import rl_zoo3.enjoy
from rl_zoo3.enjoy import enjoy
from sbx import DDPG, DQN, PPO, SAC, TD3, TQC, CrossQ
import argparse
import importlib
import os
import sys

import numpy as np
import torch as th
import yaml
from huggingface_sb3 import EnvironmentName
from stable_baselines3.common.utils import set_random_seed

import rl_zoo3.import_envs  # noqa: F401 pylint: disable=unused-import
from rl_zoo3 import ALGOS, create_test_env, get_saved_hyperparams
from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.load_from_hub import download_from_hub
from rl_zoo3.utils import StoreDict, get_model_path

import frasa_env

import os, yaml, importlib

gym.register_envs(frasa_env)

rl_zoo3.ALGOS["ddpg"] = DDPG
rl_zoo3.ALGOS["dqn"] = DQN
# See SBX readme to use DroQ configuration
# rl_zoo3.ALGOS["droq"] = DroQ
rl_zoo3.ALGOS["sac"] = SAC
rl_zoo3.ALGOS["ppo"] = PPO
rl_zoo3.ALGOS["td3"] = TD3
rl_zoo3.ALGOS["tqc"] = TQC
rl_zoo3.ALGOS["crossq"] = CrossQ
rl_zoo3.enjoy.ALGOS = rl_zoo3.ALGOS
rl_zoo3.exp_manager.ALGOS = rl_zoo3.ALGOS

def enjoy3() -> None:  # noqa: C901
    # 2. Create or load your CrossQ model
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=EnvironmentName, default="frasa-standup-v0")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="logs/")
    parser.add_argument("--algo", help="RL Algorithm", default="crossq", type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=1000, type=int)
    parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
    parser.add_argument("--n-envs", help="number of environments", default=1, type=int)
    parser.add_argument("--exp-id", help="Experiment ID (default: 0: latest, -1: no exp folder)", default=0, type=int)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument(
        "--no-render", action="store_true", default=False, help="Do not render the environment (useful for tests)"
    )
    parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions")
    parser.add_argument("--device", help="PyTorch device to be use (ex: cpu, cuda...)", default="auto", type=str)
    parser.add_argument(
        "--load-best", action="store_true", default=True, help="Load best model instead of last model if available"
    )
    parser.add_argument(
        "--load-checkpoint",
        type=int,
        help="Load checkpoint instead of last model if available, "
        "you must pass the number of timesteps corresponding to it",
    )
    parser.add_argument(
        "--load-last-checkpoint",
        action="store_true",
        default=False,
        help="Load last checkpoint instead of last model if available",
    )
    parser.add_argument("--stochastic", action="store_true", default=False, help="Use stochastic actions")
    parser.add_argument(
        "--norm-reward", action="store_true", default=False, help="Normalize reward if applicable (trained with VecNormalize)"
    )
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument("--reward-log", help="Where to log reward", default="", type=str)
    parser.add_argument(
        "--gym-packages",
        type=str,
        nargs="+",
        default=["frasa_env"],
        help="Additional external Gym environment package modules to import",
    )
    parser.add_argument(
        "--env-kwargs", type=str, nargs="+", action=StoreDict, help="Optional keyword argument to pass to the env constructor"
    )
    parser.add_argument(
        "--custom-objects", action="store_true", default=False, help="Use custom objects to solve loading issues"
    )
    parser.add_argument(
        "-P",
        "--progress",
        action="store_true",
        default=False,
        help="if toggled, display a progress bar using tqdm and rich",
    )
    args = parser.parse_args()

    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_name: EnvironmentName = args.env
    algo = args.algo
    folder = args.folder

    try:
        _, model_path, log_path = get_model_path(
            args.exp_id,
            folder,
            algo,
            env_name,
            args.load_best,
            args.load_checkpoint,
            args.load_last_checkpoint,
        )
    except (AssertionError, ValueError) as e:
        # Special case for rl-trained agents
        # auto-download from the hub
        if "rl-trained-agents" not in folder:
            raise e
        else:
            print("Pretrained model not found, trying to download it from sb3 Huggingface hub: https://huggingface.co/sb3")
            # Auto-download
            download_from_hub(
                algo=algo,
                env_name=env_name,
                exp_id=args.exp_id,
                folder=folder,
                organization="sb3",
                repo_name=None,
                force=False,
            )
            # Try again
            _, model_path, log_path = get_model_path(
                args.exp_id,
                folder,
                algo,
                env_name,
                args.load_best,
                args.load_checkpoint,
                args.load_last_checkpoint,
            )

    print(f"Loading {model_path}")

    # Off-policy algorithm only support one env for now
    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]

    set_random_seed(args.seed)

    if args.num_threads > 0:
        if args.verbose > 1:
            print(f"Setting torch.num_threads to {args.num_threads}")
        th.set_num_threads(args.num_threads)

    is_atari = ExperimentManager.is_atari(env_name.gym_id)
    is_minigrid = ExperimentManager.is_minigrid(env_name.gym_id)

    stats_path = os.path.join(log_path, env_name)
    hyperparams, maybe_stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)

    # load env_kwargs if existing
    env_kwargs = {}
    args_path = os.path.join(log_path, env_name, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path) as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]
    # overwrite with command line arguments
    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)

    log_dir = args.reward_log if args.reward_log != "" else None

    env = create_test_env(
        env_name.gym_id,
        n_envs=args.n_envs,
        stats_path=maybe_stats_path,
        seed=args.seed,
        log_dir=log_dir,
        should_render= not args.no_render,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    ) # use your real env ID from unified-humanoid-getup
    # model = CrossQ("MlpPolicy", env)     # or load a pretrained one:
    kwargs = dict(seed=args.seed)
    if algo in off_policy_algos:
        # Dummy buffer size as we don't need memory to enjoy the trained agent
        kwargs.update(dict(buffer_size=1))
        # Hack due to breaking change in v1.6
        # handle_timeout_termination cannot be at the same time
        # with optimize_memory_usage
        if "optimize_memory_usage" in hyperparams:
            kwargs.update(optimize_memory_usage=False)

    # Check if we are running python 3.8+
    # we need to patch saved model under python 3.6/3.7 to load them
    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

    custom_objects = {}
    if newer_python_version or args.custom_objects:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
            # load models with different obs bounds
            # Note: doesn't work with channel last envs
            # "observation_space": env.observation_space,
        }

    if "HerReplayBuffer" in hyperparams.get("replay_buffer_class", ""):
        kwargs["env"] = env

    model = ALGOS[algo].load(model_path, custom_objects=custom_objects, device=args.device, **kwargs)
    obs_shape = env.observation_space.shape  # ← define this before using obs_shape
    actor_state = model.policy.actor_state
    obs_dim = actor_state.params["Dense_0"]["kernel"].shape[0]


    import jax
    import jax.numpy as jnp
    from jax.experimental import jax2tf

    import tensorflow as tf
    import tf2onnx

    from sbx import  CrossQ  # assume you trained with sbx.CrossQ
    from sbx.crossq.policies import CrossQPolicy

    def inference_fn(observations: jnp.ndarray) -> jnp.ndarray:
        """
        observations: [batch, obs_dim]
        returns:      [batch, act_dim] (greedy CrossQ action)
        """
        return CrossQPolicy.select_action(actor_state, observations)

    # 1a. Create a random batch of observations
    rng = jax.random.PRNGKey(0)
    batch_size = 1
    obs_batch = np.random.randn(batch_size, obs_dim).astype(np.float32)  # batch size 4

    # 1b. Run through your JAX inference_fn
    jax_out = inference_fn(jnp.array(obs_batch))
    jax_out = np.array(jax_out)  # convert back to NumPy for easy comparison

    print("JAX output:\n", jax_out)

    inference_tf = jax2tf.convert(inference_fn, enable_xla=False)
    inference_tf = tf.function(inference_tf, autograph=False)
    tf_out = inference_tf(obs_batch)  # TF will accept a NumPy array
    tf_out = tf_out.numpy()  # convert to NumPy

    print("TF output:\n", tf_out)

    # 2b. Quick numeric check
    diff = np.max(np.abs(jax_out - tf_out))
    print(f"Max abs difference JAX vs TF: {diff:.3e}")
    # assert diff < 1e-5, "JAX→TF conversion drift is too large!"

    #
    import tf2onnx


    # inference_onnx = tf2onnx.convert.from_function(inference_tf, input_signature=[tf.TensorSpec([1, 1])])
    input_spec = tf.TensorSpec([batch_size, obs_dim], tf.float32, name="observations")

    model_proto, external_tensor_storage = tf2onnx.convert.from_function(
        inference_tf,
        input_signature=[input_spec],
        opset=17,  # adjust if needed
        output_path="crossq_live.onnx"  # writes the file for you
    )
    import onnxruntime as rt

    # 3a. Load the exported ONNX file
    sess = rt.InferenceSession("crossq_live.onnx", providers=["CPUExecutionProvider"])

    # 3b. Run inference
    inputs = {"observations": obs_batch}
    onnx_out = sess.run(None, inputs)[0]  # grab first (and only) output

    print("ONNX output:\n", onnx_out)

    # 3c. Compare to the JAX baseline
    diff2 = np.max(np.abs(jax_out - onnx_out))
    print(f"Max abs difference JAX vs ONNX: {diff2:.3e}")
    # assert diff2 < 1e-4, "JAX→ONNX drift is too large!"



if __name__ == "__main__":
    enjoy3()