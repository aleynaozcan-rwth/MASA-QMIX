# ...existing code...
# arguments.py – MASA-QMIX Step 8A.7 (Clean Fixed)
# -------------------------------------------------
# True learning configuration for stable QMIX training
# Compatible with: MASAEnv (6D obs, 64D state, 18 WorkCenters, 200-step episodes)
# -------------------------------------------------

import argparse
import logging
import copy
from types import SimpleNamespace


# (Preset system removed — defaults are set directly on parser arguments)


class ReadOnlyArgs:
    """Lightweight read-only wrapper around an argparse.Namespace.

    Attempts to set attributes raise AttributeError. Provides `as_mutable()`
    which returns a deep-copied mutable namespace, and a custom
    __deepcopy__ so callers using copy.deepcopy(...) get a mutable copy.
    """

    def __init__(self, ns):
        # store a shallow copy to avoid external aliasing
        object.__setattr__(self, "__data__", copy.deepcopy(ns))

    def __getattr__(self, item):
        data = object.__getattribute__(self, "__data__")
        try:
            return getattr(data, item)
        except AttributeError:
            raise

    def __setattr__(self, key, value):
        raise AttributeError("ReadOnlyArgs does not allow attribute assignment")

    def __repr__(self):
        return f"ReadOnlyArgs({repr(object.__getattribute__(self, '__data__'))})"

    def as_mutable(self):
        """Return a deep-copied, mutable argparse.Namespace of the data."""
        data = object.__getattribute__(self, "__data__")
        return copy.deepcopy(data)

    def __deepcopy__(self, memo):
        # Return a plain, deep-copied Namespace so callers that deepcopy a
        # ReadOnlyArgs receive a mutable object they can safely modify.
        return self.as_mutable()


def get_mutable_args():
    parser = argparse.ArgumentParser()

    # ============================================================
    # === Environment settings ===================================
    # ============================================================
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--mode', type=str, choices=['marl', 'random'], default='marl')
    parser.add_argument('--alg', type=str, default='qmix')
    parser.add_argument('--gamma', type=float, default=0.98)
    parser.add_argument('--optimizer', type=str, default='RMS')
    parser.add_argument('--evaluate_epoch', type=int, default=5)
    parser.add_argument('--model_dir', type=str, default='./MARL/model')
    parser.add_argument('--result_dir', type=str, default='./result')
    parser.add_argument('--load_model', type=bool, default=False)
    parser.add_argument('--learn', type=bool, default=True)
    parser.add_argument('--cuda', type=bool, default=False)

    # ============================================================
    # === MASA-QMIX environment parameters =======================
    # ============================================================
    parser.add_argument('--arrival_lambda', type=float, default=0.2,
                        help='Average job arrival rate λ (jobs per simulation time unit)')
    parser.add_argument('--num_operators', type=int, default=4)
    parser.add_argument('--job_min_ops', type=int, default=2,
                        help='Minimum number of operations per job (default 2)')
    parser.add_argument('--job_max_ops', type=int, default=4,
                        help='Maximum number of operations per job (default 4)')
    # Reward shaping defaults (centralized single source-of-truth)
    # === Hybrid Reward Parameters ===
    parser.add_argument('--reward_w1_completed', type=float, default=1.0,
                        help='Weight for CompletedNorm (K1)')
    parser.add_argument('--reward_w2_avgwait', type=float, default=0.6,
                        help='Weight for AvgWait (K2)')
    parser.add_argument('--reward_w3_wip', type=float, default=0.3,
                        help='Weight for Work-in-Progress (K3)')
    parser.add_argument('--reward_w4_throughput_delta', type=float, default=0.8,
                        help='Weight for ThroughputDelta (K4)')
    parser.add_argument('--reward_w5_load_variance', type=float, default=0.4,
                        help='Weight for LoadVariance (K5)')

    parser.add_argument('--reward_a1_completion', type=float, default=1.0,
                        help='Local reward weight for completed operation (a1)')
    parser.add_argument('--reward_a2_wait', type=float, default=0.5,
                        help='Local reward weight for waiting penalty (a2)')
    parser.add_argument('--reward_a3_infeasible', type=float, default=0.25,
                        help='Local reward weight for infeasible action penalty (a3)')

    parser.add_argument('--reward_alpha_mix', type=float, default=0.7,
                        help='Mixing coefficient between global and local rewards (alpha)')
    parser.add_argument('--reward_lambda_m', type=float, default=0.8,
                        help='Weight for machine utilization variance (lambda_m)')
    parser.add_argument('--reward_lambda_o', type=float, default=0.2,
                        help='Weight for operator utilization variance (lambda_o)')

    parser.add_argument('--reward_log_components', action='store_true', default=False,
                        help='If set, logs detailed reward component breakdowns during rollout.')

    # ============================================================
    # === Episode / agent configuration ==========================
    # ============================================================
    parser.add_argument('--episode_limit', type=int, default=500)
    parser.add_argument('--n_agents', type=int, default=10)
    parser.add_argument('--initial_jobs', type=int, default=4,
                        help='Number of jobs created at the start of the simulation (default 4)')
    # Training loop sizes (production defaults, CLI overrideable)
    parser.add_argument('--n_epoch', type=int, default=400,
                        help='Number of training epochs (default 400 for stable QMIX learning)')
    parser.add_argument('--n_episodes', type=int, default=4,
                        help='Episodes per epoch (default 4)')
    parser.add_argument('--evaluate_cycle', type=int, default=2,
                        help='Evaluate every N epochs')
    parser.add_argument('--n_actions', type=int, default=5,
                        help='(fallback) number of actions/workcenters when machine_list is not provided')
    parser.add_argument('--state_shape', type=int, default=64)
    parser.add_argument('--obs_shape', type=int, default=6)
    

    # ============================================================
    # === Replay buffer & training settings ======================
    # ============================================================
    # Reduced defaults so warm-up completes faster but training stays stable
    parser.add_argument('--buffer_size', type=int, default=2500)   # was 3000
    parser.add_argument('--batch_size', type=int, default=32)      # was 32
    parser.add_argument('--train_steps', type=int, default=30)     # was 10
    parser.add_argument('--min_warmup_size', type=int, default=800)  # new: minimum samples before strict warm-up
    parser.add_argument('--target_update_cycle', type=int, default=20)  # ✅ frequent sync
    parser.add_argument('--grad_norm_clip', type=float, default=10.0)

    # ============================================================
    # === Learning & optimization ================================
    # ============================================================
    parser.add_argument('--lr', type=float, default=2e-4)         # ✅ stable learning

    # ============================================================
    # === Exploration (epsilon schedule) =========================
    # ============================================================
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--epsilon_end', type=float, default=0.05)
    parser.add_argument('--epsilon_anneal_steps', type=int, default=30000,
                        help='Number of steps to linearly anneal epsilon from start to end')

    # ============================================================
    # === Logging / visualization ================================
    # ============================================================
    parser.add_argument('--enable_logs', type=bool, default=True)
    parser.add_argument('--quiet_env', action='store_true', default=False,
                        help='Suppress verbose environment debug prints (wait_for_decisions, etc.)')

    # Optional preset selector (e.g. --preset fast). If provided, apply_preset_if_requested
    # will mutate the args accordingly after parsing.
    # --preset removed; defaults are set directly in this file.

    # Use parse_known_args to avoid failing when external tooling (pytest)
    # injects unknown CLI flags during test collection.
    args, _unknown = parser.parse_known_args()
    # If user provided a CLI n_actions that differs from eventual machine_list,
    # the environment will align to the actual machine list length. Warn if
    # they used a commonly-mistaken default like 18 to avoid confusion.
    try:
        if getattr(args, 'n_actions', None) == 18:
            try:
                logging.getLogger(__name__).warning('[WARN] CLI --n_actions=18 detected. MASAEnv will override n_actions to match actual machine_list length if available.')
            except Exception:
                pass
    except Exception:
        pass
    
    # Runtime-only defaults (not defined in parser)
    args.rnn_hidden_dim   = getattr(args, "rnn_hidden_dim", 64)
    args.mix_embed_dim    = getattr(args, "mix_embed_dim", 32)
    args.two_hyper_layers = getattr(args, "two_hyper_layers", False)
    args.history_dir      = getattr(args, "history_dir", "./my_data_and_graph/historydata")
    args.save_cycle       = getattr(args, "save_cycle", 500)

    # Safe debug dump: write final active args to historydata/args_dump.txt
    # This is best-effort and must not raise; it helps debugging CLI/preset
    # interactions without changing runtime behavior. Dump at the end so it
    # always records the final active args.
    try:
        import os
        history_dir = getattr(args, 'history_dir', './my_data_and_graph/historydata')
        # Ensure the history directory exists before attempting to open the dump file
        os.makedirs(history_dir, exist_ok=True)
        out_path = os.path.join(history_dir, 'args_dump.txt')
        with open(out_path, 'w', encoding='utf-8') as fh:
            fh.write('[DEBUG] Final active arguments before training:\n')
            for key in sorted(vars(args)):
                try:
                    val = getattr(args, key)
                except Exception:
                    val = '<unreadable>'
                fh.write(f"{key} = {val!r}\n")
        try:
            print(f"[INFO] Active arguments written to {out_path}")
            print("[DEBUG_DUMP] args_dump.txt successfully written.")
        except Exception:
            # printing should never crash; ignore if stdout is unavailable
            pass
    except Exception:
        # best-effort only; do not let debug I/O affect normal execution
        pass
    return args


def get_common_args():
    """Return a read-only canonical args namespace.

    Use `get_mutable_args()` when you need to mutate fields (tests, tools,
    smoke harness). The returned object forbids attribute assignment.
    """
    return ReadOnlyArgs(get_mutable_args())


# ===============================================================
# === Algorithm-specific argument groups ========================
# ===============================================================

def _ensure_mutable(args):
    """If args is ReadOnlyArgs, return a mutable deep-copy; otherwise
    return args as-is (assumed mutable)."""
    if isinstance(args, ReadOnlyArgs):
        return args.as_mutable()
    return args


def get_mixer_args(args):
    """QMIX-specific hyperparameters for MASA-QMIX learning stability.
    
    Only sets QMIX-specific parameters. General training parameters 
    (n_epoch, n_episodes, lr, buffer_size, etc.) are already defined 
    in parser with production defaults.
    """
    # Work on a deep-copied mutable namespace so callers' originals are never changed.
    args = copy.deepcopy(_ensure_mutable(args))
    
    # QMIX-specific network architecture
    args.rnn_hidden_dim   = 64
    args.qmix_hidden_dim  = 32
    args.two_hyper_layers = True
    args.hyper_hidden_dim = 64
    args.qtran_hidden_dim = 64
    args.mix_embed_dim    = 32

    # QMIX-specific algorithm parameters
    args.lambda_opt = 1
    args.lambda_nopt = 1
    
    return args


def get_smoke_args():
    """Return a compact, deterministic argument namespace suitable for
    smoke tests and CI: small buffers, few episodes, no GPU, reproducible seed.
    This is a convenience wrapper used by tools/tests to reduce repeated
    local overrides across scripts and test files.
    """
    # NOTE: This helper returns a modified copy for smoke/test runs; it does not mutate CLI args.
    # Return a fresh, mutable configuration for smoke tests (work on its copy).
    args = copy.deepcopy(get_mutable_args())
    # small, fast defaults for smoke runs
    args.n_epoch = 1
    args.n_episodes = 3
    args.evaluate_cycle = 10
    args.evaluate_epoch = 1
    args.episode_limit = getattr(args, 'episode_limit', 600)
    args.learn = False
    args.buffer_size = 10
    args.batch_size = 4
    args.train_steps = 1
    args.save_cycle = 500
    args.cuda = False
    args.seed = 0
    args.use_machine_actions = False
    args.use_granular_actions = False
    return args