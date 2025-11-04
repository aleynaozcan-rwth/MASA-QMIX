import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from MARL.common.arguments import get_common_args
from main import marl_agent_wrapper
import copy

# Build args and override for a fast debug smoke
# Use a local run_args copy to avoid mutating the canonical args object
base_args = get_common_args()
run_args = copy.deepcopy(base_args)
run_args.n_epoch = 1
run_args.n_episodes = 1
run_args.episode_limit = 10
run_args.train_steps = 1
run_args.batch_size = 2
run_args.buffer_size = 200
run_args.min_warmup_size = 1
run_args.learn = True
# Force CPU (do NOT pass literal strings to argparse bool flags)
run_args.cuda = False
# Enable debug assertions in RolloutWorker
setattr(run_args, 'debug_assert_shapes', True)
# Be quiet on env prints
run_args.quiet_env = True

# Provide config path and enable auto_load_config so MASAEnv will populate self.config
run_args.config_path = 'configs/env_config_enabled.yaml'
run_args.auto_load_config = True

# Run using the local run_args copy
marl_agent_wrapper(run_args)
