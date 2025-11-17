import os
from MARL.common.arguments import get_smoke_args
from environment import MASAEnv
from MARL.runner import Runner

# Start with the centralized smoke-friendly args and apply only minimal, explicit overrides.
# Tool-specific minimal overrides should live in MARL.common.arguments.get_smoke_args().
# We avoid mutating the shared args object here to follow the strict no-mutate rule.

if __name__ == '__main__':
	# Safe override for local testing only
	args = get_smoke_args()

	# start environment (keep explicit env constructor args local to the tool)
	env = MASAEnv(num_jobs=5, num_operators=2, episode_limit=50)
	runner = Runner(env, args)
	# run a quick training loop (small) which will trigger evaluation at epoch 1
	runner.run(0)
	print('Smoke run finished. Check my_data_and_graph/historydata for gantt files.')
