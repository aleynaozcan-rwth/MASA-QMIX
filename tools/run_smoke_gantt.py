import os
from types import SimpleNamespace

from environment import MASAEnv
from MARL.runner import Runner

# minimal args for Runner
args = SimpleNamespace()
args.result_dir = './result'
args.alg = 'qmix'
args.map = 'masa_schedule'
args.buffer_size = 10
args.seed = 42
args.batch_size = 4
args.n_epoch = 2
args.evaluate_cycle = 1
args.n_episodes = 1
args.evaluate_epoch = 1
args.gantt_csv = True
args.gantt_snapshot_every = 0
args.snapshot_on_eval = True
args.use_machine_actions = True
args.use_granular_actions = False
args.enable_logs = False
args.clean_history = False
args.n_steps = 100
args.state_shape = 64
args.obs_shape = 11
args.cuda = False
args.last_action = False
args.reuse_network = False
args.lr = 0.0005
args.gamma = 0.99
args.grad_norm_clip = 10.0
args.target_update_cycle = 200
args.train_steps = 10
args.save_cycle = 100
args.rnn_hidden_dim = 64
args.epsilon = 0.0
# start

env = MASAEnv(num_jobs=5, num_operators=2, episode_limit=50)
runner = Runner(env, args)
# run a quick training loop (small) which will trigger evaluation at epoch 1
runner.run(0)
print('Smoke run finished. Check my_data_and_graph/historydata for gantt files.')
