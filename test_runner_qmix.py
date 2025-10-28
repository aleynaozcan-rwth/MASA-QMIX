from environment import MASAEnv
from MARL.runner import Runner
class A: pass
env = MASAEnv(num_jobs=2, num_operators=2, num_wcs=3, episode_limit=30)
args = A()
args.alg='qmix'
args.buffer_size=16
args.seed=1
args.result_dir='./result'
args.map='test'
args.batch_size=2
args.n_epoch=1
args.evaluate_cycle=1
args.n_episodes=1
args.learn=False
args.n_agents=1
args.n_actions=None
args.quiet_env=True
args.use_granular_actions=True
# policy args
args.cuda=False
args.last_action=False
args.reuse_network=False
args.load_model=False
args.model_dir='./MARL/model'
args.optimizer='RMS'
args.lr=1e-3
args.grad_norm_clip=10
args.target_update_cycle=200
args.rnn_hidden_dim=64
args.save_cycle=100
# env dims
args.state_shape = env.state_dim
args.obs_shape = env.obs_dim_agent
args.episode_limit = env.episode_limit

r = Runner(env, args)
# Run 1 training loop (this will run episodes but skip learning because learn=False)
r.run(0)
print('Runner completed')
