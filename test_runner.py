from environment import MASAEnv
from MARL.runner import Runner
class A: pass
# create env first so args can borrow dims
env = MASAEnv(num_jobs=0, num_operators=2, num_wcs=3, episode_limit=50)
args = A()
args.alg='qmix'
args.buffer_size=10
args.seed=1
args.result_dir='./result'
args.map='test'
args.batch_size=4
args.n_epoch=1
args.evaluate_cycle=1
args.n_episodes=1
args.learn=False
args.n_agents=1
args.n_actions=None
args.quiet_env=True
args.use_granular_actions=True
# agent shapes
args.state_shape = env.state_dim
args.obs_shape = env.obs_dim_agent

r = Runner(env, args)
print('Runner.args.n_actions ->', getattr(r.args,'n_actions', None))
