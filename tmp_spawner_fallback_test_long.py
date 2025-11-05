from environment import MASAEnv

class A:
    pass
args = A()
setattr(args, 'initial_jobs', 0)
setattr(args, 'n_agents', 5)
setattr(args, 'history_dir', 'my_data_and_graph/historydata')

env = MASAEnv(args=args, auto_build=True, auto_start_arrivals=False)
# force no TaskGenerator so fallback synthesis is used
env.job_generator = None
env.reset()
# run simulation for 2000 simulated seconds
env.env.run(until=2000)
print('jobs after run (fallback,long):', len(env.jobs))
for j in env.jobs:
    print('job', j.id, 'arrival', float(getattr(j, 'arrival_time', -1)))
