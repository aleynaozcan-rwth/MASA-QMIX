from environment import MASAEnv

# lightweight args namespace
class A:
    pass
args = A()
setattr(args, 'initial_jobs', 2)
setattr(args, 'n_agents', 5)
setattr(args, 'history_dir', 'my_data_and_graph/historydata')

env = MASAEnv(args=args, auto_build=True, auto_start_arrivals=False)
# reset() will start the dynamic arrival loop
env.reset()
# run simulation for 120 simulated seconds
env.env.run(until=120)
print('jobs after run:', len(env.jobs))
for j in env.jobs:
    print('job', j.id, 'arrival', float(getattr(j, 'arrival_time', -1)))
