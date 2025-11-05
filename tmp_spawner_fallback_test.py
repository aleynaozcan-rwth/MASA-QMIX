from environment import MASAEnv

# lightweight args namespace
class A:
    pass
args = A()
setattr(args, 'initial_jobs', 0)
setattr(args, 'n_agents', 5)
setattr(args, 'history_dir', 'my_data_and_graph/historydata')

# Create env but force job_generator to None to exercise fallback path
env = MASAEnv(args=args, auto_build=True, auto_start_arrivals=False)
# force no TaskGenerator so fallback synthesis is used
env.job_generator = None
# reset() will start the dynamic arrival loop
env.reset()
# run simulation for 120 simulated seconds
env.env.run(until=120)
print('jobs after run (fallback):', len(env.jobs))
for j in env.jobs:
    print('job', j.id, 'arrival', float(getattr(j, 'arrival_time', -1)))
