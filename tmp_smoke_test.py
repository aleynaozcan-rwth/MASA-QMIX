from environment import MASAEnv
import os

# simple smoke test: capacity=2, 5 jobs
env = MASAEnv(args=None)
env.max_active_agents = 2
# single-op jobs of duration 1.0 on machine 0
ops = [(0, [0], {0: 1.0})]
for i in range(5):
    env.add_job(ops, start_immediately=True)

# run simulation until all jobs finish or time limit
env.env.run(until=50)
print("Done", "Completed:", env.completed_jobs, "Active:", len(env.active_agents), "Pending:", len(env.pending_jobs))

path = os.path.join('my_data_and_graph', 'historydata', 'scheduling_timeline.txt')
try:
    with open(path, 'r', encoding='utf-8') as f:
        print('=== TIMELINE ===')
        print(f.read())
except Exception as e:
    print('No timeline file or failed to read:', e)
