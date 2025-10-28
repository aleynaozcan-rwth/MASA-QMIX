from types import SimpleNamespace
from MARL.runner import Runner
from environment import MASAEnv

args = SimpleNamespace(
    alg="qmix",
    map="masa",
    result_dir="results",
    buffer_size=1000,
    seed=42,
    learn=True,
    use_granular_actions=False,
    n_agents=10,
    n_epoch=1,
    n_episodes=1,
    evaluate_cycle=10,
    gantt_csv=False,
    clean_history=False,
    enable_logs=False,
    quiet_env=True,
    batch_size=32,
    min_warmup_size=32,
)

env = MASAEnv(num_jobs=10, num_operators=2, num_wcs=3)
# instantiate Runner which writes initial_jobs.txt in __init__
_r = Runner(env, args)
print('Wrote initial_jobs.txt via Runner init to ./my_data_and_graph/historydata/initial_jobs.txt')
with open('./my_data_and_graph/historydata/initial_jobs.txt','r') as hf:
    print(hf.read())
