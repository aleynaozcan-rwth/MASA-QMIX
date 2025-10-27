from MARL.common.arguments import get_common_args
from environment import MASAEnv
from MARL.runner import Runner

# Quick probe runner: longer than run_short but still moderate

def main():
    args = get_common_args()
    args.n_epoch = 200
    args.n_episodes = 16
    args.evaluate_cycle = 1
    args.evaluate_epoch = 2
    args.episode_limit = 200
    args.learn = True
    args.buffer_size = 20000
    args.batch_size = 64
    args.train_steps = 200
    args.save_cycle = 500
    args.cuda = False
    args.seed = 0
    args.gantt_csv = False
    args.gantt_snapshot_every = 0

    env = MASAEnv()
    runner = Runner(env, args)
    runner.run(0)

if __name__ == "__main__":
    main()
