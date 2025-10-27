from MARL.common.arguments import get_common_args
from environment import MASAEnv
from MARL.runner import Runner

def main():
    args = get_common_args()
    args.n_epoch = 50
    args.n_episodes = 8
    args.evaluate_cycle = 5
    args.evaluate_epoch = 2
    args.episode_limit = 200
    args.learn = True
    args.buffer_size = 10000
    args.batch_size = 32
    args.train_steps = 50
    args.save_cycle = 500
    args.cuda = False
    args.seed = 0

    env = MASAEnv()
    runner = Runner(env, args)
    runner.run(0)

if __name__ == "__main__":
    main()
