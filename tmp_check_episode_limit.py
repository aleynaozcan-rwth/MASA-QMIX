from environment import MASAEnv

# no args provided -> defaults should apply
env = MASAEnv(auto_build=False)
env.reset()
print('env.episode_limit=', env.episode_limit)
