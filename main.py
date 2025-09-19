
import numpy as np
import pickle
from environment import ScheduleEnv
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
from MARL.runner import Runner
from MARL.common.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, \
    get_reinforce_args, \
    get_commnet_args, get_g2anet_args
from utils.PDRs.shortestDistence import SDrules

np.random.seed(2)


# Difference between game_scores and rolling_scores:
# - game_scores: reward over the entire episode
# - rolling_scores: reward over a fixed-length "trial" (n steps)
# Neither of these is used for training; a "trial" is a user-defined window.

# RL decision wrapper that wires a DRL multi-agent into the environment

def marl_agent_wrapper():
    reset_files = [
        "accumulated_rewards.txt",
        "times.txt",
        "havealook.txt",
        "loss.txt",
        "scheduleresults.txt"
    ]
    for fname in reset_files:
        open(f"./my_data_and_graph/historydata/{fname}", "w").close()  # tamamen boş dosya oluştur

    # Reset CSV with header
    with open("./my_data_and_graph/historydata/rewards_log.csv", "w") as f:
        f.write("episode,reward,time\n")

   #with open("./my_data_and_graph/historydata/accumulated_rewards.txt", "w") as f:
   #    print("----", file=f)
   #with open("my_data_and_graph/times.txt", "w") as f:
   #    print("----", file=f)
   #with open("./my_data_and_graph/historydata/havealook.txt", "w") as f:
   #    pass
   #with open("./my_data_and_graph/historydata/loss.txt", "w") as f:
   #    pass
   #with open("./my_data_and_graph/historydata/scheduleresults.txt", "w") as f:
   #    pass
        
    # Reset rewards log CSV (for plotting)
    with open("./my_data_and_graph/historydata/rewards_log.csv", "w") as f:
        f.write("episode,reward,time\n")

    # import datetime, os, time
    # from shutil import copyfile
    # if os.path.exists("my_data_and_graph/marl.time_reward.txt"):
    #     tar = "my_data_and_graph/marlhisrtorydata/" + str(datetime.date.today()) + "-" + str(time.time()).split(".")[
    #         0] + "marl.time_reward.txt"
    #     with open(tar, "w") as f:
    #         pass
    #     copyfile("my_data_and_graph/marl.time_reward.txt", tar)
    #
    # with open("my_data_and_graph/marl.time_reward.txt", "w") as f:
    #     pass
    # for i in range(8):  #  # because there are 8 MARL algorithms in total
    args = get_common_args()

    if args.alg.find('coma') > -1:  # choose algorithm-specific hyperparameters
        args = get_coma_args(args)
    elif args.alg.find('central_v') > -1:
        args = get_centralv_args(args)
    elif args.alg.find('reinforce') > -1:
        args = get_reinforce_args(args)
    else:
        args = get_mixer_args(args)
    if args.alg.find('commnet') > -1:
        args = get_commnet_args(args)
    if args.alg.find('g2anet') > -1:
        args = get_g2anet_args(args)

    # # Load the scheduling environment
    env = ScheduleEnv()

    env.reset()
    env_info = env.get_env_info()
    args.n_actions = env_info["n_actions"]
    args.n_agents = env_info["n_agents"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]

        # --- Training Setup Summary (printed to standart output) ---
    print("\n=== Training Setup Summary (Args) ===")
    print(f"Algorithm: {args.alg}")
    print(f"Map: {args.map}")
    print(f"Random seed: {args.seed}")
    print(f"Total epochs: {args.n_epoch}")
    print(f"Episodes per epoch: {args.n_episodes}")
    print(f"Evaluation every {args.evaluate_cycle} epochs, with {args.evaluate_epoch} episodes")
    print(f"Replay buffer size: {getattr(args, 'buffer_size', 'N/A')}")
    print(f"Batch size: {getattr(args, 'batch_size', 'N/A')}")
    print(f"Learning enabled: {args.learn}")
    print(f"GPU enabled: {args.cuda}")
    print(f"Load pretrained model: {args.load_model}")
    print("====================================")

    print("\n=== Environment Info ===")
    print(f"Number of agents: {args.n_agents}")
    print(f"Number of actions: {args.n_actions}")
    print(f"State shape: {args.state_shape}")
    print(f"Observation shape: {args.obs_shape}")
    print(f"Episode limit (steps per episode): {args.episode_limit}")
    print("====================================\n")

    print("Load model (test only：", args.load_model,  "Print intermediates:", args.havelook, "Train:",args.learn)

    runner = Runner(env, args)

    if args.learn:
        runner.run(0)  # # originally supported multiple algos; run() took an algorithm id
    else:
        _, reward = runner.evaluate()
        print('The ave_reward of {} is  {}'.format(args.alg, reward))


# # Random decision baseline for environment testing
def random_agent_wrapper():

    episodes = 50

    env = ScheduleEnv()
    temp_save = [0]
    EATs = []
    schedule_processes = []
    for episode in range(episodes):

        s = env.reset()
        is_terminal = False
        while not is_terminal:
            # print(1)
            actions = []
            # # Only dispatch agents that are not currently busy   
            temp_not_idle_agents = []
            for m in range(len(env.sites)):
                if s[m] != 9:
                    temp_not_idle_agents.append(s[m])

            for i in range(len(env.planes)):
                if i in temp_not_idle_agents:  # # agent i is currently busy
                    actions.append(18)
                else:
                    # print(i)
                    avail_actions = env.get_avail_agent_actions(i)
                    tem_choose = []
                    if type(avail_actions) != str:
                        for k, eve in enumerate(avail_actions):
                            if eve == 1:
                                tem_choose.append(k)
                        if tem_choose == []:
                            action = 18
                        else:
                            action = np.random.choice(tem_choose, 1, False)[0]
                            env.has_chosen_action(action, i)
                        actions.append(action)
                    else:
                        actions.append(18)
            s, r, is_terminal, dict = env.step(actions)
            # print(actions)
            # print(s[:18])
        EATs.append(dict["time"])
        schedule_processes.append(env.job_record_for_gant)
        print(env.job_record_for_gant)
        print(dict["time"], "-----------------------------------")
    print(sum(EATs)/len(EATs))
    # # Store intermediate results
    with open("./my_data_and_graph/pickles/process.pk", "wb") as f:
        pickle.dump(schedule_processes, f)


def SDrules_agent_wrapper():
    EPISODES = 50

    sd_rules = SDrules()
    env = ScheduleEnv()
    sites_locations = env.sites_obj.sites_position

    actions = []
    for episode in range(EPISODES):
        done = False
        env.reset()
        while not done:
            actions = []
            agents_id_sequence = sd_rules.FIFO_generate_agents_sequence(8)
            # agents_id_sequence = sd_rules.MLF_generate_agents_sequence(env.planes)
            # agents_id_sequence = sd_rules.LLF_generate_agents_sequence(env.planes)


            for agent_id in agents_id_sequence:
                avail_actions = env.get_avail_agent_actions(agent_id)
                current_plane_location = env.planes[agent_id].position
                action = sd_rules.choose_action(agent_id, avail_actions, current_plane_location, sites_locations)
                actions.append(action)
                if action < 18:
                    env.has_chosen_action(action, agent_id)
            ## Reorder the actions back into agent index order
            # because the environment expects actions in agent-id order
            reorder_actions = [-1 for i in range(8)]
            for i in range(8):
                reorder_actions[agents_id_sequence[i]] = actions[i]
            _, done, info = env.step(reorder_actions)
        print(info["time"])
    print(info['episodes_situation'])


if __name__ == "__main__":
    marl_agent_wrapper()
