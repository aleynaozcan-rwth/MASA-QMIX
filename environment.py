'''
In this code, "plane" stands for a job, and "site" stands for a station.
This environment schedules planes (jobs) to sites (stations) under resource constraints.
'''
from utils.site import Sites
from utils.job import Jobs
from utils.task import Task
from utils.plane import Planes
from utils import util
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import math
# # Whole environment class
class ScheduleEnv(gym.Env):
    environment_name = "Boat Schedule"

    def __init__(self):
        #  # Declare member variables
        self.sites = []
        self.jobs = []
        self.task = []
        self.planes_obj = Planes()
        self.planes = []
        self.state = [[]]
        self.done = False
        self.state_left_time = []
        self.episode_time_slice = []  ## list of time consumed at each step within the episode
        self.plane_speed = 0  ## movement/processing speed
        self.debug = True     # I am adding this to debug couple of episodes-Aleyna
        self.initialize()  #  # initialize all environment parameters

        # Planes controlled by DQN do not need an explicit "wait" action; they will choose a feasible action.
        # Action meaning:
        #   0 .. len(self.sites)-1 : go to that site next
        #   len(self.sites)        : wait (due to resource conflict / no feasible assignment now)
        #   len(self.sites)+1      : busy (currently processing)   -> not used for training
        #   len(self.sites)+2      : finished (no jobs remaining)  -> not used for training

        
         
        #“0–17 indicate the site to go to next; 18 means wait due to a resource conflict; 19 means currently busy (processing); 20 means finished. Actions 19 and 20 are not used for training.”
        self.action_space = spaces.Discrete(len(self.sites)+3)  # +3 for [wait, busy, finished]
        self.id = "Boat Schedule"
        #   # Legacy/compat placeholders (not critical for training)
        self.reward_threshold = -1000
        self.trials = 50  # # roughly analogous to steps in older experiments

        self.job_record_for_gant = []  # # store (timestamp, job_id, site_id, plane_id) tuples for Gantt visualization

        self.sites_state_global = None  # this para is utilized to indicate the current idle sites and their processing jobs # indicates idle sites (-1) or the job-id currently being processed at each site

        #  # One global state and per-agent observations
        self.state4marl = None  # # global state maintained for MARL
        self.obs4marl = None

    def initialize(self):
        sites_obj = Sites()
        self.sites_obj = sites_obj
        jobs_obj = Jobs()
        task_obj = Task()
        self.planes_obj = Planes()
        self.sites = sites_obj.sites_object_list
        self.jobs = jobs_obj.jobs_object_list
        #   # Task: sequence of jobs
        self.task = task_obj.simple_task_object

        self.planes = self.planes_obj.planes_object_list

        self.state = [[9, [1 if j in self.sites[i].resource_ids_list else 0 for j in range(9)]] for i in range(len(self.sites))]

        self.sites_state_global = [-1 for i in range(len(self.sites))] # # -1 means no support task assigned

        self.job_record_for_gant = []  # # store (timestamp, job_id, site_id, plane_id) tuples during scheduling


        self.done = False
        self.state_left_time = np.array([0 for i in range(len(self.sites))])
        self.episode_time_slice = []
        self.plane_speed = self.planes_obj.plane_speed  # # running speed
        # print("the environment is initialized now !!")
        self.obs4marl = [[] for i in range(len(self.planes))]
        self.current_finishing_jobs = 0
        self.step_count = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.initialize()  # # re-initialize all parameters
        info = {
            "sites": [[self.sites[i].absolute_position,
                       self.state[i][0],
                       self.state[i][1]
                       ] for i in range(len(self.sites))],
            "planes": [[self.planes[i].left_job[0].index_id,
                        self.jobs[self.planes[i].left_job[0].index_id].time_span,
                        len(self.planes[i].left_job)
                        ] if len(self.planes[i].left_job) != 0
                       else [
                9,
                0,
                len(self.planes[i].left_job)
            ] for i in range(len(self.planes))],
            "planes_obj": self.planes
        }
        state = self.conduct_state(info)
        # print(info)
        # print(len(state))

        if self.debug:
            print("\n=== NEW EPISODE STARTED ===")
            print("Initial plane positions:", [p.position for p in self.planes])
            print("Initial sites state:", self.state)
            print("Initial state vector length:", len(state))

        return state  # 151

    def conduct_state(self, info):
        res = []
        temp = []

        for eve in info["sites"]:
            # res.append(eve[1])
            temp.append(eve[1])

        for eve in info["sites"]:

            res += eve[2]

        for i, eve in enumerate(info["planes"]):
            res += [eve[2]]
            temp_obs = []
            for l, eve_1 in enumerate(temp):
                if self.planes[i].left_job == []:
                    temp_obs.append(0)
                else:
                      # idle site and site can process plane's next job -> plane can go there next
                    if eve_1 == 9 and self.planes[i].left_job[0].index_id in self.sites[l].resource_ids_list:

                        temp_obs.append(util.count_path_on_road(self.planes[i].position, self.sites[l].absolute_position, self.plane_speed)/40)  # 代表处于空闲状态,下一步飞机可以去
                    else:
                        temp_obs.append(0)  # # otherwise the plane is busy; cannot go next step
            # per-agent obs = distances/eligibility + [next_job_id, remaining_jobs, time_span]           
            self.obs4marl[i] = temp_obs + [eve[0], eve[2], eve[1]]

        current_working_plane_ids = []
        for eve in self.state:
            if eve[0] != 9:
                current_working_plane_ids.append(eve[0])
        if current_working_plane_ids == []:
            for k in range(len(self.obs4marl)):
                self.obs4marl[k].append(0)
        else:
            for k in range(len(self.obs4marl)):
                if k in current_working_plane_ids:
                    #  # when busy, zero all features except the last two here; after appending busy_flag below,
                    # effectively only the last three features remain non-zero
                    self.obs4marl[k] = [0 if kk < len(self.obs4marl[k])-2 else self.obs4marl[k][kk] for kk in range(len(self.obs4marl[k]))]
                    self.obs4marl[k].append(1)
                else:
                    self.obs4marl[k].append(0)
        obslen = len(self.obs4marl[0])
        zero_obs = [0 for i in range(obslen)]


        for i, plane in enumerate(self.planes):
            if len(plane.left_job) == 0:
                self.obs4marl[i] = zero_obs

        self.state4marl = np.array(res)
        return np.array(res)

    def check_inflict_action(self, action):
        res = []

        for eve in action:
            if eve == len(self.sites) or eve == 19 or eve == 20:
                res.append(eve)
            else:
                if eve not in res:
                    res.append(eve)
                else:
                    raise Exception("sloppy error in actions", action)
                    # assert False
        return res

    # Replace certain action codes (treat BUSY/FINISHED as WAIT)
    def action_replace(self, action):
        res = []
        real_conflict_num = 0
        for eve in action:
            if eve == 20 or eve == 19:
                res.append(18)
            else:
                if eve == 18:
                    real_conflict_num += 1

                res.append(eve)
        return res, real_conflict_num

    def step(self, action):
        self.step_count += 1
        action, real_conflict_num = self.action_replace(action)  # replace 19/20 with 18

        if self.debug:
            print(f"\n[STEP {self.step_count}] Actions: {action}")

        if real_conflict_num != 0:
            if self.debug:
                print("⚠ Conflict detected! Number of conflicts:", real_conflict_num)

        count_break_rules = 0
        assert len(action) == len(self.planes)
        rewards = [0 for _ in action]
        max_time_on_roads = [0 for _ in action]
        count_for_reward = 0
        action = self.check_inflict_action(action)
        time_span_increase = np.array([0 for eve in self.sites])

        for i, site_id in enumerate(action):
            if site_id == len(self.sites):  # WAIT action
                rewards[i] = -30
                if self.debug:
                    print(f"Plane {i} chose WAIT → penalty -30")
            else:
                # assign a support/scheduling task
                if self.planes[i].left_job[0].index_id in self.sites[site_id].resource_ids_list:
                    time_on_road = util.count_path_on_road(
                        self.planes[i].position,
                        self.sites[site_id].absolute_position.tolist(),
                        self.plane_speed
                    )
                    start_time = sum(self.episode_time_slice)
                    temp_time = self.planes[i].execute_task(self.planes[i].left_job[0], self.sites[site_id])
                    duration = temp_time + time_on_road
                    end_time = start_time + duration

                    # FIX: check if left_job exists
                    if len(self.planes[i].left_job) > 0:
                        job_id = self.planes[i].left_job[0].index_id
                    else:
                        job_id = -1  # placeholder

                    # save env info
                    if type(site_id) == int:
                        self.save_env_info((start_time, end_time, job_id, site_id, i))
                    else:
                        self.save_env_info((start_time, end_time, job_id, site_id.item(), i))

                    time_span_increase[site_id] = duration
                    self.state[site_id][0] = i  # mark the site as occupied
                    count_for_reward += 1
                    max_time_on_roads[i] = time_on_road

                    if self.debug:
                        print(f"Plane {i} → Site {site_id}, Job {job_id}, "
                            f"Travel {time_on_road}, Duration {duration}, "
                            f"Reward {rewards[i]}")
                else:
                    raise Exception("Invalid action was not masked",
                                    self.sites_state_global, i, site_id, action,
                                    self.planes[i].left_job[0].index_id,
                                    self.sites[site_id].resource_ids_list,
                                    self.state)

            # update state
            self.episode_time_slice.append(max(max_time_on_roads) if max_time_on_roads else 0)

            # DEBUG: print current state vector
            if self.debug:
                print(f"[DEBUG] Current state vector (len={len(self.state)}): {self.state}")

            # Info dictionary with debug state
            info = {"state": self.state} if self.debug else {}

            #return self.state, sum(rewards), self.done, {}




        real_did = 0
        for eve in action:
            if eve < 18:
                real_did += 1

        for i, site_id in enumerate(action):
            if site_id == len(self.sites):  # # this plane is not scheduled this step

                rewards[i] = - 30  #  # penalty for waiting due to resource conflict
            else:  # scheduled
                if rewards[i] == 0:
                    # rewards[i] = -(max_time_on_roads[i]+0.1)/(max(max_time_on_roads)+0.1)-real_conflict_num
                    rewards[i] = -(max_time_on_roads[i]+0.1)/(max(max_time_on_roads)+0.1)

                else:
                    pass

        # Update remaining processing time
        self.state_left_time = self.state_left_time + time_span_increase

        min_time = util.min_but_zero(self.state_left_time)
        # print("time consumed:", min_time)
        self.episode_time_slice.append(min_time)  #  # time consumed in this step
        self.state_left_time = util.advance_by_min_time(min_time, self.state_left_time)  # # advance the step

        #  # Update site states; mainly check which ones have finished
        # state transition 2
        for i, eve_time in enumerate(self.state_left_time):
            if eve_time == 0:
                self.sites_state_global[i] = -1  #  # mark finished sites as idle
                self.sites_obj.update_site_resources(self.sites_state_global)
                self.state[i][0] = 9
                self.state[i][1] = [1 if j in self.sites[i].resource_ids_list else 0 for j in range(9)]
            else:
                assert self.state[i][0] != 9  # # with remaining time, the site must be occupied

        #  # Check whether the episode is finished
        is_all_done = [-1 for eve in self.planes]
        for i, plane in enumerate(self.planes):
            if len(plane.left_job) == 0:
                is_all_done[i] = 0
        # self.current_finishing_jobs = sum(is_all_done) + len(is_all_done) - self.current_finishing_jobs
        if sum(is_all_done) == 0:
            self.done = True
        else:
            self.done = False


        left_jobs, all_jobs = self.planes_obj.count_jobs()
        if self.done:
            reward = 6000 / (sum(self.episode_time_slice) + max(self.state_left_time))
            # print(11, reward)

        else:
            reward = real_did - self.step_count/60 - real_conflict_num*2

        info = {
            "sites": [[self.sites[i].absolute_position,
                       self.state[i][0],
                       self.state[i][1]
                       ] for i in range(len(self.sites))],
            "planes": [[self.planes[i].left_job[0].index_id,
                        len(self.planes[i].site_history),
                        len(self.planes[i].left_job)
                        ] if len(self.planes[i].left_job) != 0
                       else [
                        9,
                        len(self.planes[i].site_history),
                        len(self.planes[i].left_job)
                    ]for i in range(len(self.planes))],
            "planes_obj": self.planes
        }
        state = self.conduct_state(info)
        # print("min_time:", min_time)
        # print("left_time:", self.state_left_time)
        return self.get_state(), reward, self.done, {
            "time": sum(self.episode_time_slice)+max(self.state_left_time),
            "left": self.state_left_time,
            "original_state": self.state,
            "planes_obj": self.planes,
            "rewards": rewards,
            "count_break_rules": count_break_rules,
            "sites_state_global": self.sites_state_global,
            "episodes_situation": self.job_record_for_gant
        }


    def get_avail_agent_actions(self, agent_id):
        #  # Check whether the plane is currently busy
        for eve in self.state:
            if agent_id == eve[0]:  # # this plane is still processing
                # return [0 for i in range(18)] + [1]  # 1
                return [0 for i in range(18)] + [1, 0, 0]   # only BUSY is available

        # # only BUSY is available
        res = [0 for eve in self.sites_state_global]
        for i, eve in enumerate(self.sites_state_global):
            if eve == -1:
                if len(self.planes[agent_id].left_job) != 0:
                    #  # Check whether the plane's next job type is supported by this site
                    if self.planes[agent_id].left_job[0].index_id in self.sites[i].resource_ids_list:
                        res[i] = 1
                else:  # # this plane has finished all its scheduled jobs
                    # return [0 for i in range(18)] + [1]  # 0
                    return [0 for i in range(18)] + [0, 0, 1]   # only FINISHED is available

        print(f"[DEBUG] Agent {agent_id} avail actions: {res + [0, 1, 0]}")        
        return res + [0, 1, 0]   # add WAIT



    # state transition 1
    def has_chosen_action(self, action_id, agent_id):
        assert self.planes[agent_id].left_job != []
        # print(action_id, agent_id)
        self.sites_state_global[action_id] = self.planes[agent_id].left_job[0].index_id  # # update site status info
        #  # Update the global resource list state — careful: this is choosing a feasible action, not executing it yet
        self.sites_obj.update_site_resources(self.sites_state_global)
        self.state[action_id][0] = agent_id  # # mark this site as occupied
        self.state[action_id][1] = [1 if j in self.sites[action_id].resource_ids_list else 0 for j in range(9)]  # # update resource occupancy


    def save_env_info(self, job_transition):
        self.job_record_for_gant.append(job_transition)

    def get_state(self):
        assert self.state4marl is not None
        return self.state4marl

    def get_obs(self):
        # assert self.obs4marl is not None and self.obs4marl != []
        agents_obs = [self.get_obs_agent(i) for i in range(len(self.planes))]
        return agents_obs

    def get_obs_agent(self, agent_id):
        return self.obs4marl[agent_id]

    def get_env_info(self):
        return {
            "n_actions": len(self.sites) + 3,  # include the idle/wait action
            "n_agents": len(self.planes),
            "state_shape": len(self.get_state()),
            "obs_shape": len(self.get_obs()[0]),
            "episode_limit": 80   # Note: if the schedule can’t finish within 80 steps the program will error;
                                   # setting it too large will lead to paddings later
        }
