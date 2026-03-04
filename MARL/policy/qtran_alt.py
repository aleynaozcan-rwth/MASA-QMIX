import torch
import torch.nn as nn
import os
from MARL.network.base_net import RNN
from MARL.network.qtran_net import QtranV, QtranQAlt


class QtranAlt:
    def __init__(self, args):
        # canonical args for the instance
        self.args = args
        self.n_actions = self.args.n_actions
        self.n_agents = self.args.n_agents
        self.state_shape = self.args.state_shape
        self.obs_shape = self.args.obs_shape
        rnn_input_shape = self.obs_shape

        # Determine RNN input dimension based on parameters
        if self.args.last_action:
            rnn_input_shape += self.n_actions  # One-hot vector of current agent's last action
        if self.args.reuse_network:
            rnn_input_shape += self.n_agents
        # Neural networks
        self.eval_rnn = RNN(rnn_input_shape, args)  # individual networks
        self.target_rnn = RNN(rnn_input_shape, args)

        self.eval_joint_q = QtranQAlt(args)  # counterfactual joint networks
        self.target_joint_q = QtranQAlt(args)
        self.v = QtranV(args)

        if self.args.cuda:
            self.eval_rnn.cuda()
            self.target_rnn.cuda()
            self.eval_joint_q.cuda()
            self.target_joint_q.cuda()
            self.v.cuda()

        self.model_dir = self.args.model_dir + '/' + self.args.alg + '/' + self.args.map
        # Load model if it exists
        if self.args.load_model:
            if os.path.exists(self.model_dir + '/rnn_net_params.pkl'):
                path_rnn = self.model_dir + '/rnn_net_params.pkl'
                path_joint_q = self.model_dir + '/joint_q_params.pkl'
                path_v = self.model_dir + '/v_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
                self.eval_joint_q.load_state_dict(torch.load(path_joint_q, map_location=map_location))
                self.v.load_state_dict(torch.load(path_v, map_location=map_location))
                print('Successfully load the model: {}, {} and {}'.format(path_rnn, path_joint_q, path_v))
            else:
                raise Exception("No model!")

        # Make target_net and eval_net network parameters the same
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_joint_q.load_state_dict(self.eval_joint_q.state_dict())

        self.eval_parameters = list(self.eval_joint_q.parameters()) + \
                               list(self.v.parameters()) + \
                               list(self.eval_rnn.parameters())
        if self.args.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=self.args.lr)

        # During execution, maintain an eval_hidden for each agent
        # During learning, maintain eval_hidden and target_hidden for each agent in each episode
        self.eval_hidden = None
        self.target_hidden = None
        print('Init alg QTRAN-alt')

    def learn(self, batch, max_episode_len, train_step, epsilon=None):  # train_step indicates which learning iteration, used to control target_net parameter updates
        '''
        During learning, the extracted data is 4D with dimensions: 1-which episode, 2-which transition in episode,
        3-which agent's data, 4-specific obs dimension. Because action selection requires not only current inputs but also
        hidden_state from the neural network, and hidden_state is related to previous experiences, we cannot randomly sample
        experiences for learning. Therefore, multiple episodes are extracted at once, and transitions at the same position
        from each episode are fed to the neural network together.
        '''
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():  # Convert batch data to tensors
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        s, s_next, u, r, avail_u, avail_u_next, terminated = batch['s'], batch['s_next'], batch['u'], \
                                                             batch['r'],  batch['avail_u'], batch['avail_u_next'],\
                                                             batch['terminated']
        mask = 1 - batch["padded"].float().repeat(1, 1, self.n_agents)  # Used to set TD-error to 0 for padded experiences to prevent them from affecting learning
        if self.args.cuda:
            u = u.cuda()
            r = r.cuda()
            avail_u = avail_u.cuda()
            avail_u_next = avail_u_next.cuda()
            terminated = terminated.cuda()
            mask = mask.cuda()
        # Get Q values and hidden_states for each agent, dimensions: (episode_num, max_episode_len, n_agents, n_actions/hidden_dim)
        individual_q_evals, individual_q_targets, hidden_evals, hidden_targets = self._get_individual_q(batch, max_episode_len)

        # Get local optimal actions and their one-hot representations for each agent at current and next time steps
        individual_q_clone = individual_q_evals.clone()
        individual_q_clone[avail_u == 0.0] = - 999999
        individual_q_targets[avail_u_next == 0.0] = - 999999

        opt_onehot_eval = torch.zeros(*individual_q_clone.shape)
        opt_action_eval = individual_q_clone.argmax(dim=3, keepdim=True)
        opt_onehot_eval = opt_onehot_eval.scatter(-1, opt_action_eval[:, :].cpu(), 1)

        opt_onehot_target = torch.zeros(*individual_q_targets.shape)
        opt_action_target = individual_q_targets.argmax(dim=3, keepdim=True)
        opt_onehot_target = opt_onehot_target.scatter(-1, opt_action_target[:, :].cpu(), 1)

        # ---------------------------------------------L_td-------------------------------------------------------------

        # Calculate joint_q and v, note that each agent has joint_q but there is only one v
        # joint_q dimensions: (episode_num, max_episode_len, n_agents, n_actions), and joint_q will be used in l_nopt later
        # v dimensions: (episode_num, max_episode_len)
        joint_q_evals, joint_q_targets, v = self.get_qtran(batch, opt_onehot_target, hidden_evals, hidden_targets)

        # Extract joint_q_chosen for current agent action and joint_q for its local optimal action
        joint_q_chosen = torch.gather(joint_q_evals, dim=-1, index=u).squeeze(-1)  # (episode_num, max_episode_len, n_agents)
        joint_q_opt = torch.gather(joint_q_targets, dim=-1, index=opt_action_target).squeeze(-1)

        # loss
        y_dqn = r.repeat(1, 1, self.n_agents) + self.args.gamma * joint_q_opt * (1 - terminated.repeat(1, 1, self.n_agents))
        td_error = joint_q_chosen - y_dqn.detach()
        l_td = ((td_error * mask) ** 2).sum() / mask.sum()
        # ---------------------------------------------L_td-------------------------------------------------------------

        # ---------------------------------------------L_opt------------------------------------------------------------

        # Sum the Q values of local optimal actions  (episode_num, max_episode_len)
        # Use individual_q_clone here as it modifies Q values for unavailable actions; using individual_q_evals might use Q values of unavailable actions
        q_sum_opt = individual_q_clone.max(dim=-1)[0].sum(dim=-1)

        # Recompute joint_q_opt_eval; difference from joint_q_evals is that the former uses current local optimal actions while the latter uses current executed actions
        joint_q_opt_evals, _, _ = self.get_qtran(batch, opt_onehot_eval, hidden_evals, hidden_targets, hat=True)
        joint_q_opt_evals = torch.gather(joint_q_opt_evals, dim=-1, index=opt_action_eval).squeeze(-1)  # (episode_num, max_episode_len, n_agents)

        # Because QTRAN-alt computes l_opt for each agent, need to add an agent dimension to q_sum_opt and v
        q_sum_opt = q_sum_opt.unsqueeze(-1).expand(-1, -1, self.n_agents)
        v = v.unsqueeze(-1).expand(-1, -1, self.n_agents)
        opt_error = q_sum_opt - joint_q_opt_evals.detach() + v  # Need to fix joint_q_opt_evals when computing l_opt
        l_opt = ((opt_error * mask) ** 2).sum() / mask.sum()

        # ---------------------------------------------L_opt------------------------------------------------------------

        # ---------------------------------------------L_nopt-----------------------------------------------------------
        # Because L_nopt constrains the minimum d among all executable actions of the current agent, to prevent unavailable actions from affecting d calculation, increase q values for unavailable actions
        individual_q_evals[avail_u == 0.0] = 999999

        # Get the sum of Q values q_other_sum for executed actions of agents other than agent_i
        #   1. First get Q values q_all for each agent's executed action, (episode_num, max_episode_len, n_agents, 1)
        q_all_chosen = torch.gather(individual_q_evals, dim=-1, index=u)
        #   2. Transform q_all on the last dimension from current agent's Q value to all agents' Q values, (episode_num, max_episode_len, n_agents, n_agents)
        q_all_chosen = q_all_chosen.view((episode_num, max_episode_len, 1, -1)).repeat(1, 1, self.n_agents, 1)
        q_mask = (1 - torch.eye(self.n_agents)).unsqueeze(0).unsqueeze(0)
        if self.args.cuda:
            q_mask = q_mask.cuda()
        q_other_chosen = q_all_chosen * q_mask  # Set each agent's own Q value to 0, so that summing gives the sum of other agents' Q values
        #   3. Sum, and because for each action of the current agent we need to add q_other_sum, expand q_other_sum to n_actions dimension
        q_other_sum = q_other_chosen.sum(dim=-1, keepdim=True).repeat(1, 1, 1, self.n_actions)

        # Add Q value for each action of current agent with Q values of other agents' executed actions, obtaining the first term in D
        q_sum_nopt = individual_q_evals + q_other_sum

        # Because joint_q_evals dimensions are (episode_num, max_episode_len, n_agents, n_actions), need to expand v to include n_actions dimension
        v = v.unsqueeze(-1).expand(-1, -1, -1, self.n_actions)
        d = q_sum_nopt - joint_q_evals.detach() + v  # Need to fix qtran_q_evals when computing l_nopt
        d = d.min(dim=-1)[0]
        l_nopt = ((d * mask) ** 2).sum() / mask.sum()
        # ---------------------------------------------L_nopt-----------------------------------------------------------

        # print('l_td is {}, l_opt is {}, l_nopt is {}'.format(l_td, l_opt, l_nopt))
        loss = l_td + self.args.lambda_opt * l_opt + self.args.lambda_nopt * l_nopt
        # loss = l_td + self.args.lambda_opt * l_opt
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_joint_q.load_state_dict(self.eval_joint_q.state_dict())

    def _get_individual_q(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        q_evals, q_targets, hidden_evals, hidden_targets = [], [], [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_individual_inputs(batch, transition_idx)  # Add last_action and agent_id to obs
            if self.args.cuda:
                inputs = inputs.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
                inputs_next = inputs_next.cuda()
                self.target_hidden = self.target_hidden.cuda()
            q_eval, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)
            q_target, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden)
            hidden_eval, hidden_target = self.eval_hidden.clone(), self.target_hidden.clone()

            # Reshape q_eval dimensions back to (8, 5, n_actions)
            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            hidden_eval = hidden_eval.view(episode_num, self.n_agents, -1)
            hidden_target = hidden_target.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)
            hidden_evals.append(hidden_eval)
            hidden_targets.append(hidden_target)
        # Obtained q_eval and q_target are lists containing max_episode_len arrays, each array has dimensions (episode_num, n_agents, n_actions)
        # Convert this list to an array with dimensions (episode_num, max_episode_len, n_agents, n_actions)
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        hidden_evals = torch.stack(hidden_evals, dim=1)
        hidden_targets = torch.stack(hidden_targets, dim=1)
        return q_evals, q_targets, hidden_evals, hidden_targets

    def _get_individual_inputs(self, batch, transition_idx):
        # Extract experiences at transition_idx from all episodes, u_onehot extracts all because previous action is needed
        obs, obs_next, u_onehot = batch['o'][:, transition_idx], \
                                  batch['o_next'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)
        # Add last action and agent ID to obs
        if self.args.last_action:
            if transition_idx == 0:  # If it's the first experience, set the previous action to a zero vector
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
            inputs_next.append(u_onehot[:, transition_idx])
        if self.args.reuse_network:
            # Because current obs is 3D data where each dimension represents (episode_id, agent_id, obs_dim), we can directly add
            # corresponding vectors on dim_1. For example, add (1, 0, 0, 0, 0) after agent_0, representing agent 0 among 5 agents.
            # Since agent_0's data is exactly in row 0, the agent ID to be added is exactly an identity matrix, i.e., diagonal is 1, rest is 0
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            inputs_next.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        # Concatenate the three parts in obs, and combine data from episode_num episodes and self.args.n_agents agents into 40 pieces of (40,96) data,
        # because all agents share one neural network here, each data piece carries its own ID, so it's still its own data
        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1)
        return inputs, inputs_next

    def get_qtran(self, batch, local_opt_actions, hidden_evals, hidden_targets=None, hat=False):
        episode_num, max_episode_len, _, _ = hidden_evals.shape
        s = batch['s'][:, :max_episode_len]
        s_next = batch['s_next'][:, :max_episode_len]
        u_onehot = batch['u_onehot'][:, :max_episode_len]
        v_state = s.clone()

        # s and s_next don't have n_agents dimension, but each agent's joint_q network needs it, so convert s to 4D
        s = s.unsqueeze(-2).expand(-1, -1, self.n_agents, -1)
        s_next = s_next.unsqueeze(-2).expand(-1, -1, self.n_agents, -1)
        # Add one-hot vector corresponding to agent ID
        '''
        Because current inputs is 3D data where each dimension represents (episode_id, agent_id, inputs_dim), we can directly add
        corresponding vectors at the end. For example, add (1, 0, 0, 0, 0) after agent_0, representing agent 0 among 5 agents.
        Since agent_0's data is exactly in row 0, the agent ID to be added is exactly an identity matrix, i.e., diagonal is 1, rest is 0
        '''
        action_onehot = torch.eye(self.n_agents).unsqueeze(0).unsqueeze(0).expand(episode_num, max_episode_len, -1, -1)
        s_eval = torch.cat([s, action_onehot], dim=-1)
        s_target = torch.cat([s_next, action_onehot], dim=-1)
        if self.args.cuda:
            s_eval = s_eval.cuda()
            s_target = s_target.cuda()
            v_state = v_state.cuda()
            u_onehot = u_onehot.cuda()
            hidden_evals = hidden_evals.cuda()
            hidden_targets = hidden_targets.cuda()
            local_opt_actions = local_opt_actions.cuda()
        if hat:
            # Neural network output q_eval and q_target have dimensions (episode_num * max_episode_len * n_agents, n_actions)
            q_evals = self.eval_joint_q(s_eval, hidden_evals, local_opt_actions)
            q_targets = None
            v = None

            # Reshape q_eval dimensions back to (episode_num, max_episode_len, n_agents, n_actions)
            q_evals = q_evals.view(episode_num, max_episode_len, -1, self.n_actions)
        else:
            q_evals = self.eval_joint_q(s_eval, hidden_evals, u_onehot)
            q_targets = self.target_joint_q(s_target, hidden_targets, local_opt_actions)
            v = self.v(v_state, hidden_evals)
            # Reshape q_eval and q_target dimensions back to (episode_num, max_episode_len, n_agents, n_actions)
            q_evals = q_evals.view(episode_num, max_episode_len, -1, self.n_actions)
            q_targets = q_targets.view(episode_num, max_episode_len, -1, self.n_actions)
            # Reshape v dimensions back to (episode_num, max_episode_len)
            v = v.view(episode_num, -1)

        return q_evals, q_targets, v

    def init_hidden(self, episode_num):
        # Initialize eval_hidden and target_hidden for each agent in each episode
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        torch.save(self.eval_rnn.state_dict(),  self.model_dir + '/' + num + '_rnn_net_params.pkl')
        torch.save(self.eval_joint_q.state_dict(), self.model_dir + '/' + num + '_joint_q_params.pkl')
        torch.save(self.v.state_dict(), self.model_dir + '/' + num + '_v_params.pkl')
