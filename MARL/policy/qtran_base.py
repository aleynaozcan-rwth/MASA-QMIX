import torch
import torch.nn as nn
import os
from MARL.network.base_net import RNN
from MARL.network.qtran_net import QtranV, QtranQBase


class QtranBase:
    def __init__(self, args):
        self.args = args
        self.n_actions = self.args.n_actions
        self.n_agents = self.args.n_agents
        self.state_shape = self.args.state_shape
        self.obs_shape = self.args.obs_shape

        rnn_input_shape = self.obs_shape

        # Determine RNN input dimension based on parameters
        if self.args.last_action:
            rnn_input_shape += self.n_actions  # Current agent's previous action one-hot vector
        if self.args.reuse_network:
            rnn_input_shape += self.n_agents

        # Neural networks
        self.eval_rnn = RNN(rnn_input_shape, self.args)  # Network for each agent to select actions
        self.target_rnn = RNN(rnn_input_shape, self.args)

        self.eval_joint_q = QtranQBase(self.args)  # Joint action-value network
        self.target_joint_q = QtranQBase(self.args)

        self.v = QtranV(self.args)

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

        # During execution, maintain an eval hidden for each agent
        # During learning, maintain eval_hidden and target_hidden for each agent in each episode
        self.eval_hidden = None
        self.target_hidden = None
        print('Init alg QTRAN-base')

    def learn(self, batch, max_episode_len, train_step, epsilon=None):  # train_step represents the learning step number, used to control updating target_net network parameters
        '''
        During learning, extracted data is 4D, the four dimensions are: 1-which episode 2-which transition in episode
        3-which agent's data 4-specific obs dimension. Because action selection requires not only current inputs but also hidden_state from neural network,
        hidden_state is related to previous experiences, so we cannot randomly sample experiences for learning. Therefore we sample multiple episodes at once, then pass
        same position transitions from each episode to the neural network at once
        '''
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():  # Convert batch data to tensor
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        u, r, avail_u, avail_u_next, terminated = batch['u'], batch['r'],  batch['avail_u'], \
                                                  batch['avail_u_next'], batch['terminated']
        mask = (1 - batch["padded"].float()).squeeze(-1)  # Used to set TD-error to 0 for padded experiences, preventing them from affecting learning
        if self.args.cuda:
            u = u.cuda()
            r = r.cuda()
            avail_u = avail_u.cuda()
            avail_u_next = avail_u_next.cuda()
            terminated = terminated.cuda()
            mask = mask.cuda()
        # Get Q and hidden_states for each agent, dimensions (episode count, max_episode_len, n_agents, n_actions/hidden_dim)
        individual_q_evals, individual_q_targets, hidden_evals, hidden_targets = self._get_individual_q(batch, max_episode_len)

        # Get local optimal actions and their one-hot representations for current and next timesteps for each agent
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
        # Calculate joint_q and v
        # joint_q and v have dimensions (episode count, max_episode_len, 1); joint_q will also be used in l_nopt later
        joint_q_evals, joint_q_targets, v = self.get_qtran(batch, hidden_evals, hidden_targets, opt_onehot_target)

        # loss
        y_dqn = r.squeeze(-1) + self.args.gamma * joint_q_targets * (1 - terminated.squeeze(-1))
        td_error = joint_q_evals - y_dqn.detach()
        l_td = ((td_error * mask) ** 2).sum() / mask.sum()
        # ---------------------------------------------L_td-------------------------------------------------------------

        # ---------------------------------------------L_opt------------------------------------------------------------
        # Sum the Q values of local optimal actions
        # Need to use individual_q_clone here, which changes Q values of actions that cannot be executed; using individual_q_evals might use Q values of unavailable actions
        q_sum_opt = individual_q_clone.max(dim=-1)[0].sum(dim=-1)  # (episode count, max_episode_len)

        # Get joint_q_hat_opt again; difference from joint_q_evals is that former inputs local optimal actions, latter inputs executed actions
        # (episode count, max_episode_len)
        joint_q_hat_opt, _, _ = self.get_qtran(batch, hidden_evals, hidden_targets, opt_onehot_eval, hat=True)
        opt_error = q_sum_opt - joint_q_hat_opt.detach() + v  # Need to fix joint_q_hat_opt when calculating l_opt
        l_opt = ((opt_error * mask) ** 2).sum() / mask.sum()
        # ---------------------------------------------L_opt------------------------------------------------------------

        # ---------------------------------------------L_nopt-----------------------------------------------------------
        # Q value of each agent's executed action, (episode count, max_episode_len, n_agents, 1)
        q_individual = torch.gather(individual_q_evals, dim=-1, index=u).squeeze(-1)
        q_sum_nopt = q_individual.sum(dim=-1)  # (episode count, max_episode_len)

        nopt_error = q_sum_nopt - joint_q_evals.detach() + v  # Need to fix joint_q_evals when calculating l_nopt
        nopt_error = nopt_error.clamp(max=0)
        l_nopt = ((nopt_error * mask) ** 2).sum() / mask.sum()
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
                inputs_next = inputs_next.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
                self.target_hidden = self.target_hidden.cuda()

            # Need to use first experience to initialize target network's hidden_state; directly passing second experience to target network is incorrect
            if transition_idx == 0:
                _, self.target_hidden = self.target_rnn(inputs, self.eval_hidden)
            q_eval, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)  # inputs dimension is (40, 96), resulting q_eval dimension is (40, n_actions)
            q_target, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden)
            hidden_eval, hidden_target = self.eval_hidden.clone(), self.target_hidden.clone()

            # Reshape q_eval dimension back to (8, 5, n_actions)
            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            hidden_eval = hidden_eval.view(episode_num, self.n_agents, -1)
            hidden_target = hidden_target.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)
            hidden_evals.append(hidden_eval)
            hidden_targets.append(hidden_target)
        # The resulting q_eval and q_target are lists containing max_episode_len arrays, each array has dimension (episode count, n_agents, n_actions)
        # Convert this list to (episode count, max_episode_len, n_agents, n_actions) array
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        hidden_evals = torch.stack(hidden_evals, dim=1)
        hidden_targets = torch.stack(hidden_targets, dim=1)
        return q_evals, q_targets, hidden_evals, hidden_targets

    def _get_individual_inputs(self, batch, transition_idx):
        # Extract experiences at transition_idx from all episodes; u_onehot extracts all because we need the previous one
        obs, obs_next, u_onehot = batch['o'][:, transition_idx], \
                                  batch['o_next'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)
        # Add last action and agent ID to obs
        if self.args.last_action:
            if transition_idx == 0:  # If it's the first experience, set previous action to zero vector
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
            inputs_next.append(u_onehot[:, transition_idx])
        if self.args.reuse_network:
            # Since current obs is 3D data where each dimension represents (episode ID, agent ID, obs dimension), we can directly add the corresponding vector on dim_1
            # For example, add (1, 0, 0, 0, 0) after agent_0, representing agent 0 out of 5 agents. Since agent_0 data is exactly in row 0, the agent ID to add
            # is exactly an identity matrix, i.e., diagonal is 1, rest is 0
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            inputs_next.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        # Concatenate the three parts in obs, and combine episode_num episodes and self.args.n_agents agents data into episode_num*n_agents rows of data
        # Since all agents share one neural network here, each data row contains its own ID, so it's still its own data
        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1)
        return inputs, inputs_next

    def get_qtran(self, batch, hidden_evals, hidden_targets, local_opt_actions, hat=False):
        episode_num, max_episode_len, _, _ = hidden_targets.shape
        states = batch['s'][:, :max_episode_len]
        states_next = batch['s_next'][:, :max_episode_len]
        u_onehot = batch['u_onehot'][:, :max_episode_len]
        if self.args.cuda:
            states = states.cuda()
            states_next = states_next.cuda()
            u_onehot = u_onehot.cuda()
            hidden_evals = hidden_evals.cuda()
            hidden_targets = hidden_targets.cuda()
            local_opt_actions = local_opt_actions.cuda()
        if hat:
            # Neural network output q_eval, q_target, v have dimensions (episode_num * max_episode_len, 1)
            q_evals = self.eval_joint_q(states, hidden_evals, local_opt_actions)
            q_targets = None
            v = None

            # Reshape q_eval dimensions back to (episode_num, max_episode_len)
            q_evals = q_evals.view(episode_num, -1, 1).squeeze(-1)
        else:
            q_evals = self.eval_joint_q(states, hidden_evals, u_onehot)
            q_targets = self.target_joint_q(states_next, hidden_targets, local_opt_actions)
            v = self.v(states, hidden_evals)
            # Reshape q_eval, q_target, v dimensions back to (episode_num, max_episode_len)
            q_evals = q_evals.view(episode_num, -1, 1).squeeze(-1)
            q_targets = q_targets.view(episode_num, -1, 1).squeeze(-1)
            v = v.view(episode_num, -1, 1).squeeze(-1)

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
