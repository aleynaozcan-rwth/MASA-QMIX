import torch
import os
from MARL.network.base_net import RNN
from MARL.network.vdn_net import VDNNet


class VDN:
    def __init__(self, args):
        # canonical args
        self.args = args
        self.n_actions = self.args.n_actions
        self.n_agents = self.args.n_agents
        self.state_shape = self.args.state_shape
        self.obs_shape = self.args.obs_shape
        input_shape = self.obs_shape
        # Determine RNN input dimension based on parameters
        if self.args.last_action:
            input_shape += self.n_actions
        if self.args.reuse_network:
            input_shape += self.n_agents

        # Neural networks
        self.eval_rnn = RNN(input_shape, args)  # Network for each agent to select actions
        self.target_rnn = RNN(input_shape, args)
        self.eval_vdn_net = VDNNet()  # Network that sums agents' Q values
        self.target_vdn_net = VDNNet()
        if self.args.cuda:
            self.eval_rnn.cuda()
            self.target_rnn.cuda()
            self.eval_vdn_net.cuda()
            self.target_vdn_net.cuda()

        self.model_dir = self.args.model_dir + '/' + self.args.alg + '/' + self.args.map
        # Load model if it exists
        if self.args.load_model:
            if os.path.exists(self.model_dir + '/rnn_net_params.pkl'):
                path_rnn = self.model_dir + '/rnn_net_params.pkl'
                path_vdn = self.model_dir + '/vdn_net_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
                self.eval_vdn_net.load_state_dict(torch.load(path_vdn, map_location=map_location))
                print('Successfully load the model: {} and {}'.format(path_rnn, path_vdn))
            else:
                raise Exception("No model!")

        # Make target_net and eval_net network parameters the same
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_vdn_net.load_state_dict(self.eval_vdn_net.state_dict())

        self.eval_parameters = list(self.eval_vdn_net.parameters()) + list(self.eval_rnn.parameters())
        if self.args.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=self.args.lr)


        # During execution, maintain an eval_hidden for each agent
        # During learning, maintain eval_hidden and target_hidden for each agent in each episode
        self.eval_hidden = None
        self.target_hidden = None
        print('Init alg VDN')

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
        # TODO In pymarl the last experience is not taken, find out why
        u, r, avail_u, avail_u_next, terminated = batch['u'], batch['r'],  batch['avail_u'], \
                                                  batch['avail_u_next'], batch['terminated']
        mask = 1 - batch["padded"].float()  # Used to set TD-error to 0 for padded experiences, preventing them from affecting learning
        if self.args.cuda:
            u = u.cuda()
            r = r.cuda()
            mask = mask.cuda()
            terminated = terminated.cuda()
        # Get Q values for each agent, dimensions (episode count, max_episode_len, n_agents, n_actions)
        q_evals, q_targets = self.get_q_values(batch, max_episode_len)

        # Get Q value for each agent's action, and remove the last unnecessary dimension since it only has one value
        q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)

        # Get target_q
        q_targets[avail_u_next == 0.0] = - 9999999
        q_targets = q_targets.max(dim=3)[0]

        q_total_eval = self.eval_vdn_net(q_evals)
        q_total_target = self.target_vdn_net(q_targets)

        targets = r + self.args.gamma * q_total_target * (1 - terminated)

        td_error = targets.detach() - q_total_eval
        masked_td_error = mask * td_error  # Erase td_error for padded experiences

        # loss = masked_td_error.pow(2).mean()
        # Cannot use mean directly because many experiences are useless; need to sum and divide by actual experience count to get true mean
        loss = (masked_td_error ** 2).sum() / mask.sum()
        # print('Loss is ', loss)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_vdn_net.load_state_dict(self.eval_vdn_net.state_dict())

    def _get_inputs(self, batch, transition_idx):
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
            # Since current obs is 3D data where each dimension represents (episode, agent, obs dimension), we can directly add the corresponding vector on dim_1
            # For example, add (1, 0, 0, 0, 0) after agent_0, representing agent 0 out of 5 agents. Since agent_0 data is exactly in row 0, the agent ID to add
            # is exactly an identity matrix, i.e., diagonal is 1, rest is 0
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            inputs_next.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        # Concatenate the three parts in obs, and combine episode_num episodes and self.args.n_agents agents data into episode_num*n_agents rows of data
        # Since all agents share one neural network here, each data row contains its own ID, so it's still its own data
        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1)
        return inputs, inputs_next

    def get_q_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_inputs(batch, transition_idx)  # Add last_action and agent_id to obs
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
                self.target_hidden = self.target_hidden.cuda()
            q_eval, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)  # Resulting q_eval dimension is (episode_num*n_agents, n_actions)
            q_target, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden)

            # Reshape q_eval dimension back to (episode_num, n_agents, n_actions)
            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)
        # The resulting q_eval and q_target are lists containing max_episode_len arrays, each array has dimension (episode count, n_agents, n_actions)
        # Convert this list to (episode count, max_episode_len, n_agents, n_actions) array
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        return q_evals, q_targets

    def init_hidden(self, episode_num):
        # Initialize eval_hidden and target_hidden for each agent in each episode
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_vdn_net.state_dict(), self.model_dir + '/' + num + '_vdn_net_params.pkl')
        torch.save(self.eval_rnn.state_dict(),  self.model_dir + '/' + num + '_rnn_net_params.pkl')
