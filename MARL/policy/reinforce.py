import torch
import os
from MARL.network.base_net import RNN
from MARL.network.commnet import CommNet
from MARL.network.g2anet import G2ANet


class Reinforce:
    def __init__(self, args):
        # canonical args reference for this instance
        self.args = args
        self.n_actions = self.args.n_actions
        self.n_agents = self.args.n_agents
        self.state_shape = self.args.state_shape
        self.obs_shape = self.args.obs_shape
        actor_input_shape = self.obs_shape  # Actor network input dimension, same as VDN/QMIX RNN input, using the same network structure
        # Determine RNN input dimension based on parameters
        if self.args.last_action:
            actor_input_shape += self.n_actions
        if self.args.reuse_network:
            actor_input_shape += self.n_agents

        # Neural networks
        # Network for each agent to select actions; outputs probabilities for all actions of current agent; requires softmax operation again when selecting actions using these probabilities.
        if self.args.alg == 'reinforce':
            print('Init alg reinforce')
            self.eval_rnn = RNN(actor_input_shape, args)
        elif self.args.alg == 'reinforce+commnet':
            print('Init alg reinforce+commnet')
            self.eval_rnn = CommNet(actor_input_shape, args)
        elif self.args.alg == 'reinforce+g2anet':
            print('Init alg reinforce+g2anet')
            self.eval_rnn = G2ANet(actor_input_shape, args)
        else:
            raise Exception("No such algorithm")

        if self.args.cuda:
            self.eval_rnn.cuda()

        self.model_dir = self.args.model_dir + '/' + self.args.alg + '/' + self.args.map
        # Load model if it exists
        if self.args.load_model:
            if os.path.exists(self.model_dir + '/rnn_params.pkl'):
                path_rnn = self.model_dir + '/rnn_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
                print('Successfully load the model: {}'.format(path_rnn))
            else:
                raise Exception("No model!")

        self.rnn_parameters = list(self.eval_rnn.parameters())
        if self.args.optimizer == "RMS":
            self.rnn_optimizer = torch.optim.RMSprop(self.rnn_parameters, lr=self.args.lr_actor)

        # During execution, maintain an eval_hidden for each agent
        # During learning, maintain an eval_hidden for each agent in each episode
        self.eval_hidden = None

    def learn(self, batch, max_episode_len, train_step, epsilon):
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():  # Convert batch data to tensor
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        u, r, avail_u, terminated = batch['u'], batch['r'],  batch['avail_u'], batch['terminated']
        mask = (1 - batch["padded"].float())  # Used to set TD-error to 0 for padded experiences, preventing them from affecting learning
        if self.args.cuda:
            r = r.cuda()
            u = u.cuda()
            mask = mask.cuda()
            terminated = terminated.cuda()

        # Get return for each experience, (episode_num, max_episode_len, n_agents)
        n_return = self._get_returns(r, mask, terminated, max_episode_len)

        # Probabilities of all actions for each agent (episode_num, max_episode_len, n_agents, n_actions)
        action_prob = self._get_action_prob(batch, max_episode_len, epsilon)

        # Add n_agents dimension to mask for training each agent
        mask = mask.repeat(1, 1, self.n_agents)
        # Probability corresponding to the selected action of each agent (episode_num, max_episode_len, n_agents)
        pi_taken = torch.gather(action_prob, dim=3, index=u).squeeze(3)
        pi_taken[mask == 0] = 1.0  # Because we need to take log, for padded experiences all probabilities are 0, taking log would be negative infinity, so set them to 1
        log_pi_taken = torch.log(pi_taken)

        # Loss function, (episode_num, max_episode_len, n_agents)
        loss = - ((n_return * log_pi_taken) * mask).sum() / mask.sum()
        self.rnn_optimizer.zero_grad()
        loss.backward()
        if self.args.alg == 'reinforce+g2anet':
            torch.nn.utils.clip_grad_norm_(self.rnn_parameters, self.args.grad_norm_clip)
        self.rnn_optimizer.step()
        # print('Actor loss is', loss)

    def _get_returns(self, r, mask, terminated, max_episode_len):
        r = r.squeeze(-1)
        mask = mask.squeeze(-1)
        terminated = terminated.squeeze(-1)
        terminated = 1 - terminated
        n_return = torch.zeros_like(r)
        n_return[:, -1] = r[:, -1] * mask[:, -1]
        for transition_idx in range(max_episode_len - 2, -1, -1):
            n_return[:, transition_idx] = (r[:, transition_idx] + self.args.gamma * n_return[:, transition_idx + 1] * terminated[:, transition_idx]) * mask[:, transition_idx]
        return n_return.unsqueeze(-1).expand(-1, -1, self.n_agents)

    def _get_actor_inputs(self, batch, transition_idx):
        # Extract experiences at transition_idx from all episodes; u_onehot extracts all because we need the previous one
        obs, u_onehot = batch['o'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs = []
        inputs.append(obs)
        # Add last action and agent ID to inputs

        if self.args.last_action:
            if transition_idx == 0:  # If it's the first experience, set previous action to zero vector
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
        if self.args.reuse_network:
            # Since current inputs is 3D data where each dimension represents (episode ID, agent ID, inputs dimension), we can directly add the corresponding vector on dim_1
            # For example, add (1, 0, 0, 0, 0) after agent_0, representing agent 0 out of 5 agents. Since agent_0 data is exactly in row 0, the agent ID to add
            # is exactly an identity matrix, i.e., diagonal is 1, rest is 0
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        # Concatenate the three parts in inputs, and combine episode_num episodes and self.args.n_agents agents data into 40 rows of (40, 96) data,
        # Since all agents share one neural network here, each data row contains its own ID, so it's still its own data
        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_action_prob(self, batch, max_episode_len, epsilon):
        episode_num = batch['o'].shape[0]
        avail_actions = batch['avail_u']
        action_prob = []
        for transition_idx in range(max_episode_len):
            inputs = self._get_actor_inputs(batch, transition_idx)  # Add last_action and agent_id to obs
            if self.args.cuda:
                inputs = inputs.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
            # inputs dimension is (episode_num * n_agents, inputs_shape), resulting outputs dimension is (episode_num * n_agents, n_actions)
            outputs, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)
            # Reshape q_eval dimension back to (8, 5, n_actions)
            outputs = outputs.view(episode_num, self.n_agents, -1)
            prob = torch.nn.functional.softmax(outputs, dim=-1)
            action_prob.append(prob)
        # The resulting action_prob is a list containing max_episode_len arrays, each array has dimension (episode count, n_agents, n_actions)
        # Convert this list to (episode count, max_episode_len, n_agents, n_actions) array
        action_prob = torch.stack(action_prob, dim=1).cpu()

        action_num = avail_actions.sum(dim=-1, keepdim=True).float().repeat(1, 1, 1, avail_actions.shape[-1])   # Number of actions that can be selected
        action_prob = ((1 - epsilon) * action_prob + torch.ones_like(action_prob) * epsilon / action_num)
        action_prob[avail_actions == 0] = 0.0  # Probability of actions that cannot be executed is 0

        # Because we set probabilities of unavailable actions to 0 above, the probability sum is no longer 1, so we need to renormalize here. During execution Categorical normalizes automatically.
        action_prob = action_prob / action_prob.sum(dim=-1, keepdim=True)
        # Because many experiences are padded, their avail_actions are all filled with 0, so all action probabilities for those experiences are 0, which becomes nan during normalization.
        # Therefore we need to set the probabilities for those experiences to 0 again
        action_prob[avail_actions == 0] = 0.0
        if self.args.cuda:
            action_prob = action_prob.cuda()
        return action_prob

    def init_hidden(self, episode_num):
        # Initialize an eval_hidden for each agent in each episode
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_rnn.state_dict(),  self.model_dir + '/' + num + '_rnn_params.pkl')