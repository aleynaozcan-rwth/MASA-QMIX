import torch
import os
from MARL.network.base_net import RNN, Critic
from MARL.network.commnet import CommNet
from MARL.network.g2anet import G2ANet


class CentralV:
    def __init__(self, args):
        # canonical args for this instance
        self.args = args
        self.n_actions = self.args.n_actions
        self.n_agents = self.args.n_agents
        self.state_shape = self.args.state_shape
        self.obs_shape = self.args.obs_shape
        actor_input_shape = self.obs_shape  # Actor network input dimension, same as VDN/QMIX RNN input, using the same network structure
        critic_input_shape = self.state_shape  # Critic network input dimension
        # Determine RNN input dimension based on parameters
        if self.args.last_action:
            actor_input_shape += self.n_actions
        if self.args.reuse_network:
            actor_input_shape += self.n_agents

        # Neural networks
        # Network for each agent to select actions; outputs probabilities for all actions of current agent; requires softmax operation again when selecting actions using these probabilities.
        if self.args.alg == 'central_v':
            self.eval_rnn = RNN(actor_input_shape, args)
            print('Init alg central_v')
        elif self.args.alg == 'central_v+commnet':
            self.eval_rnn = CommNet(actor_input_shape, args)
            print('Init alg central_v+commnet')
        elif self.args.alg == 'central_v+g2anet':
            print('Init alg central_v+g2anet')
            self.eval_rnn = G2ANet(actor_input_shape, args)
        else:
            raise Exception("No such algorithm")

        self.eval_critic = Critic(critic_input_shape, self.args)
        self.target_critic = Critic(critic_input_shape, self.args)

        if self.args.cuda:
            self.eval_rnn.cuda()
            self.eval_critic.cuda()
            self.target_critic.cuda()

        self.model_dir = self.args.model_dir + '/' + self.args.alg + '/' + self.args.map
        # Load model if it exists
        if self.args.load_model:
            if os.path.exists(self.model_dir + '/rnn_params.pkl'):
                path_rnn = self.model_dir + '/rnn_params.pkl'
                path_critic = self.model_dir + '/critic_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
                self.eval_critic.load_state_dict(torch.load(path_critic, map_location=map_location))
                print('Successfully load the model: {} and {}'.format(path_rnn, path_critic))
            else:
                raise Exception("No model!")

        # Make target_net and eval_net network parameters the same
        self.target_critic.load_state_dict(self.eval_critic.state_dict())

        self.rnn_parameters = list(self.eval_rnn.parameters())
        self.critic_parameters = list(self.eval_critic.parameters())

        if self.args.optimizer == "RMS":
            self.critic_optimizer = torch.optim.RMSprop(self.critic_parameters, lr=self.args.lr_critic)
            self.rnn_optimizer = torch.optim.RMSprop(self.rnn_parameters, lr=self.args.lr_actor)

        # During execution, maintain an eval_hidden for each agent
        # During learning, maintain an eval_hidden for each agent in each episode
        self.eval_hidden = None

    def learn(self, batch, max_episode_len, train_step, epsilon):  # train_step represents the learning step number, used to control updating target_net network parameters
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():  # Convert batch data to tensor
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        u, r, avail_u, terminated = batch['u'], batch['r'],  batch['avail_u'], batch['terminated']
        mask = (1 - batch["padded"].float()).repeat(1, 1, self.n_agents)  # Used to set TD-error to 0 for padded experiences, preventing them from affecting learning
        if self.args.cuda:
            u = u.cuda()
            mask = mask.cuda()

        # Train critic network and get td_error for each experience, (episode_num, max_episode_len, 1)
        td_error = self._train_critic(batch, max_episode_len, train_step)
        td_error = td_error.repeat(1, 1, self.n_agents)

        # Probabilities of all actions for each agent (episode_num, max_episode_len, n_agents, n_actions)
        action_prob = self._get_action_prob(batch, max_episode_len, epsilon)

        # Probability corresponding to the selected action of each agent (episode_num, max_episode_len, n_agents)
        pi_taken = torch.gather(action_prob, dim=3, index=u).squeeze(3)
        pi_taken[mask == 0] = 1.0  # Because we need to take log, for padded experiences all probabilities are 0, taking log would be negative infinity, so set them to 1
        log_pi_taken = torch.log(pi_taken)

        # Loss function, (episode_num, max_episode_len, n_agents)
        loss = - ((td_error.detach() * log_pi_taken) * mask).sum() / mask.sum()
        self.rnn_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.rnn_parameters, self.args.grad_norm_clip)
        self.rnn_optimizer.step()
        # print('Actor loss is', loss)

    def _get_v_values(self, batch, max_episode_len):
        v_evals, v_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = batch['s'][:, transition_idx], batch['s_next'][:, transition_idx],
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
            # Neural network input is (episode_num, state_shape) 2D data, output is (episode_num, 1) 2D data
            v_eval = self.eval_critic(inputs)
            v_target = self.target_critic(inputs_next)

            v_evals.append(v_eval)
            v_targets.append(v_target)
        v_evals = torch.stack(v_evals, dim=1)  # (episode_num, max_episode_len, 1)
        v_targets = torch.stack(v_targets, dim=1)
        return v_evals, v_targets

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
        avail_actions = batch['avail_u']  # COMA doesn't use target_actor, so no need for next available actions of the last obs
        action_prob = []
        for transition_idx in range(max_episode_len):
            inputs = self._get_actor_inputs(batch, transition_idx)  # Add last_action and agent_id to obs
            if self.args.cuda:
                inputs = inputs.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
            outputs, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)  # inputs dimension is (40, 96), resulting q_eval dimension is (40, n_actions)
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

    def _train_critic(self, batch, max_episode_len, train_step):
        r, terminated = batch['r'], batch['terminated']
        mask = (1 - batch["padded"].float()).repeat(1, 1, self.n_agents)  # Used to set TD-error to 0 for padded experiences, preventing them from affecting learning
        if self.args.cuda:
            mask = mask.cuda()
            r = r.cuda()
            terminated = terminated.cuda()
        v_evals, v_next_target = self._get_v_values(batch, max_episode_len)

        targets = r + self.args.gamma * v_next_target * (1 - terminated)
        td_error = targets.detach() - v_evals
        masked_td_error = mask * td_error  # Erase td_error for padded experiences

        # Cannot use mean directly because many experiences are useless; need to sum and divide by actual experience count to get true mean
        loss = (masked_td_error ** 2).sum() / mask.sum()
        # print('Critic Loss is ', loss)
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_parameters, self.args.grad_norm_clip)
        self.critic_optimizer.step()
        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_critic.load_state_dict(self.eval_critic.state_dict())
        return td_error

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_critic.state_dict(), self.model_dir + '/' + num + '_critic_params.pkl')
        torch.save(self.eval_rnn.state_dict(),  self.model_dir + '/' + num + '_rnn_params.pkl')
