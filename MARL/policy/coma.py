import torch
import os
from MARL.network.base_net import RNN
from MARL.network.commnet import CommNet
from MARL.network.g2anet import G2ANet
from MARL.network.coma_critic import ComaCritic
from MARL.common.utils import td_lambda_target


class COMA:
    def __init__(self, args):
        # canonical args
        self.args = args
        self.n_actions = self.args.n_actions
        self.n_agents = self.args.n_agents
        self.state_shape = self.args.state_shape
        self.obs_shape = self.args.obs_shape
        actor_input_shape = self.obs_shape  # Actor network input dimension, same as VDN/QMIX RNN input, using the same network structure
        critic_input_shape = self._get_critic_input_shape()  # Critic network input dimension
        # Determine RNN input dimension based on parameters
        if self.args.last_action:
            actor_input_shape += self.n_actions
        if self.args.reuse_network:
            actor_input_shape += self.n_agents

        # Neural networks
        # Network for each agent to select actions; outputs probabilities for all actions of current agent; requires softmax operation again when selecting actions using these probabilities.
        if self.args.alg == 'coma':
            print('Init alg coma')
            self.eval_rnn = RNN(actor_input_shape, args)
        elif self.args.alg == 'coma+commnet':
            print('Init alg coma+commnet')
            self.eval_rnn = CommNet(actor_input_shape, args)
        elif self.args.alg == 'coma+g2anet':
            print('Init alg coma+g2anet')
            self.eval_rnn = G2ANet(actor_input_shape, args)
        else:
            raise Exception("No such algorithm")

        # Get joint Q values for all executable actions of current agent; these Q values need to be combined with actor network output probabilities to calculate advantage
        self.eval_critic = ComaCritic(critic_input_shape, self.args)
        self.target_critic = ComaCritic(critic_input_shape, self.args)

        if self.args.cuda:
            self.eval_rnn.cuda()
            self.eval_critic.cuda()
            self.target_critic.cuda()

        self.model_dir = self.args.model_dir + '/' + self.args.alg + '/' + self.args.map
        # Load model if it exists
        if self.args.load_model:
            if os.path.exists(self.model_dir + '/rnn_params.pkl'):
                path_rnn = self.model_dir + '/rnn_params.pkl'
                path_coma = self.model_dir + '/critic_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
                self.eval_critic.load_state_dict(torch.load(path_coma, map_location=map_location))
                print('Successfully load the model: {} and {}'.format(path_rnn, path_coma))
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

    def _get_critic_input_shape(self):
        # state
        input_shape = self.state_shape  # 48
        # obs
        input_shape += self.obs_shape  # 30
        # agent_id
        input_shape += self.n_agents  # 3
        # Current actions and last actions of all agents
        input_shape += self.n_actions * self.n_agents * 2  # 54

        return input_shape

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
        # Calculate Q values for each agent based on experiences to update Critic network. Then calculate execution probabilities of each action to calculate advantage to update Actor.
        q_values = self._train_critic(batch, max_episode_len, train_step)  # Train critic network and get Q values for all actions of each agent
        action_prob = self._get_action_prob(batch, max_episode_len, epsilon)  # Probabilities of all actions for each agent

        q_taken = torch.gather(q_values, dim=3, index=u).squeeze(3)  # Q value corresponding to the selected action of each agent
        pi_taken = torch.gather(action_prob, dim=3, index=u).squeeze(3)  # Probability corresponding to the selected action of each agent
        pi_taken[mask == 0] = 1.0  # Because we need to take log, for padded experiences all probabilities are 0, taking log would be negative infinity, so set them to 1
        log_pi_taken = torch.log(pi_taken)

        # Calculate advantage
        baseline = (q_values * action_prob).sum(dim=3, keepdim=True).squeeze(3).detach()
        advantage = (q_taken - baseline).detach()
        loss = - ((advantage * log_pi_taken) * mask).sum() / mask.sum()
        self.rnn_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.rnn_parameters, self.args.grad_norm_clip)
        self.rnn_optimizer.step()
        # print('Training: loss is', loss.item())
        # print('Training: critic params')
        # for params in self.eval_critic.named_parameters():
        #     print(params)
        # print('Training: actor params')
        # for params in self.eval_rnn.named_parameters():
        #     print(params)

    def _get_critic_inputs(self, batch, transition_idx, max_episode_len):
        # Extract experiences at transition_idx from all episodes
        obs, obs_next, s, s_next = batch['o'][:, transition_idx], batch['o_next'][:, transition_idx],\
                                   batch['s'][:, transition_idx], batch['s_next'][:, transition_idx]
        u_onehot = batch['u_onehot'][:, transition_idx]
        if transition_idx != max_episode_len - 1:
            u_onehot_next = batch['u_onehot'][:, transition_idx + 1]
        else:
            u_onehot_next = torch.zeros(*u_onehot.shape)
        # s and s_next are 2D without n_agents dimension because all agents have same s. Others are 3D and cannot be concatenated, so convert s to 3D
        s = s.unsqueeze(1).expand(-1, self.n_agents, -1)
        s_next = s_next.unsqueeze(1).expand(-1, self.n_agents, -1)
        episode_num = obs.shape[0]
        # Because COMA critic uses all agents' actions, need to convert u_onehot last dimension from current agent's action to all agents' actions
        u_onehot = u_onehot.view((episode_num, 1, -1)).repeat(1, self.n_agents, 1)
        u_onehot_next = u_onehot_next.view((episode_num, 1, -1)).repeat(1, self.n_agents, 1)

        if transition_idx == 0:  # If it's the first experience, set previous action to zero vector
            u_onehot_last = torch.zeros_like(u_onehot)
        else:
            u_onehot_last = batch['u_onehot'][:, transition_idx - 1]
            u_onehot_last = u_onehot_last.view((episode_num, 1, -1)).repeat(1, self.n_agents, 1)

        inputs, inputs_next = [], []
        # Add state
        inputs.append(s)
        inputs_next.append(s_next)
        # Add obs
        inputs.append(obs)
        inputs_next.append(obs_next)
        # Add last actions of all agents
        inputs.append(u_onehot_last)
        inputs_next.append(u_onehot)

        # Add current actions
        '''
        For COMA current actions, input is other agents' current actions, not current agent's action. For convenience, although we input current agent's
        current action, we set it to zero vector, effectively not inputting it.
        '''
        action_mask = (1 - torch.eye(self.n_agents))  # torch.eye() generates a 2D diagonal matrix
        # Get a matrix action_mask to convert each agent's own action to zero vector in (episode_num, n_agents, n_agents * n_actions) actions
        action_mask = action_mask.view(-1, 1).repeat(1, self.n_actions).view(self.n_agents, -1)
        inputs.append(u_onehot * action_mask.unsqueeze(0))
        inputs_next.append(u_onehot_next * action_mask.unsqueeze(0))

        # Add agent ID corresponding one-hot vector
        '''
        Since current inputs is 3D data where each dimension represents (episode ID, agent ID, inputs dimension), we can directly add the corresponding vector at the end
        For example, add (1, 0, 0, 0, 0) after agent_0, representing agent 0 out of 5 agents. Since agent_0 data is exactly in row 0, the agent ID to add
        is exactly an identity matrix, i.e., diagonal is 1, rest is 0
        '''
        inputs.append(torch.eye(self.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        inputs_next.append(torch.eye(self.n_agents).unsqueeze(0).expand(episode_num, -1, -1))

        # Concatenate the 5 inputs items, and convert dimensions from (episode_num, n_agents, inputs) 3D to (episode_num * n_agents, inputs) 2D
        inputs = torch.cat([x.reshape(episode_num * self.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num * self.n_agents, -1) for x in inputs_next], dim=1)
        return inputs, inputs_next

    def _get_q_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_critic_inputs(batch, transition_idx, max_episode_len)
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
            # Neural network input is (episode_num * n_agents, inputs) 2D data, output is (episode_num * n_agents, n_actions) 2D data
            q_eval = self.eval_critic(inputs)
            q_target = self.target_critic(inputs_next)

            # Reshape q values dimension back to (episode_num, n_agents, n_actions)
            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)
        # The resulting q_evals and q_targets are lists containing max_episode_len arrays, each array has dimension (episode count, n_agents, n_actions)
        # Convert this list to (episode count, max_episode_len, n_agents, n_actions) array
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        return q_evals, q_targets

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
        u, r, avail_u, terminated = batch['u'], batch['r'], batch['avail_u'], batch['terminated']
        u_next = u[:, 1:]
        padded_u_next = torch.zeros(*u[:, -1].shape, dtype=torch.long).unsqueeze(1)
        u_next = torch.cat((u_next, padded_u_next), dim=1)
        mask = (1 - batch["padded"].float()).repeat(1, 1, self.n_agents)  # Used to set TD-error to 0 for padded experiences to prevent them from affecting learning
        if self.args.cuda:
            u = u.cuda()
            u_next = u_next.cuda()
            mask = mask.cuda()
        # Get Q values for each agent, dimensions (episode count, max_episode_len, n_agents, n_actions)
        # q_next_target is Q value output by target network for next state-action pair, not including reward
        q_evals, q_next_target = self._get_q_values(batch, max_episode_len)
        q_values = q_evals.clone()  # Return at end of function, used to calculate advantage to update actor
        # Get Q value for each agent's action, and remove the last unnecessary dimension since it only has one value

        q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)
        q_next_target = torch.gather(q_next_target, dim=3, index=u_next).squeeze(3)
        targets = td_lambda_target(batch, max_episode_len, q_next_target.cpu(), self.args)
        if self.args.cuda:
            targets = targets.cuda()
        td_error = targets.detach() - q_evals
        masked_td_error = mask * td_error  # Erase td_error for padded experiences

        # Cannot use mean directly because many experiences are useless; need to sum and divide by actual experience count to get true mean
        loss = (masked_td_error ** 2).sum() / mask.sum()
        # print('Loss is ', loss)
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_parameters, self.args.grad_norm_clip)
        self.critic_optimizer.step()
        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_critic.load_state_dict(self.eval_critic.state_dict())
        return q_values

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_critic.state_dict(), self.model_dir + '/' + num + '_critic_params.pkl')
        torch.save(self.eval_rnn.state_dict(),  self.model_dir + '/' + num + '_rnn_params.pkl')
