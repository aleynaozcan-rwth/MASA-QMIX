import inspect
import functools
import torch


def store_args(method):
    """Stores provided method args as instance attributes."""
    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        # Get default arg values
        args = defaults.copy()
        # Add provided arg values
        for name, value in zip(arg_names, positional_args[1:]):
            args[name] = value
        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper


def td_lambda_target(batch, max_episode_len, q_targets, args):
    # batch['o'].shape = (episode_num, max_episode_len, n_agents, n_actions)
    # q_targets.shape  = (episode_num, max_episode_len, n_agents)
    episode_num = batch['o'].shape[0]

    # mask: 1 for real timesteps, 0 for padded ones (broadcast to agents)
    mask = (1 - batch["padded"].float()).repeat(1, 1, args.n_agents)
    # terminated flag inverted here so that ongoing steps keep targets (1 if not terminated)
    terminated = (1 - batch["terminated"].float()).repeat(1, 1, args.n_agents)
    # rewards broadcast to agents
    r = batch['r'].repeat((1, 1, args.n_agents))

    # -------------------------------------------------- n_step_return -------------------------------------------------
    """
    1) Each transition has multiple n-step returns; we allocate a last dimension
       of size max_episode_len to store them. Index n stores the (n+1)-step return.
    2) Because different episodes have different effective lengths, we use `mask`
       to zero out extra n-step returns for padded steps. This must be done here,
       before computing lambda-returns; otherwise, even if we mask after the
       TD-error, lambda mixing would already be contaminated.
    3) `terminated` is used to zero out q_targets and rewards beyond the true
       episode end at each trajectory.
    """
    n_step_return = torch.zeros((episode_num, max_episode_len, args.n_agents, max_episode_len))

    for transition_idx in range(max_episode_len - 1, -1, -1):
        # First compute the 1-step return:
        n_step_return[:, transition_idx, :, 0] = (
            r[:, transition_idx]
            + args.gamma * q_targets[:, transition_idx] * terminated[:, transition_idx]
        ) * mask[:, transition_idx]

        # For transition `transition_idx`, there are (max_episode_len - transition_idx) possible n-step returns.
        # Note: index n corresponds to (n+1)-step.
        for n in range(1, max_episode_len - transition_idx):
            # n-step target at time t:
            #   G_t^{(n+1)} = r_t + gamma * G_{t+1}^{(n)}
            # Special case n==1 above already uses Q at t+1 (handled via previous assignment).
            n_step_return[:, transition_idx, :, n] = (
                r[:, transition_idx] + args.gamma * n_step_return[:, transition_idx + 1, :, n - 1]
            ) * mask[:, transition_idx]
    # -------------------------------------------------- n_step_return -------------------------------------------------

    # -------------------------------------------------- lambda return -------------------------------------------------
    """
    lambda_return.shape = (episode_num, max_episode_len, n_agents)
    """
    lambda_return = torch.zeros((episode_num, max_episode_len, args.n_agents))

    for transition_idx in range(max_episode_len):
        returns = torch.zeros((episode_num, args.n_agents))
        # Accumulate geometrically-weighted n-step returns (1-step up to (T - t)-step)
        for n in range(1, max_episode_len - transition_idx):
            returns += (args.td_lambda ** (n - 1)) * n_step_return[:, transition_idx, :, n - 1]

        # Final term uses the longest available return at this t
        lambda_return[:, transition_idx] = (
            (1 - args.td_lambda) * returns
            + (args.td_lambda ** (max_episode_len - transition_idx - 1))
            * n_step_return[:, transition_idx, :, max_episode_len - transition_idx - 1]
        )
    # -------------------------------------------------- lambda return -------------------------------------------------

    return lambda_return
"""# --------------------------------------------------------------------------------
# td_lambda_target (clean explanation):
#
# Purpose:
#   - Compute the training targets (TD(λ) returns) for agents in reinforcement
#     learning, considering variable episode lengths and early terminations.
#
# Step 1: n-step returns
#   - At each time t, we can look ahead 1 step, 2 steps, ... until the end.
#   - Example:
#       1-step: reward at t + γ * Q at (t+1)
#       2-step: reward at t + γ * reward at (t+1) + γ^2 * Q at (t+2)
#       n-step: reward at t + γ * ... up to Q at (t+n)
#   - This captures "what happens if I only look ahead n steps".
#
# Step 2: λ-return
#   - Instead of choosing just one horizon (1-step OR full return),
#     we mix all n-step returns together.
#   - The mixing uses λ as a balance factor:
#       - Small λ (close to 0): focus more on short-term (less variance).
#       - Large λ (close to 1): focus more on long-term (less bias).
#   - Formula (simplified):
#       G_t^λ = (1 - λ) * [1-step + λ*2-step + λ^2*3-step + ...]
#
# Step 3: Masking and termination
#   - Some episodes are shorter, so we "pad" them with fake steps.
#   - `mask` makes sure padded steps count as 0.
#   - `terminated` makes sure that once an episode ends, no future reward is added.
#
# Output:
#   - A tensor lambda_return with shape (episode_num, max_episode_len, n_agents).
#   - This gives each agent a target value at every timestep.
#
# In short:
#   - n-step returns = "look ahead exactly n steps".
#   - λ-return = "blend all n-step returns with λ as a balance knob".
#   - This helps stabilize learning by balancing short-term and long-term views.
# --------------------------------------------------------------------------------"""
#--------------------------------------------------------------