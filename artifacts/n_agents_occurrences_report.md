n_agents occurrences report

Summary:
- Total grep hits found: ~200 (many files across MARL/, utils/, tests/).
- Major usage categories:
  1) CLI / argument default (source of default=10)
  2) Environment parameter / initial jobs (MASAEnv.num_jobs and job generation)
  3) Runner runtime auto-set / propagation to buffer/policies
  4) Rollout / transition padding and validation
  5) ReplayBuffer allocation and shaping
  6) Agent / Policy / Network sizing (QMIX, MAVEN, VDN, QTRAN, CentralV, etc.)
  7) Utility functions & training helpers (td-lambda targets, masking)
  8) Tests and fixtures

Key files (with short context and classification):

1) MARL/common/arguments.py  (CLI default)
- Relevant line:
    parser.add_argument('--n_agents', type=int, default=10)
- Classification: CLI default for agent count. This is the origin of "Initial Jobs = 10" when no environment override is provided.

2) environment.py  (env param / initial jobs)
- Relevant lines:
    self.num_jobs = _resolve(('n_agents', 'num_jobs'), int, default=0)

    def get_env_info(self):
        return {
            "n_actions": n_actions,
            "n_agents": int(self.num_jobs),
            "state_shape": int(self.state_dim),
            "obs_shape": int(self.obs_dim_agent),
            "episode_limit": int(self.episode_limit),
        }

- Classification: MASAEnv consumes `n_agents` / `num_jobs` and uses it as the authoritative initial job count (self.num_jobs). _generate_initial_jobs() iterates range(self.num_jobs) to create initial JobAgent instances.

3) MARL/runner.py  (runtime auto-set & propagation)
- Relevant lines:
    if 'n_agents' in info and info.get('n_agents') is not None:
        run_args.n_agents = int(info.get('n_agents'))

    # fallback: len(env.jobs)
    run_args.n_agents = int(len(getattr(self.env, 'jobs')))

    # final fallback: env.num_jobs
    run_args.n_agents = int(getattr(self.env, 'num_jobs'))

- Classification: Runner is responsible for setting the runtime `run_args.n_agents` used by Agents/policies and the ReplayBuffer. It treats the CLI default (10) as "unset" and will override it with env-provided counts when available.

4) MARL/common/rollout.py  (padding/truncation & validation)
- Relevant lines:
    # pad/truncate to args.n_agents
    n_agents = getattr(args, 'n_agents', len(u_list) if u_list else 1)
    if len(u_list) < n_agents:
        u_list = u_list + [0] * (n_agents - len(u_list))
    elif len(u_list) > n_agents:
        u_list = u_list[:n_agents]

    # prepare obs array sized by n_agents
    o_arr = np.zeros((n_agents, obs_dim if obs_dim is not None else 1), dtype=np.float32)

    # dev-only runtime shape assertions
    n_agents_expected = getattr(args, 'n_agents', None)

- Classification: RolloutWorker uses `args.n_agents` to pad/truncate action/obs arrays produced by the environment when building per-decision transitions. It also validates transition shapes against expected n_agents when debug flags are enabled.

5) MARL/common/replay_buffer.py  (batch allocation / shaping)
- Relevant lines:
    def __init__(..., n_agents: Optional[int] = None, ...):
        self._n_agents = int(n_agents) if n_agents is not None else None

    # allocate tensors
    o      = np.zeros((B, T, n_agents, obs_dim), dtype=np.float32)
    u      = np.zeros((B, T, n_agents, 1), dtype=np.int64)
    avail_u = np.zeros((B, T, n_agents, max_avail_len), dtype=np.float32)

- Classification: ReplayBuffer requires a stable `n_agents` to allocate batch arrays. It can accept it via constructor or infer it from episodes. Many helper functions (_as_agents_obs/_as_agents_mask) take n_agents to pad/trim.

6) MARL/agent and MARL/policy modules (QMIX, MAVEN, CentralV, etc.)  (policy/network sizing)
- Examples:
    - `Agents.__init__`: self.n_agents = self.args.n_agents
    - `QMIX.__init__`: self.n_agents = self.args.n_agents
    - `QMIX._get_inputs_t`: eye = torch.eye(self.n_agents).unsqueeze(0).expand(episode_num, -1, -1)
    - `MAVEN.init_hidden`: self.eval_hidden = torch.zeros((episode_num, self.n_agents, ...))
    - `qtran_net.py`: episode_num, max_episode_len, n_agents, n_actions = actions.shape

- Classification: `n_agents` is used heavily to size neural-network inputs, to reshape per-agent tensors, create identity agent-ID matrices, and initialize hidden states. This is the dominant usage of `n_agents` across MARL.

7) MARL/common/utils.py  (training helpers)
- Example:
    mask = (1 - batch["padded"].float()).repeat(1, 1, args.n_agents)
    n_step_return = torch.zeros((episode_num, max_episode_len, args.n_agents, max_episode_len))

- Classification: utility functions compute per-agent tensors and depend on `args.n_agents` for broadcasting and shape creation.

8) Tests (smoke/unit tests)
- `tests/test_rollout_transitions.py` defines Args(n_agents=2, ...), tests that shapes match args.n_agents.
- `tests/test_simpy_integration.py` returns get_env_info with n_agents=2 in fixtures.

- Classification: tests assume `n_agents` controls observed shapes and provide small deterministic values.

Other findings and notes:
- Legacy "plane" flags (start_planes / max_planes) were searched earlier and are largely argument definitions/term maps or archived; they are not active runtime replacements for `n_agents`.
- The canonical source-of-truth for "initial job count" is MASAEnv.num_jobs (resolved from `n_agents` or `num_jobs`). The CLI default (`--n_agents=10`) only becomes the initial job count if the Runner / environment do not override it; Runner attempts to adopt env.get_env_info() / len(env.jobs) first.

Recommended quick answers:
- Where does "Initial Jobs = 10" come from? → `MARL/common/arguments.py` (`--n_agents` default=10), which is used unless the environment or Runner overrides it.
- Is `n_agents` used to create jobs? → Indirectly. Environment uses `n_agents`/`num_jobs` to set `self.num_jobs` and generates jobs. Runner will prefer to auto-set `run_args.n_agents` from env.get_env_info() (n_agents), len(env.jobs), or env.num_jobs.

What I saved for you:
- A concise report (this file) summarizing usage and showing key contexts: `artifacts/n_agents_occurrences_report.md`

Next steps I can take (pick one):
1) Produce a complete exhaustive file that lists every grep hit (file, line, and ±5-line context) for all ~200 matches and save it to `artifacts/n_agents_full_context.txt`. This is noisy but reproducible. (I can create this immediately.)
2) Proceed to refactor suggestions to separate "initial_job_count" vs "policy_n_agents" (if you want to split semantics). I can draft a minimal change-set to make `--n_agents` only affect policy shapes and introduce a new `--initial_jobs` flag used by MASAEnv instead. This is a breaking change and I'd implement it in small steps with tests.
3) If you want the per-file classification in a different format (CSV / JSON), I can create and commit it here.

Which would you like me to do next? If you'd like the exhaustive per-line context file, I'll generate it now and attach the path in the repo.
