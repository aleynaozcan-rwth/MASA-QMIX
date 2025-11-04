import pytest


def test_taskgenerator_start_called_on_reset(monkeypatch):
    """Ensure MASAEnv.reset() constructs a TaskGenerator and calls start(env, lam)

    We monkeypatch utils.task_generator.TaskGenerator with a fake implementation
    that records whether start() was invoked and with which arguments. The
    environment should create the TaskGenerator when config['task_generator']
    contains a positive 'arrival_lambda' and then call its start() method with
    the SimPy env and lambda value.
    """
    # Import inside the test so monkeypatching works on the module path
    from environment import MASAEnv

    calls = {}

    class FakeTaskGenerator:
        def __init__(self, config_path=None):
            # record construction
            calls['constructed'] = True
            calls['config_path'] = config_path
            self.start_args = None

        def start(self, env, lam):
            # record start invocation and args
            calls['started'] = True
            calls['start_args'] = (env, lam)

    # Patch the TaskGenerator used by environment.reset()
    monkeypatch.setattr('utils.task_generator.TaskGenerator', FakeTaskGenerator)

    # Create env without a task generator initially
    env = MASAEnv(num_jobs=0)
    # Attach a config specifying a positive arrival lambda
    env.config = {'task_generator': {'arrival_lambda': 3.5}}
    env.config_path = 'dummy_cfg'
    # Ensure no pre-existing _task_generator to force construction on reset()
    env._task_generator = None

    # Call reset(), which should create the FakeTaskGenerator and call start()
    env.reset()

    assert calls.get('constructed', False) is True, "TaskGenerator was not constructed"
    assert calls.get('started', False) is True, "TaskGenerator.start was not invoked"
    tg = getattr(env, '_task_generator', None)
    assert tg is not None and isinstance(tg, FakeTaskGenerator)
    # ensure the correct lambda and env were passed (env.env is the SimPy env)
    start_env, start_lam = calls['start_args']
    assert start_lam == pytest.approx(3.5)
    assert start_env is env.env
