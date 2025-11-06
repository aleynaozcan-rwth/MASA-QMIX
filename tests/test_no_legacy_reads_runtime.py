import builtins
from types import SimpleNamespace
import os

TARGETS = ['initial_jobs.txt', 'env_config_dump.json']

class OpenGuard:
    def __init__(self, orig_open):
        self.orig_open = orig_open

    def __call__(self, file, mode='r', *args, **kwargs):
        try:
            fpath = os.fspath(file)
        except Exception:
            fpath = str(file)
        if any(t in os.path.basename(fpath) for t in TARGETS) and ('r' in mode):
            raise AssertionError(f"Attempted to read legacy file during runtime: {fpath} mode={mode}")
        return self.orig_open(file, mode, *args, **kwargs)


def test_no_runtime_reads_of_legacy_files(monkeypatch):
    """Runtime test: fail if any code path reads the legacy files during env/runner init."""
    # Import modules by filepath first (so import-time reads are not part of this runtime check
    # and to avoid package import errors in different test runners).
    import importlib.util
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    import sys
    sys.path.insert(0, repo_root)
    env_path = os.path.join(repo_root, 'environment.py')
    spec = importlib.util.spec_from_file_location('env_module', env_path)
    env_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(env_mod)
    MASAEnv = getattr(env_mod, 'MASAEnv')

    # We only need to verify that constructing the environment and calling
    # reset/start does not read the legacy files. Import-time operations are
    # not part of this test (they are covered by static checks).
    orig_open = builtins.open
    monkeypatch.setattr(builtins, 'open', OpenGuard(orig_open))

    # Construct a minimal MASAEnv without enabling dumps or arrivals and call reset
    env = MASAEnv(auto_build=False, auto_start_arrivals=False, dump_config=False)
    env.reset()
    # If construction completed without AssertionError, test passes
    assert True
