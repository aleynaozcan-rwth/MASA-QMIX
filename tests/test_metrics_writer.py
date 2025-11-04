import os
import shutil
from MARL.common.arguments import get_smoke_args
from environment import MASAEnv
from MARL.runner import Runner


def cleanup_metrics(path):
    if os.path.exists(path):
        os.remove(path)


def test_metrics_writer_tempdir(tmp_path):
    # Run a tiny runner with 3 episodes and assert metrics file has 3 rows + header
    hist = os.path.join(os.getcwd(), 'my_data_and_graph', 'historydata')
    met = os.path.join(hist, 'learning_metrics.csv')
    # backup if exists
    if os.path.exists(met):
        shutil.copy(met, met + '.testbak')
        os.remove(met)

    args = get_smoke_args()

    env = MASAEnv(args=args)
    runner = Runner(env, args)
    # run; should create metrics file with header + 3 rows
    runner.run(0)

    assert os.path.exists(met), 'learning_metrics.csv not created'
    with open(met, 'r') as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    # first line is header
    assert len(lines) == 1 + 3, f'Expected 1 header + 3 episode rows, got {len(lines)} lines'
    # unique episode lines
    ep_lines = lines[1:]
    assert len(set(ep_lines)) == 3, 'Episode rows are not unique'

    # cleanup: restore backup if present
    if os.path.exists(met + '.testbak'):
        shutil.move(met + '.testbak', met)
    else:
        try:
            os.remove(met)
        except Exception:
            pass


if __name__ == '__main__':
    # run directly for convenience
    try:
        test_metrics_writer_tempdir(None)
        print('TEST PASSED')
    except AssertionError as e:
        print('TEST FAILED:', e)
        raise
