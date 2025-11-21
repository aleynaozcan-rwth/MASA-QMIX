import io
import os
import re


def test_runner_does_not_read_initial_jobs_file_for_logging():
    """Static safety check: scan the repository for any read access to initial_jobs.txt.

    The test fails if any line in the repo opens the initial_jobs file with a
    read mode ("r" or "r+"). This enforces append-only behavior for
    `my_data_and_graph/historydata/initial_jobs.txt`.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # check for read-open usage of both initial_jobs.txt and env_config_dump.json
    pattern = re.compile(r"open\([^)]*(initial_jobs(?:\.txt)?|env_config_dump\.json)[^)]*['\"]r\+?['\"]")
    found = []
    for root, _, files in os.walk(repo_root):
        # skip common large or non-source folders
        if any(p in root for p in ['.git', '__pycache__', 'artifacts']):
            continue
        for fname in files:
            if fname.endswith(('.py', '.sh', '.md', '.json')):
                path = os.path.join(root, fname)
                # skip scanning this test file to avoid self-match
                try:
                    if os.path.abspath(path) == os.path.abspath(__file__):
                        continue
                except Exception as e:
                    logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
                try:
                    txt = io.open(path, 'r', encoding='utf-8', errors='ignore').read()
                except Exception as e:
                    logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
                    continue
                for i, line in enumerate(txt.splitlines(), 1):
                    if (('initial_jobs' in line or 'env_config_dump' in line) and 'open' in line and ("'r'" in line or '"r"' in line or "'r+" in line or '"r+' in line)):
                        found.append((path, i, line.strip()))

    assert not found, f"Found attempts to open initial_jobs.txt or env_config_dump.json for reading: {found}"
