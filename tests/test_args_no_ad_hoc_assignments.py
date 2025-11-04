import re
import os

# This test prevents adding ad-hoc `args.<field> = ...` assignments outside
# the central arguments module and a small set of permitted files.

ROOT = os.path.dirname(os.path.dirname(__file__))

# Files (relative to repo root) where explicit `args.x =` assignments are
# allowed: central argument definitions and a few runtime entrypoints/tools
# that intentionally set runtime overrides.
WHITELIST = {
    # Only the central arguments module is allowed to assign defaults.
    os.path.normpath('MARL/common/arguments.py'),
}

# Match a true assignment to an args attribute (avoid matching '==' equality checks)
PATTERN = re.compile(r"\bargs\.[A-Za-z_][A-Za-z0-9_]*\s*=(?!=)")


def iter_py_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        # skip virtualenvs and caches
        if any(p in dirpath for p in ('.venv', 'venv', '__pycache__', '.git')):
            continue
        for fn in filenames:
            if not fn.endswith('.py'):
                continue
            yield os.path.join(dirpath, fn)


def test_no_ad_hoc_args_assignments():
    failures = []
    for path in iter_py_files(ROOT):
        rel = os.path.relpath(path, ROOT)
        rel_norm = os.path.normpath(rel)
        # allow anything under tests/ to create args fixtures for isolation
        if rel_norm.startswith(os.path.normpath('tests') + os.sep):
            continue
        if rel_norm in WHITELIST:
            continue

        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, start=1):
                if PATTERN.search(line):
                    failures.append(f"{rel}:{i}: {line.strip()}")

    if failures:
        msg = (
            "Found ad-hoc `args.<field> = ...` assignments outside the allowed whitelist.\n"
            "Move defaults to MARL/common/arguments.py or add the file to the whitelist if it's a deliberate runtime override.\n\n"
            + "Occurrences:\n"
            + "\n".join(failures)
        )
        raise AssertionError(msg)
