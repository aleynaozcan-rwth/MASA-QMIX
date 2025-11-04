import os
import re


ROOT = os.path.dirname(os.path.dirname(__file__))


def iter_py_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        # skip virtual envs, .git, and python caches
        if any(part.startswith(".") for part in dirpath.split(os.sep)):
            # allow files under repo root that are dot-prefixed only if needed; keep simple
            pass
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            yield path


def test_no_args_field_assignments_outside_arguments_and_tests():
    """Fail if any file outside MARL/common/arguments.py or tests/ assigns to args.<field> ="""
    # detect assignment to args.<field> but avoid matching '==' or '>=' etc.
    assignment_re = re.compile(r"(?<!self\.)\bargs\.[A-Za-z_][A-Za-z0-9_]*\s*=(?!=)")
    violations = []
    for path in iter_py_files(ROOT):
        # allow the canonical arguments file and anything under tests/
        rel = os.path.relpath(path, ROOT)
        if rel == os.path.join("MARL", "common", "arguments.py"):
            continue
        if rel.startswith("tests"):
            continue
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        # compute triple-quoted string spans so we can ignore matches inside docstrings or multiline strings
        triple_spans = []
        for triple in ("'''", '"""'):
            start = 0
            while True:
                si = text.find(triple, start)
                if si == -1:
                    break
                sj = text.find(triple, si + 3)
                if sj == -1:
                    break
                triple_spans.append((si, sj + 3))
                start = sj + 3

        def in_triple(idx):
            for a, b in triple_spans:
                if a <= idx < b:
                    return True
            return False

        for m in assignment_re.finditer(text):
            if in_triple(m.start()):
                continue
            # skip if match is inside a comment on the same line
            lineno = text.count("\n", 0, m.start()) + 1
            line = text.splitlines()[lineno - 1]
            hash_idx = line.find('#')
            match_idx_in_line = m.start() - (text.rfind('\n', 0, m.start()) + 1)
            if hash_idx != -1 and hash_idx <= match_idx_in_line:
                continue
            violations.append((rel, lineno, line.strip()))

    assert not violations, (
        "Found ad-hoc assignments to args.<field> outside MARL/common/arguments.py or tests/:\n"
        + "\n".join([f"{p}:{ln}: {line}" for p, ln, line in violations])
    )


def test_no_mixed_args_and_self_args_in_same_file():
    """Fail if a single file contains both bare `args.` references and `self.args.` references.

    This helps enforce the Option-B convention: instance objects should use `self.args` and
    code that relies on getattr-style fallbacks may use `getattr(args, ...)` but mixing both
    in the same module is a hygiene smell.
    """
    bare_args_re = re.compile(r"(?<!self\.)\bargs\.[A-Za-z_][A-Za-z0-9_]*")
    self_args_re = re.compile(r"\bself\.args\.[A-Za-z_][A-Za-z0-9_]*")

    violations = []
    for path in iter_py_files(ROOT):
        rel = os.path.relpath(path, ROOT)
        # skip test files (they may intentionally reference args or build fake namespaces)
        if rel.startswith("tests"):
            continue
        # allow the canonical arguments file
        if rel == os.path.join("MARL", "common", "arguments.py"):
            continue
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        # find triple-quoted spans and ignore matches within them
        triple_spans = []
        for triple in ("'''", '"""'):
            start = 0
            while True:
                si = text.find(triple, start)
                if si == -1:
                    break
                sj = text.find(triple, si + 3)
                if sj == -1:
                    break
                triple_spans.append((si, sj + 3))
                start = sj + 3

        def in_triple(idx):
            for a, b in triple_spans:
                if a <= idx < b:
                    return True
            return False

        # find first bare and self args occurrences that are not inside triple strings or comments
        bare_m = None
        self_m = None
        for m in bare_args_re.finditer(text):
            if in_triple(m.start()):
                continue
            # skip if in-line comment
            lineno = text.count("\n", 0, m.start()) + 1
            line = text.splitlines()[lineno - 1]
            hash_idx = line.find('#')
            match_idx_in_line = m.start() - (text.rfind('\n', 0, m.start()) + 1)
            if hash_idx != -1 and hash_idx <= match_idx_in_line:
                continue
            bare_m = m
            break

        for m in self_args_re.finditer(text):
            if in_triple(m.start()):
                continue
            lineno = text.count("\n", 0, m.start()) + 1
            line = text.splitlines()[lineno - 1]
            hash_idx = line.find('#')
            match_idx_in_line = m.start() - (text.rfind('\n', 0, m.start()) + 1)
            if hash_idx != -1 and hash_idx <= match_idx_in_line:
                continue
            self_m = m
            break

        if bare_m and self_m:
            bare_ln = text.count("\n", 0, bare_m.start()) + 1
            self_ln = text.count("\n", 0, self_m.start()) + 1
            bare_line = text.splitlines()[bare_ln - 1].strip()
            self_line = text.splitlines()[self_ln - 1].strip()
            violations.append((rel, (bare_ln, bare_line), (self_ln, self_line)))

    assert not violations, (
        "Found mixed usage of `args.` and `self.args.` in the same file(s):\n"
        + "\n".join([
            f"{p}: bare args at {b_ln}: {b_line}\n    self.args at {s_ln}: {s_line}"
            for p, (b_ln, b_line), (s_ln, s_line) in violations
        ])
    )
