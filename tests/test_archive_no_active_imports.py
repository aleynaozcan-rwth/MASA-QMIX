import ast
from pathlib import Path


def _is_in_archive(path: Path) -> bool:
    return any(part == "archive" for part in path.parts)


def test_no_active_imports_from_archive():
    """Fail if any active (non-archive, non-tests) .py file imports from or
    references the `archive` package or `archive/legacy` path.

    This enforces that archived code remains read-only reference material and
    isn't brought into the active runtime by accidental imports.
    """
    repo_root = Path(__file__).resolve().parents[1]
    offenders = []

    for p in repo_root.rglob("*.py"):
        # skip files inside the archive itself and skip test files
        if _is_in_archive(p) or "tests" in p.parts:
            continue

        try:
            src = p.read_text()
        except Exception:
            # if a file can't be read, skip it (won't cause false positives)
            continue

        # quick substring check for any dynamic import-like references
        if "archive/legacy" in src or "archive.legacy" in src:
            offenders.append(f"{p}: contains literal 'archive/legacy' or 'archive.legacy'")
            continue

        try:
            tree = ast.parse(src)
        except SyntaxError:
            # skip files that are not parseable (generated code, etc.)
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_mod = alias.name.split(".")[0]
                    if root_mod == "archive":
                        offenders.append(f"{p}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                root_mod = mod.split(".")[0] if mod else ""
                if root_mod == "archive":
                    offenders.append(f"{p}: from {mod} import ...")

    assert not offenders, (
        "Found active source files importing or referencing archived paths:\n"
        + "\n".join(offenders)
    )
