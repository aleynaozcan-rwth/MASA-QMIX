#!/usr/bin/env python3
"""
Generate a module dependency audit for the repository.

Outputs: docs/module_dependency_audit.md

Heuristic rules:
- Parse imports with ast.Import/ImportFrom
- Map module names to repository files by matching candidate module paths
- Build reachable set from given entrypoints
- For inactive modules, tag based on presence of defs and name collisions

"""
import ast
import os
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'module_dependency_audit.md'

ENTRYPOINTS = [
    ROOT / 'main.py',
    ROOT / 'scripts' / 'run_train_qmix.py',
    ROOT / 'tools' / 'run_debug_smoke.py',
]

def iter_py_files(root):
    for p in root.rglob('*.py'):
        if '/__pycache__/' in str(p) or p.match('**/__pycache__/**'):
            continue
        if p.parts and (p.parts[0] == '.venv' or p.parts[0].startswith('.')):
            continue
        yield p

def parse_imports(path):
    try:
        src = path.read_text()
        tree = ast.parse(src)
    except Exception as e:
        logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
        return []
    mods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                mods.append(n.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                base = node.module
                mods.append(base)
    return mods

def defs_in_file(path):
    try:
        src = path.read_text()
        tree = ast.parse(src)
    except Exception as e:
        logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
        return {'functions': [], 'classes': []}
    funcs = []
    classes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            funcs.append(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
    return {'functions': funcs, 'classes': classes}

def build_module_map(py_files):
    # map from module dotted-name to path
    mod_map = {}
    for p in py_files:
        rel = p.relative_to(ROOT)
        parts = list(rel.with_suffix('').parts)
        # possible module name
        mod = '.'.join(parts)
        mod_map[mod] = p
        # also allow top-level filename without package
        if len(parts) == 1:
            mod_map[parts[0]] = p
    return mod_map

def resolve_module(mod_name, mod_map):
    # find best match: longest prefix in mod_map
    candidates = [(k, v) for k, v in mod_map.items() if mod_name == k or mod_name.startswith(k + '.')]
    if not candidates:
        return None
    # choose longest key
    candidates.sort(key=lambda kv: len(kv[0]), reverse=True)
    return candidates[0][1]

def gather_graph(entrypoints, mod_map):
    # BFS
    visited = set()
    imports_of = {}
    stack = []
    for ep in entrypoints:
        if not ep.exists():
            continue
        stack.append(ep)
        visited.add(ep)
    while stack:
        p = stack.pop()
        key = str(p.relative_to(ROOT))
        imports = parse_imports(p)
        resolved = []
        for m in imports:
            r = resolve_module(m, mod_map)
            if r is not None:
                resolved.append(str(r.relative_to(ROOT)))
                if r not in visited:
                    visited.add(r)
                    stack.append(r)
        imports_of[str(p.relative_to(ROOT))] = resolved
    return visited, imports_of

def find_references(name, all_files):
    refs = []
    for p in all_files:
        try:
            txt = p.read_text(errors='ignore')
        except Exception as e:
            logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
            continue
        if name in txt:
            refs.append(str(p.relative_to(ROOT)))
    return refs

def main():
    py_files = list(iter_py_files(ROOT))
    mod_map = build_module_map(py_files)

    # include any scripts in tools/ and scripts/
    extra_eps = []
    for d in ('tools', 'scripts'):
        dp = ROOT / d
        if dp.exists():
            for p in dp.glob('*.py'):
                extra_eps.append(p)

    eps = [e for e in ENTRYPOINTS if e.exists()] + extra_eps

    visited, imports_of = gather_graph(eps, mod_map)

    active = set(str(p.relative_to(ROOT)) for p in visited)
    all_py = set(str(p.relative_to(ROOT)) for p in py_files)
    inactive = sorted(all_py - active)

    # prepare report
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w') as fh:
        fh.write('# Module dependency audit\n\n')
        fh.write('=== Active Call Graph ===\n')
        for a in sorted(active):
            fh.write(f"{a} -> imports:\n")
            for imp in imports_of.get(a, []):
                fh.write(f"  - {imp}\n")
            # list defs and whether referenced elsewhere
            p = ROOT / a
            defs = defs_in_file(p)
            if defs['functions'] or defs['classes']:
                fh.write('  defines:\n')
                for fn in defs['functions']:
                    refs = find_references(fn, py_files)
                    used = any(r != a for r in refs)
                    fh.write(f"    - func {fn}  (used elsewhere: {'yes' if used else 'no'})\n")
                for cn in defs['classes']:
                    refs = find_references(cn, py_files)
                    used = any(r != a for r in refs)
                    fh.write(f"    - class {cn} (used elsewhere: {'yes' if used else 'no'})\n")
            fh.write('\n')

        fh.write('=== Inactive / Unused Files ===\n')
        for a in inactive:
            p = ROOT / a
            defs = defs_in_file(p)
            tag = '🔴 Obsolete'
            if defs['functions'] or defs['classes']:
                # detect duplicate names
                dup = False
                for other in active:
                    op = ROOT / other
                    od = defs_in_file(op)
                    if set(od['functions']) & set(defs['functions']) or set(od['classes']) & set(defs['classes']):
                        dup = True
                        break
                if dup:
                    tag = '🟡 Duplicate functionality'
                else:
                    tag = '🟢 Candidate for reintegration'
            fh.write(f"{a} -> {tag}\n")
            if defs['functions'] or defs['classes']:
                fh.write('  defines:\n')
                for fn in defs['functions']:
                    fh.write(f"    - func {fn}\n")
                for cn in defs['classes']:
                    fh.write(f"    - class {cn}\n")
            fh.write('\n')

    print('Wrote', OUT)

if __name__ == '__main__':
    main()
