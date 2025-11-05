#!/usr/bin/env python3
"""Normalize legacy scheduling_timeline.txt entries.

This script performs an in-place normalization of operator tokens in
`scheduling_timeline.txt` converting numeric-only operator mentions like
"| operator=0" and "by Operator 0" into the string-labeled form
"by Operator O1" (1-based operator labels). It keeps a timestamped
backup of the original file before writing the normalized content.

Usage:
  python3 tools/normalize_timeline.py [path/to/scheduling_timeline.txt]

If no path is provided, defaults to my_data_and_graph/historydata/scheduling_timeline.txt.
"""
import re
import sys
from pathlib import Path
from datetime import datetime


DEFAULT_PATH = Path('my_data_and_graph') / 'historydata' / 'scheduling_timeline.txt'


def normalize_text(text: str) -> str:
    # Pattern: started on <machine> (duration=X) | operator=N | eligible=...
    # Replace with: started on <machine> by Operator O{N+1} (duration=X) | eligible=...
    text = re.sub(
        r"(started on\s+[^\(]+\(duration=)([0-9]+\.?[0-9]*)(\)\s*\|\s*operator=)(\d+)",
        lambda m: f"{m.group(1)}{m.group(2)}) by Operator O{int(m.group(4))+1}",
        text,
    )

    # Pattern where operator=N appears without trailing eligible part
    text = re.sub(
        r"(started on\s+[^\(]+\(duration=)([0-9]+\.?[0-9]*)(\)\s*\|\s*operator=)(\d+)\s*$",
        lambda m: f"{m.group(1)}{m.group(2)}) by Operator O{int(m.group(4))+1}",
        text,
        flags=re.M,
    )

    # Pattern: finished → next queued | operator=N  => finished → next queued by Operator O{N+1}
    text = re.sub(
        r"(finished\s*→\s*next\s*queued)\s*\|\s*operator=(\d+)",
        lambda m: f"{m.group(1)} by Operator O{int(m.group(2))+1}",
        text,
    )

    # Pattern: by Operator N (duration=...) -> by Operator O{N+1}
    # but avoid converting 'O' already present (O1 etc.)
    text = re.sub(
        r"by Operator\s+([0-9]+)(?![0-9]*[A-Za-z])",
        lambda m: f"by Operator O{int(m.group(1))+1}",
        text,
    )

    # Pattern: operator=N in other contexts -> replace with Operator O{N+1}
    text = re.sub(
        r"\boperator=(\d+)\b",
        lambda m: f"operator=O{int(m.group(1))+1}",
        text,
    )

    return text


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    if not path.exists():
        print(f"No timeline file found at {path}. Nothing to normalize.")
        return 0

    # Read original content and perform normalization first. Write to a
    # temporary file and only after successful write create a timestamped
    # backup of the original and atomically replace it. This avoids a
    # situation where the original is renamed away but the normalized
    # write fails leaving only a .bak file on disk.
    content = path.read_text(encoding='utf-8')
    normalized = normalize_text(content)

    tmp = path.with_suffix(path.suffix + f'.normalized.tmp')
    try:
        tmp.write_text(normalized, encoding='utf-8')
    except Exception as e:
        print(f"Failed to write normalized temporary file {tmp}: {e}")
        # Do not touch the original file; exit with non-zero code
        return 2

    # Backup original and atomically replace with normalized content
    bak = path.with_suffix(path.suffix + f'.bak.{int(datetime.utcnow().timestamp())}')
    try:
        path.replace(bak)
        # Move tmp into place (replace semantics)
        tmp.replace(path)
        print(f"Backed up original timeline to: {bak}")
        print(f"Wrote normalized timeline to: {path}")
    except Exception as e:
        print(f"Failed to install normalized timeline atomically: {e}")
        # Attempt cleanup of tmp if still present
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return 3

    # Report diff-ish summary: counts before/after of numeric operator occurrences
    before_nums = len(re.findall(r"\boperator=\d+\b|by Operator\s+\d+", content))
    after_nums = len(re.findall(r"\boperator=\d+\b|by Operator\s+\d+", normalized))
    print(f"Replaced {before_nums - after_nums} numeric operator occurrences.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
