"""Centralized IO control for gating history / artifact writes.

Tools and writers should call `allow_history_writes()` before producing
files that are considered historical artifacts. This allows a single
runtime switch (environment variable or programmatic set) to disable
accidental writing in CI, tests, or read-only contexts.

Usage:
    from utils.io_control import allow_history_writes, set_allow_history_writes
    if allow_history_writes():
        ... write files ...
    # or set programmatically at start of process:
    set_allow_history_writes(True)
"""
from typing import Optional
import os

# module-level cache; default disabled to be conservative
_ALLOW: Optional[bool] = None


def _env_default() -> bool:
    v = os.environ.get('MASA_ALLOW_HISTORY_WRITES', '')
    if not v:
        return False
    return v.strip().lower() in ('1', 'true', 'yes', 'on')


def allow_history_writes() -> bool:
    """Return whether historical/artifact writes are allowed in this process.

    Priority: explicit set via `set_allow_history_writes` (module cache),
    otherwise determined from the `MASA_ALLOW_HISTORY_WRITES` environment
    variable (truthy values: 1,true,yes,on). Defaults to False.
    """
    global _ALLOW
    if _ALLOW is not None:
        return bool(_ALLOW)
    # determine from env
    _ALLOW = _env_default()
    return bool(_ALLOW)


def set_allow_history_writes(val: bool) -> None:
    """Programmatically enable or disable history writes for this process."""
    global _ALLOW
    _ALLOW = bool(val)
