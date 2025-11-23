"""
utils/workcenter.py — canonical WorkCenter & Machine registry
-----------------------------------------------------
This module provides a small, consistent WorkCenters/WorkCenter API and a
default topology that reflects the user's current deployment:

  - 3 WorkCenters (indices 0..2)
  - 5 machines distributed as:
      WorkCenter 0: Machine M0 (machine 1), Machine M1 (machine 2)
      WorkCenter 1: Machine M2 (machine 3), Machine M3 (machine 4)
      WorkCenter 2: Machine M4 (machine 5)

Operators and other components should consult `workcenters_list` and
`machine_registry` for authoritative topology information. The old
"Site/Sites" terminology has been removed — use WorkCenter/WorkCenters.
"""

import random
import re
import simpy
from typing import Dict, List, Any, Tuple


# A single WorkCenter
# Default configuration mirror for WorkCenters (Phase 3C.0)
# Mirrors YAML keys: 'machines', 'work_centers', 'operators'
# Example structure: mapping of machine name -> metadata and work_center grouping
# TODO(Phase3C.1): integrate with WorkCenters.from_config() via merge_config(DEFAULT_WORKCENTERS, cfg)
DEFAULT_WORKCENTERS = {
    # Synchronized to configs/env_config_enabled.yaml so default topology
    # matches the canonical enabled config (3 work centers, 5 machines).
    "machines": {
    "M0": {"wc": "WC1", "capable_ops": ["Op1", "Op2", "Op3", "Op5", "Op9"]},
    "M1": {"wc": "WC1", "capable_ops": ["Op4", "Op5", "Op8"]},
    "M2": {"wc": "WC2", "capable_ops": ["Op1", "Op2", "Op4", "Op6", "Op9"]},
    "M3": {"wc": "WC2", "capable_ops": ["Op1", "Op3", "Op7", "Op8"]},
    "M4": {"wc": "WC3", "capable_ops": ["Op1", "Op3", "Op4", "Op6", "Op8", "Op9"]},
    },
    "work_centers": {"WC1": {"id": 0}, "WC2": {"id": 1}, "WC3": {"id": 2}},
    # Operators are defined separately in utils.operator; keep a mirror here
    "operators": [
        {"id": "O1", "qualified_machines": ["M1", "M4", "M5"]},
        {"id": "O2", "qualified_machines": ["M2", "M3", "M5"]},
    ],
}
# Optional internal fallback for processing times (machine -> {OpN: mean})
# VALUES ARE TRANSPOSED FROM configs/env_config_enabled.yaml and MUST MATCH
# those YAML values exactly so the fallback is deterministic and consistent
# with the canonical config. This is the single in-code fallback source for
# ⚠️ Keep DEFAULT_PROCESSING_TIMES in sync with configs/env_config_enabled.yaml
# Validated automatically by tests/test_processing_times_sync.py
DEFAULT_PROCESSING_TIMES = {
    "M0": {
        "Op1": 1.225,
        "Op2": 1.05,
        "Op3": 1.575,
        "Op4": 1.575,
        "Op5": 2.275,
        "Op8": 1.575,
        "Op9": 2.975,
    },
    "M1": {
        "Op4": 1.575,
        "Op5": 1.75,
        "Op8": 2.975,
    },
    "M2": {
        "Op1": 1.575,
        "Op2": 1.75,
        "Op3": 1.575,
        "Op4": 1.68,
        "Op6": 2.1,
        "Op7": 2.10,
        "Op8": 2.625,
        "Op9": 3.15,
    },
    "M3": {
        "Op1": 1.4,
        "Op3": 1.75,
        "Op7": 1.82,
        "Op8": 2.625,
    },
    "M4": {
        "Op1": 1.05,
        "Op3": 1.925,
        "Op4": 2.1,
        "Op6": 2.8,
        "Op8": 1.575,
        "Op9": 1.925,
    },
}
class WorkCenter:
    def __init__(self, wc_id: int, machine_names: List[str]):
        self.id = int(wc_id)
        # dictionary of machine_name -> metadata
        self.machines: Dict[str, Dict] = {}
        for mid in machine_names:
            # Prefer capability info from module-level
            # DEFAULT_WORKCENTERS (synchronized to YAML). If an entry exists
            # for this machine name, copy its metadata (normalizing capable
            # op names like 'Op1' -> index 0). Otherwise fall back to a
            # deterministic default (no randomness) so runs are reproducible.
            caps_idx = list(range(0, 9))
            default_machines = DEFAULT_WORKCENTERS.get('machines', {}) if isinstance(DEFAULT_WORKCENTERS, dict) else {}
            meta = default_machines.get(mid)
            # try mapping names like M_0_0 -> M0..M4 used in DEFAULT_WORKCENTERS
            if meta is None:
                m = re.match(r"M_(\d+)_(\d+)$", mid)
                if m:
                    a = int(m.group(1))
                    b = int(m.group(2))
                    idx = a * 2 + b
                    key = f"M{idx}"
                    meta = default_machines.get(key)

            if isinstance(meta, dict):
                # extract capabilities, support both 'capable_ops' (YAML) and 'capabilities'
                raw_caps = meta.get('capable_ops', None) or meta.get('capabilities', None)
                if isinstance(raw_caps, (list, tuple)):
                    caps_idx = []
                    for c in raw_caps:
                        try:
                            if isinstance(c, str) and c.lower().startswith('op'):
                                caps_idx.append(int(c[2:]) - 1)
                            else:
                                caps_idx.append(int(c))
                        except Exception:
                            # ignore unparsable entries
                            continue

            self.machines[mid] = {
                "workcenter": int(wc_id),
                "capabilities": caps_idx,
            }

    def machine_list(self) -> List[str]:
        return list(self.machines.keys())


# Container of WorkCenters and a global machine registry
class WorkCenters:
    def __init__(self):
        # Default topology (3 WCs, 5 machines) — this is the canonical mapping
        # used when no external YAML config overrides it.
        self.workcenters_list: List[WorkCenter] = []
        # machine_registry: machine_name -> {workcenter, capabilities}
        self.machine_registry: Dict[str, Dict] = {}

        # default machine naming consistent with the environment code
        # ordering corresponds to user-visible machine numbers 1..5
        default_map = {
            0: ["M0", "M1"],  # WorkCenter 1 -> machine 0,1
            1: ["M2", "M3"],  # WorkCenter 2 -> machine 2,3
            2: ["M4"],          # WorkCenter 3 -> machine 4
        }

        for wc_idx in sorted(default_map.keys()):
            machines = default_map[wc_idx]
            wc = WorkCenter(wc_idx, machines)
            self.workcenters_list.append(wc)
            for mname, mmeta in wc.machines.items():
                # copy metadata into global registry
                self.machine_registry[mname] = mmeta.copy()

        # explicit helper: machine order -> machine name (1-based machine numbers)
        self.machine_order = ["M0", "M1", "M2", "M3", "M4"]

        # operator-group eligibility mapping (by WorkCenter index)
        # default: each WorkCenter has its own operator-group membership list; this
        # can be overridden by environment.config if provided.
        self.eligible_operator_groups_by_wc: Dict[int, List[int]] = {
            0: [0],
            1: [1],
            2: [2],
        }

        # Build operation -> machines reverse map for quick lookups. Each
        # operation index maps to the list of machine registry keys that
        # support that operation. This helps rollout and analysis code.
        self.operations_map: Dict[int, List[str]] = {}
        for mname, mmeta in self.machine_registry.items():
            for op in mmeta.get('capabilities', []):
                self.operations_map.setdefault(int(op), []).append(mname)
        
    @property
    def machine_list(self):
        """Alias for machine_order to maintain compatibility with environment code."""
        return self.machine_order

    # Utilities
    def machine_name_for_number(self, n: int) -> str:
        """Return machine name for 1-based machine number (1..5)."""
        if 1 <= n <= len(self.machine_order):
            return self.machine_order[n - 1]
        raise IndexError("machine number out of range")

    def workcenter_for_machine(self, machine_name: str) -> int:
        return int(self.machine_registry.get(machine_name, {}).get("workcenter", 0))

    def num_workcenters(self) -> int:
        return len(self.workcenters_list)

    def num_machines(self) -> int:
        return len(self.machine_order)

    @classmethod
    def from_config(cls, config: dict, strict_mode: bool = True) -> Tuple['WorkCenters', int, int]:
        """Factory to build a WorkCenters instance from a parsed config dict.

        Returns (workcenters_meta, num_wcs, num_ops).
        """
        # Deprecated: config-driven construction. Always return a WorkCenters
        # instance built from in-module defaults so the system is config-free.
        inst = cls()
        try:
            num_wcs = int(len(getattr(inst, 'workcenters_list', []) or []))
        except Exception:
            num_wcs = 1
        # Derive num_ops from operations_map (max op index + 1) when possible
        try:
            ops_keys = list(getattr(inst, 'operations_map', {}).keys())
            if ops_keys:
                num_ops = int(max(ops_keys) + 1)
            else:
                num_ops = 1
        except Exception:
            num_ops = 1
        return inst, int(num_wcs), int(num_ops)
    # ------------------------------------------------------------------
    def create_decision_item(self, env: Any, job: Any, op: Any) -> Dict:
        """Create a canonical decision_item describing an operation.

        Returns a dict containing at minimum the keys:
            - job_id
            - obs
            - avail_row
            - allowed_machine_indices (list of machine indices)
            - per_machine_durations (dict machine_index -> duration)
            - base_duration

        Note: this function does NOT create or return a resume Event. The
        environment is responsible for creating a `simpy.Event` and attaching
        it to the returned decision_item under the key `resume_evt` before
        presenting the item to the policy/runner. This keeps the resume Event
        ownership in the env (single source of truth for SimPy events).
        """

        # normalize op tuple formats
        op_type = None
        allowed_machine_indices = []
        per_wc_durations = None
        base_dur = None
        if isinstance(op, (list, tuple)):
            if len(op) == 2:
                allowed_machine_indices, base_dur = op
                op_type = None
            elif len(op) == 3:
                op_type = op[0]
                allowed_machine_indices = op[1]
                third = op[2]
                if isinstance(third, dict):
                    per_wc_durations = third
                else:
                    base_dur = float(third)
            else:
                try:
                    allowed_machine_indices, base_dur = op[0], op[1]
                except Exception:
                    allowed_machine_indices, base_dur = [], 0.0
        else:
            allowed_machine_indices, base_dur = [], 0.0

        allowed_machines = []
        # allowed_machine_indices may have been provided in the op tuple; if
        # not, we'll build it from the registry below.
        allowed_machine_indices = list(allowed_machine_indices) if isinstance(allowed_machine_indices, (list, tuple)) else []
        per_machine_durations = {}

        try:
            op_idx_local = int(op_type) if (op_type is not None) else int(getattr(job, 'current_op_idx', 0))
            mlist = list(getattr(self, 'machine_list', []))
            mindex = getattr(self, 'machine_index', {})
            for mname in mlist:
                try:
                    mreg = self.machine_registry.get(mname, {})
                    caps = mreg.get('capabilities', [])
                    if op_idx_local in caps:
                        allowed_machines.append(mname)
                        allowed_machine_indices.append(int(mindex.get(mname, len(allowed_machine_indices))))
                except Exception as e:
                    logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
                    continue

            # Use DEFAULT_PROCESSING_TIMES directly - no config dependency
            op_name = f"Op{op_idx_local+1}"
            for m in allowed_machines:
                mi = int(mindex.get(m, 0))
                if m in DEFAULT_PROCESSING_TIMES and op_name in DEFAULT_PROCESSING_TIMES[m]:
                    per_machine_durations[mi] = float(DEFAULT_PROCESSING_TIMES[m][op_name])
        except Exception as e:
            logging.getLogger(__name__).warning(f"[create_decision_item] Failed to populate per_machine_durations: {e}")
            allowed_machines = []
            allowed_machine_indices = []
            per_machine_durations = {}

        # In strict mode, we expect per_machine_durations to have been
        # populated from processing_time_means. For compatibility expose a
        # base_duration as the mean of per-machine durations when present.
        if per_machine_durations:
            try:
                vals = [float(v) for v in per_machine_durations.values() if v is not None]
                base_duration_val = sum(vals) / len(vals) if vals else 0.0
            except Exception:
                base_duration_val = 0.0
        else:
            base_duration_val = 0.0

        # NOTE: Decision validation and resume Event completion is the
        # responsibility of the environment/runner that owns the SimPy Event.
        # This function builds the canonical per-operation metadata only.

        # Build eligible operator-groups mapping keyed by workcenter index for
        # workcenters that own at least one allowed machine.
        eligible_ops_by_wc = {}
        try:
            for m in allowed_machines:
                try:
                    wc_idx = int(self.machine_registry.get(m, {}).get('workcenter', 0))
                    eligible_ops_by_wc[wc_idx] = getattr(self, 'eligible_operator_groups_by_wc', {}).get(int(wc_idx), [])
                except Exception as e:
                    logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
                    continue
        except Exception:
            eligible_ops_by_wc = {}

        decision_item = {
            "job_id": getattr(job, 'id', getattr(job, 'agent_id', None)),
            "obs": env._build_agent_obs(job),
            "avail_row": env._avail_row_for_job(job),
            "allowed_machines": list(allowed_machines),
            "allowed_machine_indices": list(dict.fromkeys(allowed_machine_indices)),
            "per_machine_durations": dict(per_machine_durations),
            "base_duration": float(base_duration_val),
            "eligible_ops_by_wc": eligible_ops_by_wc,
        }

        return decision_item
