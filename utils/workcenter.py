"""
utils/workcenter.py — canonical WorkCenter & Machine registry
-----------------------------------------------------
This module provides a small, consistent WorkCenters/WorkCenter API and a
default topology that reflects the user's current deployment:

  - 3 WorkCenters (indices 0..2)
  - 5 machines distributed as:
      WorkCenter 0: Machine M_0_0 (machine 1), Machine M_0_1 (machine 2)
      WorkCenter 1: Machine M_1_0 (machine 3), Machine M_1_1 (machine 4)
      WorkCenter 2: Machine M_2_0 (machine 5)

Operators and other components should consult `workcenters_list` and
`machine_registry` for authoritative topology information. The old
"Site/Sites" terminology has been removed — use WorkCenter/WorkCenters.
"""

import random
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
        "M1": {"wc": "WC1", "capable_ops": ["Op1", "Op2", "Op3", "Op5", "Op9"], "speed_factor": 1.0},
        "M2": {"wc": "WC1", "capable_ops": ["Op4", "Op5", "Op8"], "speed_factor": 1.0},
        "M3": {"wc": "WC2", "capable_ops": ["Op1", "Op2", "Op4", "Op6", "Op9"], "speed_factor": 1.0},
        "M4": {"wc": "WC2", "capable_ops": ["Op1", "Op3", "Op7", "Op8"], "speed_factor": 1.0},
        "M5": {"wc": "WC3", "capable_ops": ["Op1", "Op3", "Op4", "Op6", "Op8", "Op9"], "speed_factor": 1.0},
    },
    "work_centers": {"WC1": {"id": 0}, "WC2": {"id": 1}, "WC3": {"id": 2}},
    # Operators are defined separately in utils.operator; keep a mirror here
    "operators": [
        {"id": "O1", "qualified_machines": ["M1", "M4", "M5"]},
        {"id": "O2", "qualified_machines": ["M2", "M3", "M5"]},
    ],
}
class WorkCenter:
    def __init__(self, wc_id: int, machine_names: List[str]):
        self.id = int(wc_id)
        # dictionary of machine_name -> metadata
        self.machines: Dict[str, Dict] = {}
        for mid in machine_names:
            # small random speed factor for variability (stable enough)
            self.machines[mid] = {
                "workcenter": int(wc_id),
                # default to supporting op types 0..8 unless overridden by config
                "capabilities": list(range(0, 9)),
                "speed_factor": round(random.uniform(0.9, 1.1), 3),
            }

    def machine_list(self) -> List[str]:
        return list(self.machines.keys())


# Container of WorkCenters and a global machine registry
class WorkCenters:
    def __init__(self):
        # Default topology (3 WCs, 5 machines) — this is the canonical mapping
        # used when no external YAML config overrides it.
        self.workcenters_list: List[WorkCenter] = []
        # machine_registry: machine_name -> {workcenter, capabilities, speed_factor}
        self.machine_registry: Dict[str, Dict] = {}

        # default machine naming consistent with the environment code
        # ordering corresponds to user-visible machine numbers 1..5
        default_map = {
            0: ["M_0_0", "M_0_1"],  # WorkCenter 1 -> machine 1,2
            1: ["M_1_0", "M_1_1"],  # WorkCenter 2 -> machine 3,4
            2: ["M_2_0"],            # WorkCenter 3 -> machine 5
        }

        for wc_idx in sorted(default_map.keys()):
            machines = default_map[wc_idx]
            wc = WorkCenter(wc_idx, machines)
            self.workcenters_list.append(wc)
            for mname, mmeta in wc.machines.items():
                # copy metadata into global registry
                self.machine_registry[mname] = mmeta.copy()

        # explicit helper: machine order -> machine name (1-based machine numbers)
        self.machine_order = ["M_0_0", "M_0_1", "M_1_0", "M_1_1", "M_2_0"]

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
        # Merge provided config into module-level defaults so callers may omit
        # sections. This keeps YAML and in-module DEFAULT_WORKCENTERS consistent.
        try:
            from utils.config_loader import merge_config  # local import to avoid cycles
            config = merge_config(DEFAULT_WORKCENTERS, config)
        except Exception:
            # if merge fails, continue with whatever config was passed
            pass

        inst = cls()
        num_wcs = 1
        num_ops = 1
        try:
            machines_cfg = config.get('machines', {}) or {}
            wc_cfg = config.get('work_centers', {}) or {}

            if wc_cfg:
                wc_names = list(wc_cfg.keys())
            else:
                seen = {}
                for mname, mconf in machines_cfg.items():
                    wcn = mconf.get('wc')
                    if wcn and wcn not in seen:
                        seen[wcn] = True
                wc_names = list(seen.keys())

            wc_name_to_idx = {name: idx for idx, name in enumerate(wc_names)}

            machine_registry = {}
            wc_to_machines = {idx: [] for idx in range(len(wc_names))}
            for mname, mconf in machines_cfg.items():
                wcn = mconf.get('wc')
                if wcn is None:
                    wci = len(wc_name_to_idx)
                    wc_name_to_idx[mname] = wci
                    wc_names.append(mname)
                    wc_to_machines[wci] = []
                wci = wc_name_to_idx.get(wcn, wc_name_to_idx.get(mname, 0))

                caps = mconf.get('capable_ops', []) or []
                caps_idx = []
                for c in caps:
                    try:
                        if isinstance(c, str) and c.lower().startswith('op'):
                            caps_idx.append(int(c[2:]) - 1)
                        else:
                            caps_idx.append(int(c))
                    except Exception:
                        pass

                machine_registry[mname] = {
                    'workcenter': int(wci),
                    'capabilities': caps_idx,
                    'speed_factor': float(mconf.get('speed_factor', 1.0)),
                }
                wc_to_machines[int(wci)].append(mname)

            eligible = {}
            ops_cfg = config.get('operators', []) or []
            for idx in range(len(wc_names)):
                eligible[idx] = []
                for op_idx, opconf in enumerate(ops_cfg):
                    q = opconf.get('qualified_machines', []) or []
                    for mname in wc_to_machines.get(idx, []):
                        if mname in q:
                            eligible[idx].append(op_idx)
                            break

            inst.machine_registry = machine_registry
            try:
                inst.machine_list = list(machine_registry.keys())
                inst.machine_index = {m: i for i, m in enumerate(inst.machine_list)}
            except Exception:
                inst.machine_list = list(machine_registry.keys())
                inst.machine_index = {m: i for i, m in enumerate(inst.machine_list)}

            # build WorkCenter objects grouping their machines
            try:
                wc_list = []
                for wc_idx in range(len(wc_names)):
                    machine_names = wc_to_machines.get(wc_idx, [])
                    wc_obj = WorkCenter(wc_idx, list(machine_names))
                    wc_obj.machines = {}
                    for mi, mname in enumerate(machine_names):
                        wc_obj.machines[mname] = machine_registry.get(mname, {}).copy()
                    wc_list.append(wc_obj)
                inst.workcenters_list = wc_list
            except Exception:
                pass

            try:
                inst.eligible_operator_groups_by_wc = eligible
            except Exception:
                pass

            num_wcs = max(1, len(wc_names))
            num_ops = max(1, len(ops_cfg))
        except Exception:
            num_wcs = getattr(inst, 'num_workcenters', lambda: 1)()
            num_ops = 1

        return inst, int(num_wcs), int(num_ops)
    # ------------------------------------------------------------------
    def create_decision_item(self, env: Any, job: Any, op: Any) -> Tuple[Dict, simpy.Event]:
        """Create a decision_item for an operation and a SimPy resume event.

        This method centralizes the logic that was previously embedded inside
        `environment._job_process`. It returns a tuple (decision_item, resume_evt)
        where `decision_item` is a dict compatible with existing Runner code
        and `resume_evt` is a `simpy.Event` that the job process will yield.
        """
        # create resume event on the env's simpy.Environment
        resume_evt = simpy.Event(env.env)

        # normalize op tuple formats
        op_type = None
        allowed_wcs = []
        per_wc_durations = None
        base_dur = None
        if isinstance(op, (list, tuple)):
            if len(op) == 2:
                allowed_wcs, base_dur = op
                op_type = None
            elif len(op) == 3:
                op_type = op[0]
                allowed_wcs = op[1]
                third = op[2]
                if isinstance(third, dict):
                    per_wc_durations = third
                else:
                    base_dur = float(third)
            else:
                try:
                    allowed_wcs, base_dur = op[0], op[1]
                except Exception:
                    allowed_wcs, base_dur = [], 0.0
        else:
            allowed_wcs, base_dur = [], 0.0

        allowed_machines = []
        allowed_machine_indices = []
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
                except Exception:
                    continue

            # extract processing_time_means from env.config if present
            if getattr(env, 'config', None):
                try:
                    proc_means = env.config.get('processing_time_means', {})
                    op_name = f"Op{op_idx_local+1}"
                    op_means = proc_means.get(op_name, {}) if isinstance(proc_means, dict) else {}
                    for m in allowed_machines:
                        if m in op_means:
                            per_machine_durations[int(mindex.get(m))] = float(op_means.get(m))
                except Exception:
                    pass

            # fallback: estimate per-machine durations using base and speed_factor
            for m in allowed_machines:
                mi = int(mindex.get(m, 0))
                if mi in per_machine_durations:
                    continue
                try:
                    speed = float(self.machine_registry.get(m, {}).get('speed_factor', 1.0))
                    # base_duration_val will be computed below; use placeholder if missing
                    per_machine_durations[mi] = float(0.0)
                except Exception:
                    per_machine_durations[mi] = float(0.0)
        except Exception:
            allowed_machines = []
            allowed_machine_indices = []
            per_machine_durations = {}

        # compute base_duration_val similar to environment logic
        if base_dur is not None:
            try:
                base_duration_val = float(base_dur)
            except Exception:
                base_duration_val = 0.0
        elif per_wc_durations:
            try:
                vals = [float(v) for v in per_wc_durations.values() if v is not None]
                base_duration_val = sum(vals) / len(vals) if vals else 0.0
            except Exception:
                base_duration_val = 0.0
        else:
            base_duration_val = 0.0

        # Fill in per_machine_durations that were placeholders
        try:
            mindex = getattr(self, 'machine_index', {})
            for m in allowed_machines:
                mi = int(mindex.get(m, 0))
                if mi in per_machine_durations and per_machine_durations[mi] == 0.0:
                    try:
                        speed = float(self.machine_registry.get(m, {}).get('speed_factor', 1.0))
                        per_machine_durations[mi] = round(float(base_duration_val) / max(1e-6, speed), 6)
                    except Exception:
                        per_machine_durations[mi] = float(base_duration_val)
        except Exception:
            pass

        # resume callable that will validate choice and succeed the resume_evt
        def _resume_with(choice, _resume_evt=resume_evt):
            try:
                mlist_local = getattr(self, 'machine_list', []) or []
                mreg = getattr(self, 'machine_registry', {}) or {}
                if isinstance(choice, str):
                    if choice not in mreg:
                        _resume_evt.succeed(None)
                        return
                    mi = int(getattr(self, 'machine_index', {}).get(choice, -1))
                else:
                    try:
                        c = int(choice)
                    except Exception:
                        _resume_evt.succeed(None)
                        return
                    if mlist_local and 0 <= c < len(mlist_local):
                        mi = int(c)
                    else:
                        if c in allowed_wcs:
                            mi = None
                            for mname, md in (mreg or {}).items():
                                try:
                                    if int(md.get('workcenter', -1)) == int(c):
                                        mi_candidate = int(getattr(self, 'machine_index', {}).get(mname, -1))
                                        mi = mi_candidate
                                        break
                                except Exception:
                                    continue
                            if mi is None:
                                _resume_evt.succeed(None)
                                return
                        else:
                            _resume_evt.succeed(None)
                            return

                allowed_inds = decision_item.get('allowed_machine_indices', [])
                if allowed_inds and (mi not in allowed_inds):
                    _resume_evt.succeed(None)
                    return
                _resume_evt.succeed(int(mi))
            except Exception:
                _resume_evt.succeed(None)
                return

        decision_item = {
            "job_id": getattr(job, 'id', getattr(job, 'agent_id', None)),
            "obs": env._build_agent_obs(job),
            "avail_row": env._avail_row_for_job(job),
            "allowed_wcs": list(allowed_wcs),
            "allowed_machines": list(allowed_machines),
            "allowed_machine_indices": list(allowed_machine_indices),
            "per_machine_durations": dict(per_machine_durations),
            "base_duration": float(base_duration_val),
            "resume": _resume_with,
            "eligible_ops_by_wc": {wc: getattr(self, 'eligible_operator_groups_by_wc', {}).get(int(wc), []) for wc in allowed_wcs},
        }

        return decision_item, resume_evt
