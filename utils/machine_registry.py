# Deprecated: merged into utils/workcenter.py (Phase 4A.1)

"""Deprecated compatibility module.

The implementation was moved to `utils.workcenter.WorkCenters.from_config` and
the full original implementation is preserved in
`archive/legacy/machine_registry.py`.

Importing this module will raise ImportError to encourage callers to switch
to the canonical API. If you need the archived implementation, see
`archive/legacy/machine_registry.py`.
"""
from typing import Any

raise ImportError(
    "utils.machine_registry is deprecated and has been merged into utils.workcenter.WorkCenters.from_config. "
    "See archive/legacy/machine_registry.py for the archived copy."
)
"""utils/machine_registry.py
---------------------------------
Small helper to construct a WorkCenters metadata object (machine registry,
workcenter grouping, eligible operator groups) from a parsed config dict.

API:
  build_machine_registry(config: dict, strict_mode: bool=True)
    -> (workcenters_meta, num_wcs, num_ops)

This function extracts the config handling currently embedded in
`MASAEnv.__init__` and returns an instance of `WorkCenters` from
`utils.workcenter` with machine_registry, machine_list and other
attributes populated.
"""
from typing import Tuple, Dict, Any

try:
    from utils.workcenter import WorkCenters, WorkCenter
except Exception:
    # minimal fallback if workcenter module not importable
    class WorkCenters:
        def __init__(self):
            self.machine_registry = {}
            self.machine_list = []
            self.machine_index = {}
            self.workcenters_list = []
            self.eligible_operator_groups_by_wc = {}


def build_machine_registry(config: Dict[str, Any], strict_mode: bool = True) -> Tuple[WorkCenters, int, int]:
    """Build and return (workcenters_meta, num_wcs, num_ops).

    The function tolerantly parses config and attempts to build the same
    structures the environment previously created inline. It is defensive and
    returns reasonable defaults on error to avoid breaking callers.
    """
    wc_meta = WorkCenters()
    num_wcs = None
    num_ops = getattr(wc_meta, 'num_ops', 1) if hasattr(wc_meta, 'num_ops') else 1

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
        wc_to_machines: Dict[int, list] = {idx: [] for idx in range(len(wc_names))}
        for mname, mconf in machines_cfg.items():
            wcn = mconf.get('wc')
            if wcn is None:
                # assign to its own new WC bucket deterministically
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

        # operators -> eligible groups per WC
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

        # populate wc_meta
        wc_meta.machine_registry = machine_registry
        try:
            wc_meta.machine_list = list(machine_registry.keys())
            wc_meta.machine_index = {m: i for i, m in enumerate(wc_meta.machine_list)}
        except Exception:
            wc_meta.machine_list = list(machine_registry.keys())
            wc_meta.machine_index = {m: i for i, m in enumerate(wc_meta.machine_list)}

        # create WorkCenter objects grouping their machines (if WorkCenter type exists)
        try:
            wc_list = []
            for wc_idx in range(len(wc_names)):
                machine_names = wc_to_machines.get(wc_idx, [])
                # aggregate capabilities
                caps_set = set()
                for mname in machine_names:
                    caps_set.update(machine_registry.get(mname, {}).get('capabilities', []))
                wc_obj = WorkCenter(wc_idx, list(machine_names))
                # replace default machines with actual metadata
                wc_obj.machines = {}
                for mi, mname in enumerate(machine_names):
                    wc_obj.machines[mname] = machine_registry.get(mname, {}).copy()
                wc_list.append(wc_obj)
            wc_meta.workcenters_list = wc_list
        except Exception:
            pass

        try:
            wc_meta.eligible_operator_groups_by_wc = eligible
        except Exception:
            pass

        num_wcs = max(1, len(wc_names))
        num_ops = max(1, len(ops_cfg))
    except Exception:
        # on any error, return the default WorkCenters instance and best-effort sizes
        try:
            num_wcs = max(1, num_wcs or 1)
        except Exception:
            num_wcs = 1
        num_ops = max(1, num_ops or 1)

    return wc_meta, int(num_wcs), int(num_ops)
