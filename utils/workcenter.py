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
from typing import Dict, List


# A single WorkCenter
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
